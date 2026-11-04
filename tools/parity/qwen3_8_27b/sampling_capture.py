"""Independent FP64 oracle for eager, represented-input sampling captures.

This evaluates the public p-less law and Algorithm 2, not GPU reduction trees.
A mismatch is diagnostic evidence requiring FP32 boundary review, not automatic
proof that an EOS selected by the model was erroneous.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import sys
from pathlib import Path

MASK64 = (1 << 64) - 1


def uniform(seed: int, position: int, purpose: int, sub: int = 0) -> float:
    def mix(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    value = mix(seed ^ (((position & 0xFFFFFFFF) * 0xD1B54A32D192ED03) & MASK64))
    value = mix(value ^ ((purpose & 0xFFFFFFFF) << 21) ^ ((sub * 0x2545F4914F6CDD1D) & MASK64))
    return (value >> 40) / 16777216.0


def law(
    logits: list[float], config: dict, mask: list[int] | None, column: int
) -> tuple[dict[int, float], dict]:
    if config.get("presence_penalty", 0) != 0 or config.get("frequency_penalty", 0) != 0:
        raise ValueError(
            "nonzero-penalty captures require uncaptured token counts and are unsupported"
        )
    suppressed = set(config["suppressed_tokens"])
    allowed = [
        i
        for i in range(len(logits))
        if i not in suppressed and (mask is None or (mask[i // 32] >> (i % 32)) & 1)
    ]
    if not allowed or any(not math.isfinite(logits[i]) for i in allowed):
        raise ValueError("eligible domain must be nonempty and finite")
    order = sorted(allowed, key=lambda i: (-logits[i], i))
    mode = order[0]
    temperature = config["temperature"]
    if temperature <= 0:
        return {mode: 1.0}, {"mode": mode, "eligible_count": len(allowed), "fallback": "greedy"}
    if not config["p_less"]:
        raise ValueError("positive-temperature non-p-less captures are unsupported")
    exclude = config["typical_exclude"] if column == 0 else -1
    weights = {i: math.exp((logits[i] - logits[mode]) / temperature) for i in allowed}
    total = math.fsum(weights.values())
    probabilities = {i: weight / total for i, weight in weights.items()}
    collision = math.fsum(p * p for p in probabilities.values())
    threshold = max(collision * math.exp(-0.125 / temperature), 1.0 / 1024.0)
    support_before_exclusion = [i for i in allowed if probabilities[i] >= threshold]
    retained = [i for i in support_before_exclusion if i != exclude]
    fallback = None
    if not retained:
        token = order[1] if mode == exclude and len(order) > 1 else mode
        distribution = {token: 1.0}
        fallback = "excluded_singleton" if support_before_exclusion else "empty_membership_support"
    else:
        mass = math.fsum(probabilities[i] for i in retained)
        distribution = {i: probabilities[i] / mass for i in retained}
    detail = {
        "mode": mode,
        "runner_up": order[1] if len(order) > 1 else None,
        "eligible_count": len(allowed),
        "collision": collision,
        "threshold": threshold,
        "support_size_before_exclusion": len(support_before_exclusion),
        "support_size": len(distribution),
        "typical_exclude": exclude,
        "fallback": fallback,
        "minimum_membership_margin": min(abs(p - threshold) for p in probabilities.values()),
    }
    return distribution, detail


def inverse_cdf(distribution: dict[int, float], draw: float) -> tuple[int, dict]:
    positive = sorted((token, weight) for token, weight in distribution.items() if weight > 0)
    if not positive:
        raise ValueError("cannot draw from an empty distribution")
    total = math.fsum(weight for _, weight in positive)
    goal = draw * total
    cumulative = 0.0
    picked = positive[-1][0]
    lower, upper = 0.0, total
    for token, weight in positive:
        next_cumulative = cumulative + weight
        if goal < next_cumulative:
            picked, lower, upper = token, cumulative, next_cumulative
            break
        cumulative = next_cumulative
    return picked, {
        "uniform": draw,
        "mass": total,
        "goal": goal,
        "selected_interval": [lower, upper],
        "cdf_margin": min(abs(goal - lower), abs(upper - goal)) / total,
    }


def verify_chain(
    laws: list[dict[int, float]],
    drafts: list[int],
    proposals: list[dict[int, float]],
    seed: int,
    frontier: int,
) -> dict:
    extent = len(drafts)
    if len(laws) != extent + 1 or len(proposals) != extent:
        raise ValueError("chain column/proposal extent mismatch")
    tau, tau_weight, previous = 0, 1.0, 1.0
    hops = []
    for i in range(1, extent + 1):
        draft = drafts[i - 1]
        qd = proposals[i - 1].get(draft, 0.0)
        if not qd > 0:
            raise ValueError("recorded draft is outside its proposal support")
        weight = min(previous * laws[i - 1].get(draft, 0.0) / qd, 1.0)
        residual_mass = None
        h = weight
        if i < extent:
            # Complete positive part, independent of the GPU's covered-mass shortcut.
            residual_mass = math.fsum(
                max(weight * p - proposals[i].get(token, 0.0), 0.0) for token, p in laws[i].items()
            )
            denominator = residual_mass + 1.0 - weight
            h = residual_mass / denominator if denominator > 0 else 1.0
        eta = uniform(seed, frontier + 1, 5, i)
        hops.append(
            {
                "hop": i,
                "prefix_weight": weight,
                "residual_mass": residual_mass,
                "acceptance_threshold": h,
                "uniform": eta,
                "margin": abs(eta - h),
            }
        )
        if eta < h:
            tau, tau_weight = i, weight
        previous = weight
    purpose = 3 if tau < extent else 4
    distribution = laws[tau]
    residual_fallback = False
    if tau < extent:
        residual = {
            token: max(tau_weight * p - proposals[tau].get(token, 0.0), 0.0)
            for token, p in laws[tau].items()
        }
        if math.fsum(residual.values()) > 0:
            distribution = residual
        else:
            residual_fallback = True
    selected, cdf = inverse_cdf(distribution, uniform(seed, frontier + tau + 1, purpose))
    return {
        "accepted_drafts": tau,
        "licensed_tokens": [*drafts[:tau], selected],
        "correction_column": tau,
        "correction_purpose": purpose,
        "residual_fallback": residual_fallback,
        "acceptance": hops,
        "draw": cdf,
    }


def load_array(path: Path, typecode: str, count: int) -> list:
    values = array.array(typecode)
    raw = path.read_bytes()
    if len(raw) != count * values.itemsize:
        raise ValueError(f"wrong binary size: {path}")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def verify_capture(path: Path) -> dict:
    capture = json.loads(path.read_text())
    if (
        capture.get("artifact_type") != "ninfer_sampling_decision_capture"
        or capture.get("schema_version") != 1
    ):
        raise ValueError("unsupported sampling capture schema")
    if capture["kind"] not in {"ordinary", "dflash_chain"}:
        raise ValueError("only ordinary and chain verification captures are supported")
    columns, vocab = capture["columns"], capture["token_domain"]
    if (
        columns < 1
        or vocab < 1
        or capture["logits"]["dtype"] != "bf16_le"
        or capture["logits"]["shape"] != [columns, vocab]
    ):
        raise ValueError("invalid logit dtype/shape")
    bits = load_array(path.parent / capture["logits"]["file"], "H", columns * vocab)
    words = array.array("I", (b << 16 for b in bits))
    floats = array.array("f")
    floats.frombytes(words.tobytes())
    masks = None
    words_per_mask = (vocab + 31) // 32
    if capture["allowed_token_words"] is not None:
        descriptor = capture["allowed_token_words"]
        if descriptor["dtype"] != "u32_le" or descriptor["shape"] != [columns, words_per_mask]:
            raise ValueError("invalid mask dtype/shape")
        masks = load_array(path.parent / descriptor["file"], "I", columns * words_per_mask)
    config = capture["config"]
    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed <= MASK64:
        raise ValueError("seed must be an exact uint64 integer")
    distributions, details = [], []
    for col in range(columns):
        values = floats[col * vocab : (col + 1) * vocab].tolist()
        mask = None if masks is None else masks[col * words_per_mask : (col + 1) * words_per_mask]
        distribution, detail = law(values, config, mask, col)
        detail["column"] = col
        # Raw unmasked T1 logprob is deliberately labeled separately from sampling probability.
        finite_raw = all(math.isfinite(x) for x in values)
        maximum = max(values) if finite_raw else 0.0
        logsum = math.log(math.fsum(math.exp(x - maximum) for x in values)) if finite_raw else None
        detail["watched_tokens"] = [
            {
                "token": token,
                "logit": values[token],
                "target_sampling_probability": distribution.get(token, 0.0),
                "raw_t1_logprob": values[token] - maximum - logsum if logsum is not None else None,
            }
            for token in capture["watch_tokens"]
            if 0 <= token < vocab
        ]
        distributions.append(distribution)
        details.append(detail)
    actual = capture["licensed_tokens"]
    if capture["kind"] == "ordinary":
        if columns != 1 or len(actual) != 1:
            raise ValueError("ordinary capture must have one column and token")
        token, cdf = inverse_cdf(
            distributions[0], uniform(seed, capture["logical_position"], capture["rng_purpose"])
        )
        predicted = {"licensed_tokens": [token], "draw": cdf}
    elif config["temperature"] <= 0:
        drafts = capture["drafts"]
        tau = 0
        while tau < len(drafts) and drafts[tau] in distributions[tau]:
            tau += 1
        predicted = {
            "accepted_drafts": tau,
            "licensed_tokens": [*drafts[:tau], next(iter(distributions[tau]))],
        }
    else:
        if capture["proposal_extent"] != len(capture["drafts"]):
            raise ValueError("wrong draft extent")
        proposals = []
        for ids, probabilities in zip(capture["selector_ids"], capture["selector_q"], strict=True):
            if len(ids) != len(probabilities) or len(set(ids)) != len(ids):
                raise ValueError("selector IDs must be distinct with one probability each")
            if (
                any(not math.isfinite(p) or p < 0 for p in probabilities)
                or abs(math.fsum(probabilities) - 1) > 2e-5
            ):
                raise ValueError("invalid represented proposal probability law")
            proposals.append(dict(zip(ids, probabilities, strict=True)))
        predicted = verify_chain(
            distributions, capture["drafts"], proposals, seed, capture["base_frontier"]
        )
    support = [
        {
            "slot": i,
            "token": token,
            "column": i,
            "target_sampling_probability": distributions[i].get(token, 0.0),
        }
        for i, token in enumerate(actual)
    ]
    return {
        "artifact_type": "ninfer_sampling_oracle_report",
        "schema_version": 1,
        "capture": str(path),
        "oracle": "FP64 full represented-input p-less/Algorithm2",
        "columns": details,
        "licensed_support": support,
        "actual_licensed_tokens": actual,
        "predicted": predicted,
        "tokens_match": actual == predicted["licensed_tokens"],
        "accepted_drafts_match": capture.get("accepted_drafts") == predicted.get("accepted_drafts"),
        "limitation": "FP32 reduction, transcendental and CDF association can change near-boundary decisions. Inspect membership, acceptance and CDF margins before attributing a mismatch. This is an eager Op-licensed capture, not proof of publication or historical graph reproduction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_capture(args.capture)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "tokens_match": report["tokens_match"],
                "accepted_drafts_match": report["accepted_drafts_match"],
            }
        )
    )


if __name__ == "__main__":
    main()
