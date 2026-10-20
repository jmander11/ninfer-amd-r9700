"""Independent sampler-law and captured-format regressions; no GPU required."""

import array
import json
import math

import pytest

from .sampling_capture import inverse_cdf, law, uniform, verify_capture, verify_chain


def config(**overrides):
    return {
        "temperature": 1.5,
        "p_less": 1,
        "typical_exclude": -1,
        "suppressed_tokens": [],
        "seed": 8,
        **overrides,
    }


def test_domain_and_root_only_exclusion():
    values = [0.0, 0.0, 100.0, 100.0]
    cfg = config(typical_exclude=0, suppressed_tokens=[3])
    root, detail = law(values, cfg, [0b1011], 0)
    later, _ = law(values, cfg, [0b1011], 1)
    assert root == {1: 1.0}
    assert later == {0: 0.5, 1: 0.5}
    assert detail["eligible_count"] == 2
    assert math.isclose(detail["collision"], 0.5)


def test_floor_and_singleton_are_dirac_laws():
    flat, detail = law([0.0] * 2048, config(), None, 1)
    assert flat == {0: 1.0}
    assert detail["fallback"] == "empty_membership_support"
    excluded, detail = law([10.0, 0.0, -10.0], config(typical_exclude=0), None, 0)
    assert excluded == {1: 1.0}
    assert detail["fallback"] == "excluded_singleton"


def test_independent_empty_support_block_transition():
    # h1=0 because q1 already covers p1's Dirac; h2=.5. With these known draws,
    # neither accepts, and the first-column residual contains token1 only.
    assert uniform(8, 1, 5, 1) < 0.5 < uniform(8, 1, 5, 2)
    result = verify_chain(
        [{0: 0.5, 1: 0.5}, {0: 1.0}, {2: 1.0}], [0, 0], [{0: 1.0}, {0: 1.0}], 8, 0
    )
    assert result["acceptance"][0]["acceptance_threshold"] == 0
    assert result["acceptance"][1]["acceptance_threshold"] == 0.5
    assert result["accepted_drafts"] == 0
    assert result["licensed_tokens"] == [1]


def test_zero_uniform_cannot_accept_zero_probability():
    assert uniform(13539153, 1, 5, 1) == 0
    result = verify_chain([{0: 1.0}, {0: 1.0}], [1], [{1: 1.0}], 13539153, 0)
    assert result["accepted_drafts"] == 0
    assert result["licensed_tokens"] == [0]
    assert inverse_cdf({0: 0.0, 1: 1.0}, 0.0)[0] == 1


def test_capture_reads_logical_bf16_and_mask(tmp_path):
    logits = array.array("H", [0x0000, 0x3F80, 0x4000, 0x4040])
    masks = array.array("I", [0b0011])
    (tmp_path / "input.bf16").write_bytes(logits.tobytes())
    (tmp_path / "input.mask").write_bytes(masks.tobytes())
    capture = {
        "artifact_type": "ninfer_sampling_decision_capture",
        "schema_version": 1,
        "kind": "ordinary",
        "columns": 1,
        "token_domain": 4,
        "logits": {"file": "input.bf16", "dtype": "bf16_le", "shape": [1, 4]},
        "allowed_token_words": {"file": "input.mask", "dtype": "u32_le", "shape": [1, 1]},
        "config": config(temperature=0),
        "watch_tokens": [1, 3],
        "logical_position": 3,
        "rng_purpose": 1,
        "licensed_tokens": [1],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(capture))
    report = verify_capture(path)
    assert report["tokens_match"]
    assert report["columns"][0]["watched_tokens"][1]["target_sampling_probability"] == 0
    assert report["columns"][0]["watched_tokens"][0]["raw_t1_logprob"] < 0
    capture["kind"] = "dflash_tree"
    path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="only ordinary and chain"):
        verify_capture(path)


def test_penalized_greedy_capture_is_rejected():
    with pytest.raises(ValueError, match="nonzero-penalty"):
        law(
            [1.0, 0.0],
            config(temperature=0, presence_penalty=2, token_counts_present=True),
            None,
            0,
        )
