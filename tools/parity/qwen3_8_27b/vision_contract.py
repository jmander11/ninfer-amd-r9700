"""Host-only report inventory for the fixed Qwen3.8 Vision diagnostic."""

REPORT_FORMAT = "ninfer_vision_intermediate_validation_v2"
GATE = "all 471 boundaries, exact metadata, finite metrics and explicit numerical criteria"


def trace_shapes(patches: int, tokens: int) -> dict[str, list[int]]:
    shapes = {"input/patch_f32": [patches, 1536], "input/patch_bf16": [patches, 1536]}
    for name in ("linear", "bias", "position"):
        shapes[f"patch/{name}"] = [patches, 1152]
    for layer in range(27):
        for name, shape in (
            ("norm1", [patches, 1152]), ("qkv_linear", [patches, 3456]),
            ("qkv_bias", [patches, 3456]), ("q_rope", [patches, 16, 72]),
            ("k_rope", [patches, 16, 72]), ("value", [patches, 16, 72]),
            ("attention", [patches, 1152]), ("projection_linear", [patches, 1152]),
            ("projection_bias", [patches, 1152]), ("attention_residual", [patches, 1152]),
            ("norm2", [patches, 1152]), ("fc1_linear", [patches, 4304]),
            ("fc1_bias", [patches, 4304]), ("gelu", [patches, 4304]),
            ("fc2_linear", [patches, 1152]), ("fc2_bias", [patches, 1152]),
            ("mlp_residual", [patches, 1152]),
        ):
            shapes[f"block_{layer:02d}/{name}"] = shape
    shapes["merger/norm"] = [patches, 1152]
    for name in ("grouped", "fc1_linear", "fc1_bias", "gelu"):
        shapes[f"merger/{name}"] = [tokens, 4608]
    for name in ("fc2_linear", "fc2_bias"):
        shapes[f"merger/{name}"] = [tokens, 5120]
    return shapes


EXPECTED_TRACE_NAMES = tuple(trace_shapes(1, 1))
assert len(EXPECTED_TRACE_NAMES) == 471


# The cross-runtime profile permits independent BF16 reduction association at every public Op.
# sqrt(54) times the linear A16 gross allowance (two residual branches in each of 27 blocks), plus
# the merger, is about 6.2%; cosine additionally catches coherent directional drift that a
# norm-only bound can miss. This is a schedule diagnostic criterion, not a replacement for the
# tighter independent FP32/FP64 criteria in each C++ Op test.
PRODUCTION_RELATIVE_RMSE_LIMIT = 7.0e-2
PRODUCTION_COSINE_MINIMUM = 0.998
PRODUCTION_FINAL_RELATIVE_RMSE_LIMIT = 5.0e-2
# Per-token and per-feature gates prevent real-shape averaging from hiding a dropped row or a
# bad strided column. Each group uses its own reference RMS with a 5% whole-tensor activity floor,
# so near-dead groups do not create unbounded ratios. The limits remain above normal BF16
# association drift while a replaced or zeroed material group has scaled RMSE near one.
LOCAL_RELATIVE_RMSE_LIMIT = 6.0e-3
LOCAL_COSINE_MINIMUM = 0.99998
LOCAL_GROUP_SCALED_RMSE_LIMIT = 1.6e-2
LOCAL_GROUP_COSINE_MINIMUM = 0.99996
# Independent FFmpeg/OpenCV decoding may differ by one source-pixel code. The relative full-image
# bound prevents that local allowance from hiding a systematic resize or normalization mismatch.
PREPROCESSING_ABSOLUTE_LIMIT = 7.844e-3
PREPROCESSING_RELATIVE_RMSE_LIMIT = 1.0e-4
POSITION_WEIGHT_ABSOLUTE_LIMIT = 6.0e-6
# Q4/Q5 matrices appear twice per residual block. A root-sum-square accumulation of a 5% Q4
# weight profile across 54 branches is about 37%. The final merger is materially tighter because
# LayerNorm bounds its input and both merger matrices are W8.
SOURCE_RELATIVE_RMSE_LIMIT = 3.8e-1
SOURCE_COSINE_MINIMUM = 0.93
SOURCE_FINAL_RELATIVE_RMSE_LIMIT = 2.5e-1
SOURCE_FINAL_COSINE_MINIMUM = 0.97
GROUP_ACTIVITY_FLOOR = 5.0e-2

CRITERIA = {
            "production_relative_rmse": PRODUCTION_RELATIVE_RMSE_LIMIT,
            "production_cosine": PRODUCTION_COSINE_MINIMUM,
            "production_final_relative_rmse": PRODUCTION_FINAL_RELATIVE_RMSE_LIMIT,
            "local_relative_rmse": LOCAL_RELATIVE_RMSE_LIMIT,
            "local_cosine": LOCAL_COSINE_MINIMUM,
            "local_group_scaled_rmse": LOCAL_GROUP_SCALED_RMSE_LIMIT,
            "local_group_cosine": LOCAL_GROUP_COSINE_MINIMUM,
            "group_activity_floor": GROUP_ACTIVITY_FLOOR,
            "preprocessing_absolute": PREPROCESSING_ABSOLUTE_LIMIT,
            "preprocessing_relative_rmse": PREPROCESSING_RELATIVE_RMSE_LIMIT,
            "position_weight_absolute": POSITION_WEIGHT_ABSOLUTE_LIMIT,
            "source_relative_rmse": SOURCE_RELATIVE_RMSE_LIMIT,
            "source_cosine": SOURCE_COSINE_MINIMUM,
            "source_final_relative_rmse": SOURCE_FINAL_RELATIVE_RMSE_LIMIT,
            "source_final_cosine": SOURCE_FINAL_COSINE_MINIMUM,
}

def summarize(
    comparisons: list[dict[str, object]], *, profile: str
) -> dict[str, object]:
    worst_relative = max(comparisons, key=lambda value: value["relative_rmse"])
    worst_cosine = min(comparisons, key=lambda value: value["cosine"])
    worst_token_rmse = max(
        comparisons, key=lambda value: value["tokens"]["worst_scaled_rmse"]
    )
    worst_token_cosine = min(
        comparisons, key=lambda value: value["tokens"]["worst_cosine"]
    )
    worst_feature_rmse = max(
        comparisons, key=lambda value: value["features"]["worst_scaled_rmse"]
    )
    worst_feature_cosine = min(
        comparisons, key=lambda value: value["features"]["worst_cosine"]
    )
    nonfinite = [
        value["name"]
        for value in comparisons
        if not value["actual_finite"] or not value["reference_finite"]
    ]
    result = {
        "count": len(comparisons),
        "all_finite": not nonfinite,
        "nonfinite": nonfinite,
        "worst_relative_rmse": {
            "item": worst_relative["item"],
            "name": worst_relative["name"],
            "value": worst_relative["relative_rmse"],
        },
        "worst_cosine": {
            "item": worst_cosine["item"],
            "name": worst_cosine["name"],
            "value": worst_cosine["cosine"],
        },
        "worst_token_scaled_rmse": {
            "item": worst_token_rmse["item"],
            "name": worst_token_rmse["name"],
            "index": worst_token_rmse["tokens"]["worst_scaled_rmse_index"],
            "value": worst_token_rmse["tokens"]["worst_scaled_rmse"],
        },
        "worst_token_cosine": {
            "item": worst_token_cosine["item"],
            "name": worst_token_cosine["name"],
            "index": worst_token_cosine["tokens"]["worst_cosine_index"],
            "value": worst_token_cosine["tokens"]["worst_cosine"],
        },
        "worst_feature_scaled_rmse": {
            "item": worst_feature_rmse["item"],
            "name": worst_feature_rmse["name"],
            "index": worst_feature_rmse["features"]["worst_scaled_rmse_index"],
            "value": worst_feature_rmse["features"]["worst_scaled_rmse"],
        },
        "worst_feature_cosine": {
            "item": worst_feature_cosine["item"],
            "name": worst_feature_cosine["name"],
            "index": worst_feature_cosine["features"]["worst_cosine_index"],
            "value": worst_feature_cosine["features"]["worst_cosine"],
        },
        "comparisons": comparisons,
    }
    if profile == "source":
        failures = [
            value
            for value in comparisons
            if value["relative_rmse"] > SOURCE_RELATIVE_RMSE_LIMIT
            or value["cosine"] < SOURCE_COSINE_MINIMUM
        ]
        finals = [value for value in comparisons if value["name"] == "merger/fc2_bias"]
        if len(finals) != len(comparisons) // len(EXPECTED_TRACE_NAMES):
            raise RuntimeError("source schedule final-boundary inventory is incomplete")
        final_failures = [
            value
            for value in finals
            if value["relative_rmse"] > SOURCE_FINAL_RELATIVE_RMSE_LIMIT
            or value["cosine"] < SOURCE_FINAL_COSINE_MINIMUM
        ]
        result["passed"] = (
            not nonfinite
            and not failures
            and not final_failures
        )
        result["criterion_failures"] = [
            {"item": value["item"], "name": value["name"], "criterion": "source_general"}
            for value in failures
        ] + [
            {"item": value["item"], "name": value["name"], "criterion": "source_final"}
            for value in final_failures
        ]
    elif profile == "local":
        failures = [
            value
            for value in comparisons
            if value["relative_rmse"] > LOCAL_RELATIVE_RMSE_LIMIT
            or value["cosine"] < LOCAL_COSINE_MINIMUM
            or value["tokens"]["worst_scaled_rmse"] > LOCAL_GROUP_SCALED_RMSE_LIMIT
            or value["tokens"]["worst_cosine"] < LOCAL_GROUP_COSINE_MINIMUM
            or value["features"]["worst_scaled_rmse"] > LOCAL_GROUP_SCALED_RMSE_LIMIT
            or value["features"]["worst_cosine"] < LOCAL_GROUP_COSINE_MINIMUM
        ]
        result["passed"] = not nonfinite and not failures
        result["criterion_failures"] = [
            {"item": value["item"], "name": value["name"], "criterion": "local_op"}
            for value in failures
        ]
    elif profile == "production":
        failures = [
            value
            for value in comparisons
            if value["relative_rmse"] > PRODUCTION_RELATIVE_RMSE_LIMIT
            or value["cosine"] < PRODUCTION_COSINE_MINIMUM
        ]
        finals = [value for value in comparisons if value["name"] == "merger/fc2_bias"]
        if len(finals) != len(comparisons) // len(EXPECTED_TRACE_NAMES):
            raise RuntimeError("production schedule final-boundary inventory is incomplete")
        final_failures = [
            value
            for value in finals
            if value["relative_rmse"] > PRODUCTION_FINAL_RELATIVE_RMSE_LIMIT
        ]
        result["passed"] = (
            not nonfinite
            and not failures
            and not final_failures
        )
        result["criterion_failures"] = [
            {
                "item": value["item"],
                "name": value["name"],
                "criterion": "production_general",
            }
            for value in failures
        ] + [
            {"item": value["item"], "name": value["name"], "criterion": "production_final"}
            for value in final_failures
        ]
    else:
        raise ValueError(f"unknown comparison profile {profile!r}")
    return result
