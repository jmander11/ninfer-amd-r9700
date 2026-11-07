"""Small dependency-neutral constants shared by benchmark matrix authorities."""

MATRIX_SCHEMA_VERSION = 14
PRODUCTION_PREFILL_CHUNKS = (1024, 2048, 4096, 8192)
PRODUCT_CONCURRENCIES = (1, 2, 3, 4)
R9700_KV_PLANE_LAYOUTS = {
    "key": "token-fastest-head-major",
    "value": "feature-fastest-page-major",
    "value_scale": "feature-fastest-page-major",
}
# Startup-selected Text/MTP KV codec: CLI `--kv-cache` value -> report `kv_cache_format` name.
KV_CACHE_FORMATS = {"fp8-int4": "fp8-k-int4-v", "nvfp4": "nvfp4"}
DEFAULT_KV_CACHE = "fp8-int4"
