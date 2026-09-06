#define NINFER_DFLASH_SMALL_T_ROWS 5120U
#define NINFER_DFLASH_SMALL_T_COLUMNS 17408U
// Prior static screening rejected T6; the retained exact T5 assembly contains no T6 kernel.
#define NINFER_DFLASH_SMALL_T_WIDTHS 5U
#define NINFER_DFLASH_MLP_DOWN_T5_ONLY 1
#define NINFER_DFLASH_SMALL_T_CALL(args) \
    linear::a8q4g64_linear_dflash_mlp_down_t5_qualification((args), nullptr)
#define NINFER_DFLASH_SMALL_T_SCOPE \
    "standalone Q4 MLP-down N5120/K17408 at exact DFlash T5 only; no production routing"
#define NINFER_DFLASH_SMALL_T_SCHEMA \
    "ninfer.r9700.a8q4-dflash-mlp-down-small-t-cell.v1"
#define NINFER_DFLASH_SMALL_T_LABEL "MLP-down small-T"
#define NINFER_DFLASH_SMALL_T_QUALIFIER_SOURCE \
    "tools/r9700/a8q4_dflash_mlp_down_small_t_qual.hip"
#define NINFER_DFLASH_SMALL_T_SHARED_HARNESS_SOURCE \
    "tools/r9700/a8q4_dflash_small_t_qual.hip"
#include "a8q4_dflash_small_t_qual.hip"
