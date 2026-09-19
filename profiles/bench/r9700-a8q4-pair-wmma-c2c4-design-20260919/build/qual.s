	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj,"axG",@progbits,_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj,comdat
	.globl	_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj ; -- Begin function _ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj
	.p2align	8
	.type	_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj,@function
_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj: ; @_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x20
	s_load_b256 s[4:11], s[0:1], 0x0
	s_lshl_b32 s16, ttmp9, 6
	v_mov_b32_e32 v2, 0
	v_or_b32_e32 v1, s16, v0
	s_mov_b32 s2, ttmp7
	s_mov_b32 s0, exec_lo
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s14, v1
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_mov_b32_e32 v2, 0
	s_mov_b32 s3, 0
	s_mov_b32 s18, s14
	s_mov_b32 s19, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_mul_u64 s[18:19], s[18:19], s[2:3]
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	s_lshl_b64 s[18:19], s[18:19], 1
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_add_co_u32 v1, vcc_lo, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s19, v2, vcc_lo
	global_load_u16 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v1
.LBB0_2:
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_nlg_f32_e64 0x7f800000, |v2|
	s_cbranch_execz .LBB0_6
; %bb.3:
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 1
	global_atomic_or_b32 v1, v2, s[12:13] scope:SCOPE_DEV
.LBB0_5:
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v2, 0
.LBB0_6:
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7fffffff, v2
	v_lshlrev_b32_e32 v1, 2, v0
	v_cmp_gt_u32_e64 s0, 32, v0
	ds_store_b32 v1, v2
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_8
; %bb.7:
	ds_load_2addr_b32 v[2:3], v1 offset1:32
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v3, v3, v3 :: v_dual_max_num_f32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v2, v2, v3
	ds_store_b32 v1, v2
.LBB0_8:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB0_10
; %bb.9:
	ds_load_2addr_b32 v[2:3], v1 offset1:16
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v3, v3, v3 :: v_dual_max_num_f32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v2, v2, v3
	ds_store_b32 v1, v2
.LBB0_10:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 8, v0
	s_cbranch_execz .LBB0_12
; %bb.11:
	ds_load_2addr_b32 v[2:3], v1 offset1:8
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v3, v3, v3 :: v_dual_max_num_f32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v2, v2, v3
	ds_store_b32 v1, v2
.LBB0_12:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 4, v0
	s_cbranch_execz .LBB0_14
; %bb.13:
	ds_load_2addr_b32 v[2:3], v1 offset1:4
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v3, v3, v3 :: v_dual_max_num_f32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v2, v2, v3
	ds_store_b32 v1, v2
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 2, v0
	s_cbranch_execz .LBB0_16
; %bb.15:
	ds_load_2addr_b32 v[2:3], v1 offset1:2
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v3, v3, v3 :: v_dual_max_num_f32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v2, v2, v3
	ds_store_b32 v1, v2
.LBB0_16:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_18
; %bb.17:
	ds_load_2addr_b32 v[2:3], v1 offset1:1
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v3, v3, v3 :: v_dual_max_num_f32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v2, v2, v3
	ds_store_b32 v1, v2
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_22
; %bb.19:
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s3, 0
	ds_load_b32 v2, v1
	s_wait_dscnt 0x0
	v_cmp_eq_f32_e32 vcc_lo, 0, v2
	s_cbranch_vccnz .LBB0_21
; %bb.20:
	v_div_scale_f32 v3, null, 0x42fe0000, 0x42fe0000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v4, v3
	v_xor_b32_e32 v3, 0x80000000, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, v3, v4, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, vcc_lo, v2, 0x42fe0000, v2
	v_mul_f32_e32 v6, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v7, v3, v6, v5
	v_fmac_f32_e32 v6, v7, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v5, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v3, 0x42fe0000, v2
	v_cvt_f16_f32_e32 v3.l, v3
.LBB0_21:
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v4, 0xffff, v3
	v_cmp_neq_f32_e32 vcc_lo, 0, v2
	s_lshr_b32 s20, s15, 6
	s_mov_b32 s21, s3
	s_mov_b32 s18, ttmp9
	v_cmp_eq_u32_e64 s1, 0, v4
	s_mul_u64 s[20:21], s[20:21], s[2:3]
	s_mov_b32 s19, s3
	s_lshl_b64 s[20:21], s[20:21], 1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[18:19], s[18:19], 1
	s_and_b32 s1, vcc_lo, s1
	s_add_nc_u64 s[10:11], s[10:11], s[20:21]
	v_cndmask_b32_e64 v2, v3, 1, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[10:11], s[18:19]
	ds_store_b16 v1, v2 offset:256
	global_store_b16 v1, v2, s[10:11]
.LBB0_22:
	s_or_b32 exec_lo, exec_lo, s12
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_32
; %bb.23:
	s_mov_b32 s3, 0
	v_mov_b32_e32 v6, 0
	v_lshl_or_b32 v2, v0, 1, s16
	v_dual_mov_b32 v0, s2 :: v_dual_mov_b32 v1, s3
	ds_load_u16_d16 v3, v6 offset:256
	v_cmp_le_u32_e32 vcc_lo, s14, v2
	s_wait_dscnt 0x0
	v_readfirstlane_b32 s0, v3
	v_readfirstlane_b32 s1, v3
	s_cmp_neq_f16 s0, 0
	s_cselect_b32 s0, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_f16 s1, 0
	s_cselect_b32 s1, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s1, exec_lo, s10
; %bb.24:
	v_dual_mov_b32 v0, s2 :: v_dual_mov_b32 v1, s3
; %bb.25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s1, s1
	v_cvt_f32_f16_e32 v1, v3.l
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_mov_b32_e32 v3, 0
	s_mov_b32 s11, 0
	s_mov_b32 s10, s14
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mul_u64 s[2:3], s[10:11], s[2:3]
	v_lshlrev_b64_e32 v[3:4], 1, v[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, vcc_lo, s2, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s3, v4, vcc_lo
	s_movk_i32 s2, 0xff81
	global_load_u16 v3, v[3:4], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v3, 16, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v4, null, v1, v1, v3
	v_rcp_f32_e32 v5, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v4, v5, 1.0
	v_fmac_f32_e32 v5, v6, v5
	v_div_scale_f32 v6, vcc_lo, v3, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, v6, v5
	v_fma_f32 v8, -v4, v7, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v8, v5
	v_fma_f32 v4, -v4, v7, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v5, v7
	v_div_fixup_f32 v3, v4, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v3, v3
	v_cvt_i32_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_med3_i32 v6, v3, s2, 0x7f
.LBB0_27:
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v4, 1, v2
	v_mov_b32_e32 v3, 0
	s_xor_b32 s0, s0, -1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_le_u32_e32 vcc_lo, s14, v4
	v_dual_mov_b32 v5, v3 :: v_dual_mov_b32 v4, v2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s1
; %bb.28:
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v4, v2
                                        ; implicit-def: $vgpr1
; %bb.29:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_mad_co_u64_u32 v[7:8], null, s14, v0, 0
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	s_movk_i32 s1, 0xff81
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 1, v[7:8]
	v_add_co_u32 v7, vcc_lo, s4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s5, v8, vcc_lo
	v_add_co_u32 v2, vcc_lo, v7, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, v8, v3, vcc_lo
	global_load_u16 v2, v[2:3], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 16, v2
	v_div_scale_f32 v3, null, v1, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v7, v3
	v_fma_f32 v8, -v3, v7, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v8, v7
	v_div_scale_f32 v8, vcc_lo, v2, v1, v2
	v_mul_f32_e32 v9, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v10, -v3, v9, v8
	v_fmac_f32_e32 v9, v10, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v9, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v3, v1, v2
	v_rndne_f32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_i32_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_med3_i32 v3, v1, s1, 0x7f
.LBB0_31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mad_co_u64_u32 v[0:1], null, v0, s15, v[4:5]
	v_and_b32_e32 v2, 15, v6
	v_bfe_u32 v4, v6, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v5, v3, 4, v2
	v_and_or_b32 v4, 0xf0, v3, v4
	v_lshrrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s7, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, s8, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s9, v1, vcc_lo
	global_store_b8 v[2:3], v5, off
	global_store_b8 v[0:1], v4, off
.LBB0_32:
	s_endpgm
.Lfunc_end0:
	.size	_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj, .Lfunc_end0-_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj
		.amdhsa_group_segment_fixed_size 260
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 48
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 11
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj,"axG",@progbits,_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.num_vgpr, 11
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.numbered_sgpr, 22
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1796
; TotalNumSgprs: 24
; NumVgprs: 11
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 260 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 11
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_18fill_u16EPtmt,"axG",@progbits,_ZN12_GLOBAL__N_18fill_u16EPtmt,comdat
	.globl	_ZN12_GLOBAL__N_18fill_u16EPtmt ; -- Begin function _ZN12_GLOBAL__N_18fill_u16EPtmt
	.p2align	8
	.type	_ZN12_GLOBAL__N_18fill_u16EPtmt,@function
_ZN12_GLOBAL__N_18fill_u16EPtmt:        ; @_ZN12_GLOBAL__N_18fill_u16EPtmt
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x24
	s_load_b128 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[0:1], null, s2, ttmp9, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u64_e64 s[6:7], v[0:1]
	s_cbranch_execz .LBB1_2
; %bb.1:
	s_load_b32 s0, s[0:1], 0x10
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s4, v0
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, s0
	global_store_b16 v[0:1], v2, off
.LBB1_2:
	s_endpgm
.Lfunc_end1:
	.size	_ZN12_GLOBAL__N_18fill_u16EPtmt, .Lfunc_end1-_ZN12_GLOBAL__N_18fill_u16EPtmt
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_18fill_u16EPtmt
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_ZN12_GLOBAL__N_18fill_u16EPtmt)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_18fill_u16EPtmt,"axG",@progbits,_ZN12_GLOBAL__N_18fill_u16EPtmt,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.num_vgpr, 3
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.numbered_sgpr, 8
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_18fill_u16EPtmt.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 120
; TotalNumSgprs: 10
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 3
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj,"axG",@progbits,_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj,comdat
	.globl	_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj ; -- Begin function _ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj
	.p2align	8
	.type	_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj,@function
_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj: ; @_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x2c
	s_load_b96 s[4:6], s[0:1], 0x10
	v_mov_b32_e32 v1, 0
	s_mov_b32 s3, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s7, s3
	s_wait_kmcnt 0x0
	s_and_b32 s8, s2, 0xffff
	s_mov_b32 s2, s4
	v_mad_co_u64_u32 v[0:1], null, s8, ttmp9, v[0:1]
	s_mul_u64 s[2:3], s[6:7], s[2:3]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b64 s[2:3], s[2:3], 1
	v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[0:1]
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB2_10
; %bb.1:
	v_lshlrev_b64_e32 v[2:3], 1, v[0:1]
                                        ; implicit-def: $vgpr4_vgpr5
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s2
	s_cbranch_execz .LBB2_3
; %bb.2:
	s_cvt_f32_u32 s2, s6
	s_sub_nc_u64 s[10:11], 0, s[6:7]
	s_mov_b32 s15, 0
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s2, 0, 0x4f800000, s2
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s2, s2
	s_mul_f32 s2, s2, 0x5f7ffffc
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s4, s2, 0x2f800000
	s_trunc_f32 s4, s4
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s2, s4, 0xcf800000, s2
	s_cvt_u32_f32 s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_u64 s[12:13], s[10:11], s[8:9]
	s_mul_hi_u32 s17, s8, s13
	s_mul_i32 s16, s8, s13
	s_mul_hi_u32 s14, s8, s12
	s_mul_i32 s4, s9, s12
	s_add_nc_u64 s[16:17], s[14:15], s[16:17]
	s_mul_hi_u32 s2, s9, s12
	s_mul_hi_u32 s7, s9, s13
	s_add_co_u32 s4, s16, s4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_ci_u32 s14, s17, s2
	s_mul_i32 s12, s9, s13
	s_add_co_ci_u32 s13, s7, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[12:13], s[14:15], s[12:13]
	s_add_co_u32 s8, s8, s12
	s_add_co_ci_u32 s9, s9, s13
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[10:11], s[10:11], s[8:9]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_hi_u32 s13, s8, s11
	s_mul_i32 s12, s8, s11
	s_mul_hi_u32 s14, s8, s10
	s_mul_i32 s4, s9, s10
	s_add_nc_u64 s[12:13], s[14:15], s[12:13]
	s_mul_hi_u32 s2, s9, s10
	s_mul_hi_u32 s7, s9, s11
	s_add_co_u32 s4, s12, s4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_ci_u32 s14, s13, s2
	s_mul_i32 s10, s9, s11
	s_add_co_ci_u32 s11, s7, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[10:11], s[14:15], s[10:11]
	s_add_co_u32 s2, s8, s10
	s_add_co_ci_u32 s4, s9, s11
	s_wait_alu depctr_sa_sdst(0)
	v_mul_hi_u32 v10, v2, s2
	v_mad_co_u64_u32 v[4:5], null, v2, s4, 0
	v_mad_co_u64_u32 v[6:7], null, v3, s2, 0
	v_mad_co_u64_u32 v[8:9], null, v3, s4, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v10, v4
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e32 v4, vcc_lo, v5, v7, vcc_lo
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e32 v5, vcc_lo, 0, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[4:5], null, s6, v4, 0
	v_mad_co_u64_u32 v[5:6], null, s6, v6, v[5:6]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_u32 v2, vcc_lo, v2, v4
	s_wait_alu depctr_va_vcc(0)
	v_sub_co_ci_u32_e64 v3, null, v3, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_u32 v4, vcc_lo, v2, s6
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v5, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_le_u32_e32 vcc_lo, s6, v4
	v_cmp_eq_u32_e64 s2, 0, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v6, 0, -1, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s6, v2
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v6, -1, v6, s2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, 0, -1, vcc_lo
	v_cmp_eq_u32_e64 s2, 0, v3
	v_sub_co_u32 v8, vcc_lo, v4, s6
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, -1, v7, s2
	v_cmp_ne_u32_e32 vcc_lo, 0, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v4, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e32 vcc_lo, 0, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v2, v4, vcc_lo
                                        ; implicit-def: $vgpr2_vgpr3
.LBB2_3:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_5
; %bb.4:
	v_cvt_f32_u32_e32 v3, s6
	s_sub_co_i32 s3, 0, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v3, v3
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_u32_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v4, s3, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_hi_u32 v4, v3, v4
	v_add_nc_u32_e32 v3, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_hi_u32 v3, v2, v3
	v_mul_lo_u32 v3, v3, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v2, v2, v3
	v_subrev_nc_u32_e32 v3, s6, v2
	v_cmp_le_u32_e32 vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo
	v_subrev_nc_u32_e32 v3, s6, v2
	v_cmp_le_u32_e32 vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v4, v2, v3, vcc_lo
.LBB2_5:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b128 s[0:3], s[0:1], 0x0
	s_mov_b32 s4, exec_lo
	v_cmpx_le_u32_e64 s5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB2_7
; %bb.6:
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s1, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, s2, v0
	v_mov_b16_e32 v2.l, 17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	global_store_b8 v[3:4], v2, off
	global_store_b8 v[0:1], v2, off
                                        ; implicit-def: $vgpr4_vgpr5
                                        ; implicit-def: $vgpr0_vgpr1
.LBB2_7:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB2_10
; %bb.8:
	v_add_nc_u32_e32 v2, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s5, v2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB2_10
; %bb.9:
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s1, v1, vcc_lo
	v_add_co_u32 v5, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s3, v1, vcc_lo
	global_load_d16_u8 v2, v[3:4], off
	s_wait_loadcnt 0x0
	v_and_b16 v2.l, v2.l, 15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_or_b16 v2.l, v2.l, 16
	global_store_b8 v[3:4], v2, off
	global_load_d16_u8 v0, v[5:6], off
	s_wait_loadcnt 0x0
	v_and_b16 v0.l, v0.l, 15
	v_or_b16 v0.l, v0.l, 16
	global_store_b8 v[5:6], v0, off
.LBB2_10:
	s_endpgm
.Lfunc_end2:
	.size	_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj, .Lfunc_end2-_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 11
		.amdhsa_next_free_sgpr 18
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj,"axG",@progbits,_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.num_vgpr, 11
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.numbered_sgpr, 18
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1084
; TotalNumSgprs: 20
; NumVgprs: 11
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 11
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj,"axG",@progbits,_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj,comdat
	.globl	_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj ; -- Begin function _ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj
	.p2align	8
	.type	_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj,@function
_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj: ; @_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[28:31], s[0:1], 0x60
	s_lshl_b32 s2, ttmp9, 4
	v_and_b32_e32 v1, 15, v0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s3, s2, s29
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s3, s2, s3
	s_cmp_ge_u32 s2, s29
	v_add_nc_u32_e32 v16, s3, v1
	s_cselect_b32 s2, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s2, exec_lo
	s_cselect_b32 s29, s30, s29
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e64 s29, v16
	s_cbranch_execz .LBB3_124
; %bb.1:
	s_load_b512 s[12:27], s[0:1], 0x0
	v_lshrrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v36, 3, v2
	s_wait_kmcnt 0x0
	s_load_b32 s3, s[18:19], 0x0
	s_wait_kmcnt 0x0
	s_cmp_eq_u32 s3, 0
	s_mov_b32 s3, 0
	s_cselect_b32 s18, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s18
	s_cbranch_vccnz .LBB3_3
; %bb.2:
	v_lshlrev_b32_e32 v4, 3, v2
	s_branch .LBB3_4
.LBB3_3:
	s_mov_b32 s3, -1
                                        ; implicit-def: $vgpr4
.LBB3_4:
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB3_28
; %bb.5:
	v_dual_mov_b32 v35, 0 :: v_dual_and_b32 v4, 15, v16
	v_dual_mov_b32 v20, 0 :: v_dual_and_b32 v37, 16, v0
	s_and_b32 s3, s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v33, 0 :: v_dual_lshlrev_b32 v0, 1, v4
	s_cselect_b32 s6, s26, s22
	v_cmp_gt_u32_e32 vcc_lo, s28, v1
	v_mul_u32_u24_e32 v38, 0x1400, v1
	v_dual_mov_b32 v34, 0 :: v_dual_lshlrev_b32 v1, 3, v4
	v_add_co_u32 v39, s6, s6, v0
	v_or_b32_e32 v0, 1, v36
	s_cselect_b32 s4, s24, s20
	s_cselect_b32 s3, s25, s21
	v_add_co_u32 v41, s4, s4, v1
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v42, null, s3, 0, s4
	v_or_b32_e32 v1, 2, v36
	v_mul_u32_u24_e32 v43, 0x280, v2
	v_cmp_gt_u32_e64 s4, s28, v0
	v_or_b32_e32 v2, 3, v36
	v_mul_u32_u24_e32 v44, 0x50, v0
	v_or_b32_e32 v0, 4, v36
	v_lshrrev_b32_e32 v3, 4, v16
	s_cselect_b32 s5, s27, s23
	v_mul_u32_u24_e32 v45, 0x50, v1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_ci_u32_e64 v40, null, s5, 0, s6
	v_cmp_gt_u32_e64 s5, s28, v1
	v_cmp_gt_u32_e64 s6, s28, v2
	v_mul_u32_u24_e32 v46, 0x50, v2
	v_or_b32_e32 v1, 5, v36
	v_cmp_gt_u32_e64 s7, s28, v0
	v_or_b32_e32 v2, 6, v36
	v_mul_u32_u24_e32 v47, 0x50, v0
	v_or_b32_e32 v0, 7, v36
	v_mad_co_u64_u32 v[17:18], null, 0x50, v3, 0
	v_cmp_gt_u32_e64 s3, s28, v36
	v_cmp_gt_u32_e64 s8, s28, v1
	v_mul_u32_u24_e32 v48, 0x50, v1
	v_cmp_gt_u32_e64 s9, s28, v2
	v_mul_u32_u24_e32 v49, 0x50, v2
	v_cmp_gt_u32_e64 s10, s28, v0
	v_mul_u32_u24_e32 v50, 0x50, v0
	v_dual_mov_b32 v32, 0 :: v_dual_lshlrev_b32 v51, 3, v37
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	s_mov_b32 s19, 0
	s_branch .LBB3_7
.LBB3_6:                                ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	v_lshl_add_u32 v0, v8, 4, v0
	v_lshl_add_u32 v1, v9, 4, v1
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v8, v21, v52, neg(0) op_sel_hi:[1,0,0]
	v_lshl_add_u32 v2, v10, 4, v2
	v_fma_mix_f32 v9, v21, v23, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v0, v0
	v_cvt_f32_i32_e32 v1, v1
	v_lshl_add_u32 v3, v11, 4, v3
	v_fma_mix_f32 v10, v21, v22, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v35, v0, v8 :: v_dual_fmac_f32 v34, v1, v9
	v_lshl_add_u32 v0, v12, 4, v4
	v_cvt_f32_i32_e32 v1, v3
	v_lshl_add_u32 v3, v13, 4, v5
	v_lshl_add_u32 v4, v14, 4, v6
	v_lshl_add_u32 v6, v15, 4, v7
	v_cvt_f32_i32_e32 v0, v0
	v_fma_mix_f32 v5, v21, v24, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v3, v3
	v_fma_mix_f32 v7, v21, v27, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v33, v2, v10
	v_fma_mix_f32 v2, v21, v25, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v4, v4
	v_fma_mix_f32 v8, v21, v26, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v6, v6
	v_fma_mix_f32 v9, v21, v19, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v32, v1, v2 :: v_dual_fmac_f32 v31, v0, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v30, v3, v7 :: v_dual_fmac_f32 v29, v4, v8
	v_fmac_f32_e32 v28, v6, v9
	s_add_co_i32 s19, s19, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s19, 0x50
	s_cbranch_scc0 .LBB3_27
.LBB3_7:                                ; =>This Inner Loop Header: Depth=1
	v_lshl_or_b32 v0, s19, 6, v37
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	s_and_saveexec_b32 s11, vcc_lo
	s_cbranch_execz .LBB3_9
; %bb.8:                                ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v1, v0, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v1, 1, v1
	s_clause 0x1
	global_load_b64 v[8:9], v1, s[12:13]
	global_load_b64 v[26:27], v1, s[14:15]
.LBB3_9:                                ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_add_co_u32 v21, s11, v17, s19
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v22, null, 0, v18, s11
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v54, 32, v0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v25, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 9, v[21:22]
	v_mov_b32_e32 v24, 0
	v_add_co_u32 v19, s11, v41, v1
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v53, null, v42, v2, s11
	v_add_co_u32 v1, s11, v19, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, 0, v53, s11
	global_load_b64 v[55:56], v[1:2], off
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[0:7], v[8:9], v[55:56], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:15], v[26:27], v[55:56], 0 neg_lo:[1,1,0]
	v_mov_b32_e32 v26, 0
	s_and_saveexec_b32 s11, vcc_lo
	s_cbranch_execz .LBB3_11
; %bb.10:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v23, v54, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v24, 1, v23
	v_or_b32_e32 v23, 4, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v26, s20, s12, v23
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v27, null, s13, 0, s20
	v_add_co_u32 v55, s20, s14, v23
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v56, null, s15, 0, s20
	s_clause 0x1
	global_load_b32 v23, v24, s[12:13]
	global_load_b32 v25, v24, s[14:15]
	global_load_b32 v24, v[26:27], off
	global_load_b32 v26, v[55:56], off
.LBB3_11:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_and_b32_e32 v27, 48, v54
	v_lshlrev_b64_e32 v[21:22], 5, v[21:22]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 3, v27
	v_add_co_u32 v54, s11, v19, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v55, null, 0, v53, s11
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v21, s11, v39, v21
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v22, null, v40, v22, s11
	global_load_b64 v[53:54], v[54:55], off
	global_load_d16_b16 v21, v[21:22], off
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[0:7], v[23:24], v[53:54], v[0:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:15], v[25:26], v[53:54], v[8:15] neg_lo:[1,1,0]
	s_and_saveexec_b32 s20, s3
	s_cbranch_execz .LBB3_13
; %bb.12:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[22:23], 1, v[19:20]
	v_add_co_u32 v22, s11, s16, v22
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v23, null, s17, v23, s11
	global_load_d16_b16 v19, v[22:23], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v52, v19.l
.LBB3_13:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	s_and_saveexec_b32 s20, s4
	s_cbranch_execz .LBB3_15
; %bb.14:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[23:24], 1, v[19:20]
	v_add_co_u32 v23, s11, s16, v23
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v24, null, s17, v24, s11
	global_load_d16_b16 v19, v[23:24], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v23, v19.l
.LBB3_15:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	s_and_saveexec_b32 s20, s5
	s_cbranch_execz .LBB3_17
; %bb.16:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[24:25], 1, v[19:20]
	v_add_co_u32 v24, s11, s16, v24
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v25, null, s17, v25, s11
	global_load_d16_b16 v19, v[24:25], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v22, v19.l
.LBB3_17:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	s_and_saveexec_b32 s20, s6
	s_cbranch_execz .LBB3_19
; %bb.18:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 1, v[19:20]
	v_add_co_u32 v25, s11, s16, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s11
	global_load_d16_b16 v19, v[25:26], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v25, v19.l
.LBB3_19:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	s_and_saveexec_b32 s20, s7
	s_cbranch_execz .LBB3_21
; %bb.20:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[26:27], 1, v[19:20]
	v_add_co_u32 v26, s11, s16, v26
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v27, null, s17, v27, s11
	global_load_d16_b16 v19, v[26:27], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v24, v19.l
.LBB3_21:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	s_and_saveexec_b32 s20, s8
	s_cbranch_execnz .LBB3_24
; %bb.22:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	s_and_saveexec_b32 s20, s9
	s_cbranch_execnz .LBB3_25
.LBB3_23:                               ;   in Loop: Header=BB3_7 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	v_mov_b32_e32 v19, 0
	s_and_saveexec_b32 s20, s10
	s_cbranch_execz .LBB3_6
	s_branch .LBB3_26
.LBB3_24:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 1, v[19:20]
	v_add_co_u32 v53, s11, s16, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v54, null, s17, v54, s11
	global_load_d16_b16 v19, v[53:54], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v27, v19.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	s_and_saveexec_b32 s20, s9
	s_cbranch_execz .LBB3_23
.LBB3_25:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 1, v[19:20]
	v_add_co_u32 v53, s11, s16, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v54, null, s17, v54, s11
	global_load_d16_b16 v19, v[53:54], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v26, v19.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s20
	v_mov_b32_e32 v19, 0
	s_and_saveexec_b32 s20, s10
	s_cbranch_execz .LBB3_6
.LBB3_26:                               ;   in Loop: Header=BB3_7 Depth=1
	v_add_nc_u32_e32 v19, s19, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 1, v[19:20]
	v_add_co_u32 v53, s11, s16, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v54, null, s17, v54, s11
	global_load_d16_b16 v19, v[53:54], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v19, v19.l
	s_branch .LBB3_6
.LBB3_27:
	v_mov_b32_e32 v4, v36
.LBB3_28:
	s_load_b256 s[4:11], s[0:1], 0x40
	s_and_b32 s0, s2, exec_lo
	v_subrev_co_u32 v5, s0, 0x1800, v16
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v0, s9 :: v_dual_mov_b32 v1, s8
	s_cselect_b32 s13, s7, s5
	s_cselect_b32 s12, s6, s4
	s_cmp_lg_u32 s31, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, s7, v0, s0
	s_cselect_b32 s9, -1, 0
	v_cndmask_b32_e64 v7, s6, v1, s0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, 0, 1, s9
	s_xor_b32 s3, s2, -1
	s_or_b32 s1, s2, s0
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s3, s0, s3
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_u32_e64 s28, v4
	s_cbranch_execz .LBB3_40
; %bb.29:
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_dual_mov_b32 v3, s29 :: v_dual_mov_b32 v2, v16
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_cbranch_vccnz .LBB3_33
; %bb.30:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v2, 0x1800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v9, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v3, 0x400, v2, s1
	v_cndmask_b32_e64 v2, v5, v16, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v9, v7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s3
; %bb.31:
	v_dual_mov_b32 v3, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v2, v5
; %bb.32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_33:
	v_mov_b16_e32 v9.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_cbranch_vccnz .LBB3_39
; %bb.34:
	v_and_b32_e32 v9, 0x7f800000, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v9
                                        ; implicit-def: $vgpr9
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.35:
	v_bfe_u32 v9, v35, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v35, v9, 0x7fff
                                        ; implicit-def: $vgpr35
; %bb.36:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.37:
	v_and_b32_e32 v9, 0xffff, v35
	v_or_b32_e32 v10, 0x10000, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	v_cndmask_b32_e32 v9, v10, v35, vcc_lo
; %bb.38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_39:
	v_mad_co_u64_u32 v[10:11], null, v3, v4, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v9, off
.LBB3_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v3, 1, v4
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s28, v3
	s_cbranch_execz .LBB3_52
; %bb.41:
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_dual_mov_b32 v9, s29 :: v_dual_mov_b32 v2, v16
	s_cbranch_vccnz .LBB3_45
; %bb.42:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v2, 0x1800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v10, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v9, 0x400, v2, s1
	v_cndmask_b32_e64 v2, v5, v16, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v10, v7, s2
	s_and_saveexec_b32 s9, s3
; %bb.43:
	v_dual_mov_b32 v9, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v2, v5
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_45:
	v_mov_b16_e32 v10.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_51
; %bb.46:
	v_and_b32_e32 v10, 0x7f800000, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.47:
	v_bfe_u32 v10, v34, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v10, v34, v10, 0x7fff
                                        ; implicit-def: $vgpr34
; %bb.48:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.49:
	v_and_b32_e32 v10, 0xffff, v34
	v_or_b32_e32 v11, 0x10000, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v11, v34, vcc_lo
; %bb.50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_51:
	v_mad_co_u64_u32 v[11:12], null, v9, v3, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v10, off
.LBB3_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v3, 2, v4
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s28, v3
	s_cbranch_execz .LBB3_64
; %bb.53:
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_dual_mov_b32 v9, s29 :: v_dual_mov_b32 v2, v16
	s_cbranch_vccnz .LBB3_57
; %bb.54:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v2, 0x1800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v10, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v9, 0x400, v2, s1
	v_cndmask_b32_e64 v2, v5, v16, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v10, v7, s2
	s_and_saveexec_b32 s9, s3
; %bb.55:
	v_dual_mov_b32 v9, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v2, v5
; %bb.56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_57:
	v_mov_b16_e32 v10.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_63
; %bb.58:
	v_and_b32_e32 v10, 0x7f800000, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.59:
	v_bfe_u32 v10, v33, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v10, v33, v10, 0x7fff
                                        ; implicit-def: $vgpr33
; %bb.60:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.61:
	v_and_b32_e32 v10, 0xffff, v33
	v_or_b32_e32 v11, 0x10000, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v11, v33, vcc_lo
; %bb.62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_63:
	v_mad_co_u64_u32 v[11:12], null, v9, v3, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v10, off
.LBB3_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v3, 3, v4
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s28, v3
	s_cbranch_execz .LBB3_76
; %bb.65:
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_dual_mov_b32 v9, s29 :: v_dual_mov_b32 v2, v16
	s_cbranch_vccnz .LBB3_69
; %bb.66:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v2, 0x1800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v10, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v9, 0x400, v2, s1
	v_cndmask_b32_e64 v2, v5, v16, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v10, v7, s2
	s_and_saveexec_b32 s9, s3
; %bb.67:
	v_dual_mov_b32 v9, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v2, v5
; %bb.68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_69:
	v_mov_b16_e32 v10.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_75
; %bb.70:
	v_and_b32_e32 v10, 0x7f800000, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.71:
	v_bfe_u32 v10, v32, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v10, v32, v10, 0x7fff
                                        ; implicit-def: $vgpr32
; %bb.72:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.73:
	v_and_b32_e32 v10, 0xffff, v32
	v_or_b32_e32 v11, 0x10000, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v11, v32, vcc_lo
; %bb.74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_75:
	v_mad_co_u64_u32 v[11:12], null, v9, v3, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v10, off
.LBB3_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v3, 4, v4
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s28, v3
	s_cbranch_execz .LBB3_88
; %bb.77:
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_dual_mov_b32 v9, s29 :: v_dual_mov_b32 v2, v16
	s_cbranch_vccnz .LBB3_81
; %bb.78:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v2, 0x1800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v10, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v9, 0x400, v2, s1
	v_cndmask_b32_e64 v2, v5, v16, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v10, v7, s2
	s_and_saveexec_b32 s9, s3
; %bb.79:
	v_dual_mov_b32 v9, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v2, v5
; %bb.80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_81:
	v_mov_b16_e32 v10.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_87
; %bb.82:
	v_and_b32_e32 v10, 0x7f800000, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.83:
	v_bfe_u32 v10, v31, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v10, v31, v10, 0x7fff
                                        ; implicit-def: $vgpr31
; %bb.84:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.85:
	v_and_b32_e32 v10, 0xffff, v31
	v_or_b32_e32 v11, 0x10000, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v11, v31, vcc_lo
; %bb.86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_87:
	v_mad_co_u64_u32 v[11:12], null, v9, v3, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v10, off
.LBB3_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v3, 5, v4
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s28, v3
	s_cbranch_execz .LBB3_100
; %bb.89:
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_dual_mov_b32 v9, s29 :: v_dual_mov_b32 v2, v16
	s_cbranch_vccnz .LBB3_93
; %bb.90:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v2, 0x1800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v10, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v9, 0x400, v2, s1
	v_cndmask_b32_e64 v2, v5, v16, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v10, v7, s2
	s_and_saveexec_b32 s9, s3
; %bb.91:
	v_dual_mov_b32 v9, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v2, v5
; %bb.92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_93:
	v_mov_b16_e32 v10.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_99
; %bb.94:
	v_and_b32_e32 v10, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.95:
	v_bfe_u32 v10, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v10, v30, v10, 0x7fff
                                        ; implicit-def: $vgpr30
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.97:
	v_and_b32_e32 v10, 0xffff, v30
	v_or_b32_e32 v11, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v11, v30, vcc_lo
; %bb.98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_99:
	v_mad_co_u64_u32 v[11:12], null, v9, v3, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v10, off
.LBB3_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v3, 6, v4
	s_mov_b32 s8, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s28, v3
	s_cbranch_execz .LBB3_112
; %bb.101:
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_dual_mov_b32 v9, s29 :: v_dual_mov_b32 v2, v16
	s_cbranch_vccnz .LBB3_105
; %bb.102:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v2, 0x1800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v10, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v9, 0x400, v2, s1
	v_cndmask_b32_e64 v2, v5, v16, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v10, v7, s2
	s_and_saveexec_b32 s9, s3
; %bb.103:
	v_dual_mov_b32 v9, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v2, v5
; %bb.104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_105:
	v_mov_b16_e32 v10.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_111
; %bb.106:
	v_and_b32_e32 v10, 0x7f800000, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.107:
	v_bfe_u32 v10, v29, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v10, v29, v10, 0x7fff
                                        ; implicit-def: $vgpr29
; %bb.108:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.109:
	v_and_b32_e32 v10, 0xffff, v29
	v_or_b32_e32 v11, 0x10000, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v11, v29, vcc_lo
; %bb.110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB3_111:
	v_mad_co_u64_u32 v[11:12], null, v9, v3, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v10, off
.LBB3_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v2, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, s28, v2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB3_124
; %bb.113:
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	s_cbranch_vccnz .LBB3_117
; %bb.114:
	v_dual_mov_b32 v0, s5 :: v_dual_mov_b32 v1, s4
	v_mov_b32_e32 v3, 0x1800
	v_cndmask_b32_e64 v16, v5, v16, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v0, s7, v0, s0
	v_cndmask_b32_e64 v4, s6, v1, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, 0x400, v3, s1
	v_cndmask_b32_e64 v1, v0, v6, s2
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v0, v4, v7, s2
	s_and_saveexec_b32 s0, s3
; %bb.115:
	v_dual_mov_b32 v3, 0x400 :: v_dual_mov_b32 v0, s10
	v_dual_mov_b32 v1, s11 :: v_dual_mov_b32 v16, v5
; %bb.116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b16_e32 v4.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_123
	s_branch .LBB3_118
.LBB3_117:
	v_dual_mov_b32 v0, s12 :: v_dual_mov_b32 v1, s13
	v_mov_b32_e32 v3, s29
	v_mov_b16_e32 v4.h, 0x7fc1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_123
.LBB3_118:
	v_and_b32_e32 v4, 0x7f800000, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.119:
	v_bfe_u32 v4, v28, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v4, v28, v4, 0x7fff
                                        ; implicit-def: $vgpr28
; %bb.120:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.121:
	v_and_b32_e32 v4, 0xffff, v28
	v_or_b32_e32 v5, 0x10000, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v5, v28, vcc_lo
; %bb.122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.LBB3_123:
	v_mad_co_u64_u32 v[2:3], null, v3, v2, 0
	v_mov_b32_e32 v17, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 1, v[16:17]
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v6, vcc_lo
	global_store_d16_hi_b16 v[0:1], v4, off
.LBB3_124:
	s_endpgm
.Lfunc_end3:
	.size	_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj, .Lfunc_end3-_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 112
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 57
		.amdhsa_next_free_sgpr 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj,"axG",@progbits,_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.num_vgpr, 57
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.numbered_sgpr, 32
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4928
; TotalNumSgprs: 34
; NumVgprs: 57
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 57
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj,"axG",@progbits,_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj,comdat
	.globl	_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj ; -- Begin function _ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj
	.p2align	8
	.type	_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj,@function
_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj: ; @_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x10
	s_load_b32 s3, s[0:1], 0x2c
	v_mov_b32_e32 v1, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s2, s7
	s_and_b32 s3, s3, 0xffff
	s_mov_b32 s7, 0
	v_mad_co_u64_u32 v[0:1], null, s3, ttmp9, v[0:1]
	s_mov_b32 s3, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[2:3], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[0:1]
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB4_6
; %bb.1:
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr2_vgpr3
	v_cmpx_ne_u32_e32 0, v1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB4_3
; %bb.2:
	s_cvt_f32_u32 s3, s6
	s_sub_nc_u64 s[10:11], 0, s[6:7]
	s_mov_b32 s15, 0
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s3, 0, 0x4f800000, s3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s3, s3
	s_mul_f32 s3, s3, 0x5f7ffffc
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s8, s3, 0x2f800000
	s_trunc_f32 s8, s8
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s3, s8, 0xcf800000, s3
	s_cvt_u32_f32 s9, s8
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s8, s3
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_u64 s[12:13], s[10:11], s[8:9]
	s_mul_hi_u32 s17, s8, s13
	s_mul_i32 s16, s8, s13
	s_mul_hi_u32 s14, s8, s12
	s_mul_i32 s7, s9, s12
	s_add_nc_u64 s[16:17], s[14:15], s[16:17]
	s_mul_hi_u32 s3, s9, s12
	s_mul_hi_u32 s18, s9, s13
	s_add_co_u32 s7, s16, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_ci_u32 s14, s17, s3
	s_mul_i32 s12, s9, s13
	s_add_co_ci_u32 s13, s18, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[12:13], s[14:15], s[12:13]
	s_add_co_u32 s8, s8, s12
	s_add_co_ci_u32 s9, s9, s13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_u64 s[10:11], s[10:11], s[8:9]
	s_mul_hi_u32 s13, s8, s11
	s_mul_i32 s12, s8, s11
	s_mul_hi_u32 s14, s8, s10
	s_mul_i32 s7, s9, s10
	s_add_nc_u64 s[12:13], s[14:15], s[12:13]
	s_mul_hi_u32 s3, s9, s10
	s_mul_hi_u32 s16, s9, s11
	s_add_co_u32 s7, s12, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_ci_u32 s14, s13, s3
	s_mul_i32 s10, s9, s11
	s_add_co_ci_u32 s11, s16, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[10:11], s[14:15], s[10:11]
	s_add_co_u32 s3, s8, s10
	s_add_co_ci_u32 s7, s9, s11
	s_wait_alu depctr_sa_sdst(0)
	v_mul_hi_u32 v8, v0, s3
	v_mad_co_u64_u32 v[2:3], null, v0, s7, 0
	v_mad_co_u64_u32 v[4:5], null, v1, s3, 0
	v_mad_co_u64_u32 v[6:7], null, v1, s7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, v8, v2
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e32 v2, vcc_lo, v3, v5, vcc_lo
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e32 v3, vcc_lo, 0, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v2, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, s6, v5, 0
	v_mad_co_u64_u32 v[3:4], null, s6, v6, v[3:4]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_u32 v2, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_sub_co_ci_u32_e64 v3, null, v1, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_u32 v4, vcc_lo, v2, s6
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v7, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e32 vcc_lo, s6, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, -1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v5, 2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, 0, v6, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v2, 0, -1, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, -1, v4, vcc_lo
	v_add_co_u32 v7, vcc_lo, v5, 1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, -1, v2, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v7, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e32 vcc_lo, 0, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v5, v3, vcc_lo
.LBB4_3:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB4_5
; %bb.4:
	v_cvt_f32_u32_e32 v2, s6
	s_sub_co_i32 s3, 0, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_u32_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v3, s3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_hi_u32 v3, v2, v3
	v_add_nc_u32_e32 v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_hi_u32 v2, v0, v2
	v_mul_lo_u32 v3, v2, s6
	v_add_nc_u32_e32 v4, 1, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v3, v0, v3
	v_subrev_nc_u32_e32 v5, s6, v3
	v_cmp_le_u32_e32 vcc_lo, s6, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v3, v3, v5 :: v_dual_cndmask_b32 v2, v2, v4
	v_cmp_le_u32_e32 vcc_lo, s6, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v4, 1, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
.LBB4_5:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b128 s[0:3], s[0:1], 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[3:4], null, v2, s4, 0
	v_mul_lo_u32 v6, v2, s6
	v_mov_b32_e32 v5, 0
	s_mov_b32 s7, 0
	s_mov_b32 s6, s5
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[6:7], 1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[3:4]
	v_sub_nc_u32_e32 v4, v0, v6
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v6, vcc_lo, s0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v3, vcc_lo
	v_lshlrev_b64_e32 v[2:3], 1, v[4:5]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v6, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s5, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v4, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	global_load_d16_b16 v2, v[2:3], off
	s_wait_loadcnt 0x0
	global_store_b16 v[0:1], v2, off
.LBB4_6:
	s_endpgm
.Lfunc_end4:
	.size	_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj, .Lfunc_end4-_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 19
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj,"axG",@progbits,_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.num_vgpr, 10
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.numbered_sgpr, 19
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 992
; TotalNumSgprs: 21
; NumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 21
; NumVGPRsForWavesPerEU: 10
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_5d743e954893e67d,@object ; @__hip_cuid_5d743e954893e67d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5d743e954893e67d
__hip_cuid_5d743e954893e67d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5d743e954893e67d, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5d743e954893e67d
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 260
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_124baseline_quantize_kernelEPK12hip_bfloat16PhS3_PtPjjj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     11
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     by_value
      - .offset:         16
        .size:           2
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_18fill_u16EPtmt
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_18fill_u16EPtmt.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_121poison_q4_padded_tailEPhS0_jjj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     11
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         88
        .size:           8
        .value_kind:     global_buffer
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           _ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_129a8q4g64_pair_wmma_c2c4_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_jjjj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     57
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj
    .private_segment_fixed_size: 0
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_119pair_extract_kernelEPK12hip_bfloat16PS0_jjjj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
