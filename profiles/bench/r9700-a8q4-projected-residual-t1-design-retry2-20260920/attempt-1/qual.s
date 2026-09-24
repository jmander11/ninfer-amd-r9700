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
	.section	.text._ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j,"axG",@progbits,_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j,comdat
	.globl	_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j ; -- Begin function _ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j
	.p2align	8
	.type	_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j,@function
_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j: ; @_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x4c
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x1400, v0
	s_cbranch_execz .LBB3_3
; %bb.1:
	s_load_b256 s[4:11], s[0:1], 0x0
	s_wait_kmcnt 0x0
	s_load_b32 s2, s[10:11], 0x0
	s_load_b128 s[12:15], s[0:1], 0x30
	s_wait_kmcnt 0x0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s2, 0
	s_cbranch_scc1 .LBB3_4
; %bb.2:
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 1, v[0:1]
	v_add_co_u32 v3, vcc_lo, s12, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, s13, v2, vcc_lo
	v_mov_b16_e32 v1.l, 0x7fc1
	global_store_b16 v[3:4], v1, off
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_5
.LBB3_3:
	s_endpgm
.LBB3_4:
.LBB3_5:
	s_load_b32 s2, s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_lshr_b32 s2, s2, 6
	s_cbranch_scc0 .LBB3_8
; %bb.6:
	v_lshrrev_b32_e32 v1, 4, v0
	s_load_b128 s[16:19], s[0:1], 0x20
	v_and_b32_e32 v3, 15, v0
	s_mov_b64 s[0:1], 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[1:2], null, s2, v1, 0
	s_lshl_b32 s2, s2, 5
	v_lshlrev_b64_e32 v[4:5], 9, v[1:2]
	v_lshlrev_b64_e32 v[1:2], 5, v[1:2]
	v_lshl_or_b32 v6, v3, 3, v4
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_or_b32 v1, v3, 1, v1
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s16, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v5, vcc_lo
	v_add_co_u32 v5, vcc_lo, s18, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s19, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x100, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v7, vcc_lo
	v_mov_b32_e32 v3, 0
.LBB3_7:                                ; =>This Inner Loop Header: Depth=1
	s_clause 0x3
	global_load_b64 v[7:8], v[1:2], off offset:-256
	global_load_b64 v[9:10], v[1:2], off offset:-128
	global_load_b64 v[11:12], v[1:2], off
	global_load_b64 v[13:14], v[1:2], off offset:128
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v15, vcc_lo, v5, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s1, v6, vcc_lo
	s_add_nc_u64 s[10:11], s[4:5], s[0:1]
	global_load_d16_b16 v17, v4, s[8:9]
	global_load_d16_b16 v15, v[15:16], off
	s_add_nc_u64 s[14:15], s[6:7], s[0:1]
	s_load_b256 s[16:23], s[10:11], 0x0
	s_load_b256 s[24:31], s[14:15], 0x0
	v_add_co_u32 v1, vcc_lo, 0x200, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_add_nc_u64 s[0:1], s[0:1], 32
	s_add_nc_u64 s[8:9], s[8:9], 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s2, s0
	s_wait_loadcnt 0x5
	s_wait_kmcnt 0x0
	v_dot8_i32_iu4 v16, s16, v7, 0 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s24, v7, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v16, s17, v8, v16 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s25, v8, v7 neg_lo:[1,1,0]
	s_wait_loadcnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v8, s18, v9, v16 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s26, v9, v7 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v8, s19, v10, v8 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s27, v10, v7 neg_lo:[1,1,0]
	s_wait_loadcnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v8, s20, v11, v8 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s28, v11, v7 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v8, s21, v12, v8 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s29, v12, v7 neg_lo:[1,1,0]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v8, s22, v13, v8 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s30, v13, v7 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v8, s23, v14, v8 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v7, s31, v14, v7 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v7, v7, 4, v8
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v8, v17, v15, neg(0) op_sel_hi:[1,1,0]
	v_cvt_f32_i32_e32 v7, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, v7, v8
	s_cbranch_scc1 .LBB3_7
	s_branch .LBB3_9
.LBB3_8:
	v_mov_b32_e32 v3, 0
.LBB3_9:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0x7f800000, v3
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr2
	v_cmpx_ne_u32_e32 0x7f800000, v1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.10:
	v_bfe_u32 v1, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v2, v3, v1, 0x7fff
                                        ; implicit-def: $vgpr3
; %bb.11:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.12:
	v_or_b32_e32 v2, 0x10000, v3
	v_and_b32_e32 v1, 0xffff, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo
; %bb.13:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v2, 0xffff0000, v2
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s12, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_load_u16 v3, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v3, 16, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v2, v3
	v_and_b32_e32 v3, 0x7f800000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v3
                                        ; implicit-def: $vgpr3
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.14:
	v_bfe_u32 v3, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v3, v2, v3, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.15:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.16:
	v_and_b32_e32 v3, 0xffff, v2
	v_or_b32_e32 v4, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v4, v2, vcc_lo
; %bb.17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	global_store_d16_hi_b16 v[0:1], v3, off
	s_endpgm
.Lfunc_end3:
	.size	_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j, .Lfunc_end3-_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
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
		.amdhsa_next_free_vgpr 18
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j,"axG",@progbits,_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.num_vgpr, 18
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.numbered_sgpr, 32
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1008
; TotalNumSgprs: 34
; NumVgprs: 18
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 18
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm,"axG",@progbits,_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm,comdat
	.globl	_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm ; -- Begin function _ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm
	.p2align	8
	.type	_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm,@function
_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm: ; @_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x1c
	s_load_b128 s[0:3], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[0:1], null, s4, ttmp9, v[0:1]
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[0:1]
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB4_2
; %bb.1:
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	s_mov_b32 s0, 0x19660d
	global_load_b32 v2, v[0:1], off
	s_wait_loadcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[2:3], null, v2, s0, 0x3c6ef35f
	global_store_b32 v[0:1], v2, off
.LBB4_2:
	s_endpgm
.Lfunc_end4:
	.size	_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm, .Lfunc_end4-_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
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
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 5
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm,"axG",@progbits,_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.num_vgpr, 4
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.numbered_sgpr, 5
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 144
; TotalNumSgprs: 7
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 7
; NumVGPRsForWavesPerEU: 4
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
	.type	__hip_cuid_1e336cecd1f11c9b,@object ; @__hip_cuid_1e336cecd1f11c9b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1e336cecd1f11c9b
__hip_cuid_1e336cecd1f11c9b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1e336cecd1f11c9b, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_1e336cecd1f11c9b
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
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_136a8q4g64_projected_residual_t1_kernelEPKhS1_PKtPKjS1_S3_P12hip_bfloat16j.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     18
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
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_137projected_residual_cache_scrub_kernelEPjm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
