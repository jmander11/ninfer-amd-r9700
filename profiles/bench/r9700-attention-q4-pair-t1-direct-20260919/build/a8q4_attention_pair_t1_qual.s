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
	.section	.text._ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_,"axG",@progbits,_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_,comdat
	.globl	_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_ ; -- Begin function _ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_
	.p2align	8
	.type	_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_,@function
_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_: ; @_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x6c
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[3:4], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x3800, v3
	s_cbranch_execz .LBB3_14
; %bb.1:
	s_load_b256 s[4:11], s[0:1], 0x40
	s_mov_b32 s3, exec_lo
	v_cmp_lt_u32_e32 vcc_lo, 0x1bff, v3
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr2
	v_cmpx_gt_u32_e32 0x1c00, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB3_3
; %bb.2:
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s5
	v_mov_b32_e32 v4, s4
	v_subrev_co_u32 v1, s2, 0x1800, v3
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v2, v1, v3, s2
	v_cndmask_b32_e64 v1, s7, v0, s2
	v_cndmask_b32_e64 v0, s6, v4, s2
.LBB3_3:
	s_wait_alu depctr_sa_sdst(0)
	s_or_saveexec_b32 s3, s3
	v_add_nc_u32_e32 v5, 0xffffe400, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB3_5
; %bb.4:
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s9
	v_mov_b32_e32 v4, s8
	v_subrev_co_u32 v1, s2, 0x3400, v3
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v2, v1, v5, s2
	v_cndmask_b32_e64 v1, s11, v0, s2
	v_cndmask_b32_e64 v0, s10, v4, s2
.LBB3_5:
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_kmcnt 0x0
	s_load_b512 s[0:15], s[0:1], 0x0
	v_mov_b16_e32 v4.h, 0x7fc1
	s_wait_kmcnt 0x0
	s_load_b32 s6, s[6:7], 0x0
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s6, 0
	s_cbranch_scc1 .LBB3_13
; %bb.6:
	v_dual_cndmask_b32 v5, v3, v5 :: v_dual_mov_b32 v10, s12
	v_dual_mov_b32 v9, s13 :: v_dual_mov_b32 v12, s14
	v_mov_b32_e32 v11, s15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v6, 4, v5
	v_dual_cndmask_b32 v10, s8, v10 :: v_dual_and_b32 v5, 15, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v9, s9, v9, vcc_lo
	v_cndmask_b32_e32 v12, s10, v12, vcc_lo
	v_mad_co_u64_u32 v[3:4], null, 0xa000, v6, 0
	v_mad_co_u64_u32 v[7:8], null, 0xa00, v6, 0
	v_dual_mov_b32 v6, 0 :: v_dual_cndmask_b32 v11, s11, v11
	s_mov_b64 s[6:7], 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_or_b32 v3, v5, 3, v3
	v_lshl_or_b32 v5, v5, 1, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, v10, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v9, v4, vcc_lo
	v_add_co_u32 v7, vcc_lo, v12, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v11, v8, vcc_lo
	v_add_co_u32 v3, vcc_lo, 0x100, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	v_mov_b32_e32 v5, 0
.LBB3_7:                                ; =>This Inner Loop Header: Depth=1
	s_clause 0x3
	global_load_b64 v[9:10], v[3:4], off offset:-256
	global_load_b64 v[11:12], v[3:4], off offset:-128
	global_load_b64 v[13:14], v[3:4], off
	global_load_b64 v[15:16], v[3:4], off offset:128
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v17, vcc_lo, v7, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s7, v8, vcc_lo
	s_add_nc_u64 s[8:9], s[0:1], s[6:7]
	global_load_d16_b16 v19, v6, s[4:5]
	global_load_d16_b16 v17, v[17:18], off
	s_add_nc_u64 s[16:17], s[2:3], s[6:7]
	s_load_b256 s[8:15], s[8:9], 0x0
	s_load_b256 s[16:23], s[16:17], 0x0
	v_add_co_u32 v3, vcc_lo, 0x200, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	s_add_nc_u64 s[6:7], s[6:7], 32
	s_add_nc_u64 s[4:5], s[4:5], 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 0xa00
	s_wait_loadcnt 0x5
	s_wait_kmcnt 0x0
	v_dot8_i32_iu4 v18, s8, v9, 0 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s16, v9, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v18, s9, v10, v18 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s17, v10, v9 neg_lo:[1,1,0]
	s_wait_loadcnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v10, s10, v11, v18 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s18, v11, v9 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v10, s11, v12, v10 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s19, v12, v9 neg_lo:[1,1,0]
	s_wait_loadcnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v10, s12, v13, v10 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s20, v13, v9 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v10, s13, v14, v10 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s21, v14, v9 neg_lo:[1,1,0]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v10, s14, v15, v10 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s22, v15, v9 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v10, s15, v16, v10 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s23, v16, v9 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v9, v9, 4, v10
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, v19, v17, neg(0) op_sel_hi:[1,1,0]
	v_cvt_f32_i32_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v5, v9, v10
	s_cbranch_scc1 .LBB3_7
; %bb.8:
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7f800000, v5
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr4
	v_cmpx_ne_u32_e32 0x7f800000, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.9:
	v_bfe_u32 v3, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v4, v5, v3, 0x7fff
; %bb.10:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.11:
	v_or_b32_e32 v4, 0x10000, v5
	v_and_b32_e32 v3, 0xffff, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo
; %bb.12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.LBB3_13:
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	global_store_d16_hi_b16 v[0:1], v4, off
.LBB3_14:
	s_endpgm
.Lfunc_end3:
	.size	_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_, .Lfunc_end3-_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 352
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
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 24
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_,"axG",@progbits,_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.num_vgpr, 20
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.numbered_sgpr, 24
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 992
; TotalNumSgprs: 26
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 26
; NumVGPRsForWavesPerEU: 20
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj,"axG",@progbits,_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj,comdat
	.globl	_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj ; -- Begin function _ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj
	.p2align	8
	.type	_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj,@function
_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj: ; @_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x24
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s4, v[0:1]
	v_cmp_gt_u32_e32 vcc_lo, s3, v0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB4_2
; %bb.1:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[2:3], 1, v[0:1]
	v_add_nc_u32_e32 v0, s2, v0
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s4, v0
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_load_d16_b16 v0, v[0:1], off
	v_add_co_u32 v1, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s7, v3, vcc_lo
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off
.LBB4_2:
	s_endpgm
.Lfunc_end4:
	.size	_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj, .Lfunc_end4-_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj
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
		.amdhsa_next_free_vgpr 4
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj,"axG",@progbits,_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.num_vgpr, 4
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.numbered_sgpr, 8
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 160
; TotalNumSgprs: 10
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 10
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
	.type	__hip_cuid_1975d7e79afb520d,@object ; @__hip_cuid_1975d7e79afb520d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1975d7e79afb520d
__hip_cuid_1975d7e79afb520d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1975d7e79afb520d, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_1975d7e79afb520d
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
        .value_kind:     hidden_block_count_x
      - .offset:         100
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         104
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         108
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         110
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         112
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         114
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         116
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         118
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         136
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         144
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         152
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         160
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 352
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_
    .private_segment_fixed_size: 0
    .sgpr_count:     26
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_139a8q4g64_attention_pair_t1_direct_kernelEPKhS1_PKtPKjS1_S3_S1_S3_P12hip_bfloat16S7_S7_S7_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     20
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
    .name:           _ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_124attention_extract_kernelEPK12hip_bfloat16PS0_jj.kd
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
