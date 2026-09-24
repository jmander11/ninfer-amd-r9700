
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx1201
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x3c
	s_load_b64 s[2:3], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[1:2], null, ttmp9, s4, v[0:1]
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_u32_e64 s2, v1
	s_cbranch_execz .LBB0_10
; %bb.1:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_mov_b32_e32 v2, 0
	s_load_b64 s[0:1], s[0:1], 0x20
	s_cmp_eq_u32 s3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 1, v[1:2]
	v_add_nc_u32_e32 v1, s2, v1
	v_lshlrev_b64_e32 v[6:7], 1, v[1:2]
	v_add_nc_u32_e32 v1, s2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 1, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v10, vcc_lo, s8, v4
	v_add_co_ci_u32_e64 v11, null, s9, v5, vcc_lo
	v_add_co_u32 v12, vcc_lo, s8, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s9, v7, vcc_lo
	v_add_co_u32 v14, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s9, v9, vcc_lo
	s_clause 0x2
	global_load_d16_b16 v3, v[10:11], off
	global_load_d16_b16 v0, v[12:13], off
	global_load_d16_hi_b16 v0, v[14:15], off
	s_mov_b32 s9, 0
	s_cbranch_scc1 .LBB0_8
; %bb.2:
	v_add_nc_u32_e32 v1, s2, v1
	v_add_co_u32 v10, vcc_lo, s6, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s7, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_add_co_u32 v12, vcc_lo, s6, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v7, vcc_lo
	v_add_co_u32 v14, vcc_lo, s6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s7, v9, vcc_lo
	v_add_co_u32 v1, vcc_lo, s6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s7, v2, vcc_lo
	s_clause 0x3
	global_load_u16 v16, v[10:11], off
	global_load_u16 v12, v[12:13], off
	global_load_u16 v13, v[14:15], off
	global_load_u16 v1, v[1:2], off
	v_add_co_u32 v10, vcc_lo, s4, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s5, v5, vcc_lo
	s_mov_b32 s8, s2
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v16
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v15, 16, v12
	v_add_co_u32 v12, vcc_lo, s0, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v13
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v17, 16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s1, v5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[8:9], 1
	s_branch .LBB0_4
.LBB0_3:                                ;   in Loop: Header=BB0_4 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_co_u32 v10, vcc_lo, v10, s0
	global_store_d16_hi_b16 v[12:13], v18, off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s1, v11, vcc_lo
	v_add_co_u32 v12, vcc_lo, v12, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s1, v13, vcc_lo
	v_mov_b16_e32 v3.l, v2.l
	v_mov_b16_e32 v0.h, v1.l
	s_add_co_i32 s3, s3, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s3, 0
	s_cbranch_scc1 .LBB0_9
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	global_load_d16_b16 v1, v[10:11], off
	v_mov_b16_e32 v2.l, v0.l
	v_mov_b16_e32 v0.l, v0.h
	v_lshlrev_b32_e32 v3, 16, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e32 v18.l, v2.l
	v_mov_b16_e32 v19.l, v0.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v3, v14, v3, 0
	v_lshlrev_b32_e32 v19, 16, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 16, v18
	v_fmac_f32_e32 v3, v15, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v16, v19
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v18.l, v1.l
	v_lshlrev_b32_e32 v18, 16, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v17, v18
	v_mul_f32_e32 v18, 0xbfb8aa3b, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v19, 0xbfb8aa3b, v3, -v18
	v_rndne_f32_e32 v20, v18
	v_dual_fmac_f32 v19, 0xb2a5705f, v3 :: v_dual_sub_f32 v18, v18, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v18, v18, v19
	v_cvt_i32_f32_e32 v19, v20
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v3
	v_exp_f32_e32 v18, v18
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v18, v18, v19
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, 0, v18, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v18, 0x7f800000, v18, vcc_lo
	v_add_f32_e32 v18, 1.0, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v19, null, v18, v18, v3
	v_rcp_f32_e32 v20, v19
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v21, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v21, v20
	v_div_scale_f32 v21, vcc_lo, v3, v18, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v22, v21, v20
	v_fma_f32 v23, -v19, v22, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v22, v23, v20
	v_fma_f32 v19, -v19, v22, v21
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v19, v19, v20, v22
	v_div_fixup_f32 v3, v19, v18, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v18, 0x7f800000, v3
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b32 s2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s2
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfe_u32 v18, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v18, v3, v18, 0x7fff
                                        ; implicit-def: $vgpr3
; %bb.6:                                ;   in Loop: Header=BB0_4 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB0_3
; %bb.7:                                ;   in Loop: Header=BB0_4 Depth=1
	v_and_b32_e32 v18, 0xffff, v3
	v_or_b32_e32 v19, 0x10000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, v19, v3, vcc_lo
	s_branch .LBB0_3
.LBB0_8:
	s_wait_loadcnt 0x2
	v_mov_b16_e32 v2.l, v3.l
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v1.l, v0.h
.LBB0_9:
	v_add_co_u32 v3, vcc_lo, s10, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s11, v5, vcc_lo
	v_add_co_u32 v5, vcc_lo, s10, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s11, v7, vcc_lo
	v_add_co_u32 v7, vcc_lo, s10, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s11, v9, vcc_lo
	s_clause 0x2
	global_store_b16 v[3:4], v2, off
	global_store_b16 v[5:6], v0, off
	global_store_b16 v[7:8], v1, off
.LBB0_10:
	s_endpgm
.Lfunc_end0:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj, .Lfunc_end0-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
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
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.num_vgpr, 24
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.numbered_sgpr, 12
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1060
; TotalNumSgprs: 14
; NumVgprs: 24
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 24
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x2c
	s_load_b64 s[2:3], s[0:1], 0x18
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[4:5], null, ttmp9, s4, v[0:1]
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_u32_e64 s2, v4
	s_cbranch_execz .LBB1_10
; %bb.1:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v5, 0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_cmp_eq_u32 s3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[4:5]
	v_add_nc_u32_e32 v4, s2, v4
	v_lshlrev_b64_e32 v[2:3], 1, v[4:5]
	v_add_nc_u32_e32 v4, s2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]
	s_wait_kmcnt 0x0
	v_add_co_u32 v6, vcc_lo, s6, v0
	v_add_co_ci_u32_e64 v7, null, s7, v1, vcc_lo
	v_add_co_u32 v10, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s7, v3, vcc_lo
	v_add_co_u32 v12, vcc_lo, s6, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v5, vcc_lo
	s_clause 0x2
	global_load_d16_hi_b16 v9, v[6:7], off
	global_load_d16_b16 v8, v[10:11], off
	global_load_d16_hi_b16 v8, v[12:13], off
	s_mov_b32 s6, 0
	s_cbranch_scc1 .LBB1_9
; %bb.2:
	s_cmp_eq_u32 s3, 1
	s_cselect_b32 s7, -1, 0
	s_cmp_lg_u32 s2, 1
	s_cselect_b32 s8, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s7, s7, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_6
; %bb.3:
	v_add_co_u32 v6, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s5, v1, vcc_lo
	s_mov_b32 s9, 0
	s_and_b32 s6, s3, -2
	s_mov_b32 s8, s9
.LBB1_4:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[10:11], s[8:9], 1
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v10, vcc_lo, v6, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s11, v7, vcc_lo
	s_add_co_i32 s8, s8, 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, s6
	global_load_b32 v8, v[10:11], off
	s_cbranch_scc1 .LBB1_4
; %bb.5:
	s_cmp_lg_u32 s3, s6
	s_cselect_b32 s7, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_7
	s_branch .LBB1_9
.LBB1_6:
                                        ; implicit-def: $vgpr9_hi16
	s_cbranch_execz .LBB1_9
.LBB1_7:
	s_mov_b32 s9, 0
	s_mov_b32 s8, s2
	s_mov_b32 s7, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[10:11], s[6:7], s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[10:11], s[10:11], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[4:5], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v6, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s5, v1, vcc_lo
	s_sub_co_i32 s4, s3, s6
	s_lshl_b64 s[2:3], s[8:9], 1
.LBB1_8:                                ; =>This Inner Loop Header: Depth=1
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v9.h, v8.l
	v_mov_b16_e32 v8.l, v8.h
	global_load_d16_hi_b16 v8, v[6:7], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v6, vcc_lo, v6, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo
	s_add_co_i32 s4, s4, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 0
	s_cbranch_scc0 .LBB1_8
.LBB1_9:
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, s0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	v_add_co_u32 v4, vcc_lo, s0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	s_wait_loadcnt 0x2
	global_store_d16_hi_b16 v[0:1], v9, off
	s_wait_loadcnt 0x0
	s_clause 0x1
	global_store_b16 v[2:3], v8, off
	global_store_d16_hi_b16 v[4:5], v8, off
.LBB1_10:
	s_endpgm
.Lfunc_end1:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj, .Lfunc_end1-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
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
		.amdhsa_next_free_vgpr 14
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.num_vgpr, 14
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.numbered_sgpr, 12
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 636
; TotalNumSgprs: 14
; NumVgprs: 14
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 14
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_ ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x24
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x2800, v0
	s_cbranch_execz .LBB2_2
; %bb.1:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_e32 v1, v2
	v_lshlrev_b64_e32 v[3:4], 1, v[0:1]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s4, v3
	v_add_co_ci_u32_e64 v6, null, s5, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, 0xffa000, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v6, vcc_lo
	v_subrev_co_u32 v1, vcc_lo, 0x1000, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[1:2]
	v_add_co_u32 v9, s0, s6, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s7, v1, s0
	v_add_co_u32 v0, s0, 0x2fee000, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, 0, v10, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v0, v7, vcc_lo
	v_add_co_u32 v7, s0, 0xffc000, v5
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v1, v1, v8, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v6, s0
	v_add_co_u32 v11, s0, 0x2ff4000, v9
	global_load_d16_b16 v0, v[0:1], off
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, 0, v10, s0
	v_add_co_u32 v1, s0, s2, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s3, v4, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_cndmask_b32 v4, v12, v8 :: v_dual_cndmask_b32 v3, v11, v7
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off
	global_load_d16_b16 v0, v[3:4], off
	v_add_co_u32 v3, s0, 0xffe000, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v6, s0
	v_add_co_u32 v5, s0, 0x2ffa000, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_cndmask_b32 v3, v5, v3 :: v_dual_cndmask_b32 v4, v6, v4
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off offset:20480
	global_load_d16_b16 v0, v[3:4], off
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off offset:40960
.LBB2_2:
	s_endpgm
.Lfunc_end2:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_, .Lfunc_end2-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
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
		.amdhsa_next_free_vgpr 13
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.num_vgpr, 13
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.numbered_sgpr, 8
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 452
; TotalNumSgprs: 10
; NumVgprs: 13
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 13
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
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
	v_cmpx_gt_u32_e32 0x2800, v3
	s_cbranch_execz .LBB3_54
; %bb.1:
	s_clause 0x1
	s_load_b512 s[4:19], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x58
	s_mov_b32 s22, ttmp7
	s_mov_b64 s[20:21], 0
	s_mov_b32 s23, 0
	s_wait_kmcnt 0x0
	s_cmp_eq_u64 s[12:13], 0
	s_mov_b32 s24, s2
	s_cbranch_scc1 .LBB3_3
; %bb.2:
	s_lshl_b64 s[24:25], s[22:23], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[12:13], s[12:13], s[24:25]
	s_load_b32 s24, s[12:13], 0x0
.LBB3_3:
	s_wait_kmcnt 0x0
	s_min_u32 s12, s24, s2
	s_cmp_gt_i32 s24, 0
	s_cselect_b32 s24, s12, 0
	s_lshl_b64 s[12:13], s[22:23], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[14:15], s[14:15], s[12:13]
	s_load_b32 s14, s[14:15], 0x0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s14, -1
	s_cselect_b32 s15, -1, 0
	s_cmp_lt_u32 s14, s3
	s_cselect_b32 s25, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s25, s15, s25
	s_xor_b32 s26, s25, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s26
	s_cbranch_vccz .LBB3_8
; %bb.4:
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v26, 0
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccz .LBB3_9
.LBB3_5:
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccz .LBB3_10
.LBB3_6:
	v_mov_b32_e32 v4, 0
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_11
.LBB3_7:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, v3
	s_cbranch_execz .LBB3_12
	s_branch .LBB3_13
.LBB3_8:
	s_mov_b32 s15, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_u64 s[20:21], s[14:15], 0x7800
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v26, 0
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccnz .LBB3_5
.LBB3_9:
	v_mov_b32_e32 v4, 0
	s_lshl_b64 s[14:15], s[20:21], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_add_nc_u64 s[14:15], s[10:11], s[14:15]
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s14, v0
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v26, 16, v0
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_6
.LBB3_10:
	v_mov_b32_e32 v4, 0
	s_lshl_b64 s[14:15], s[20:21], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[14:15], s[10:11], s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_load_u16 v0, v[0:1], off offset:20480
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v15, 16, v0
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_7
.LBB3_11:
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v0, v3 :: v_dual_mov_b32 v1, v4
.LBB3_12:
	v_lshlrev_b64_e32 v[4:5], 1, v[3:4]
	s_lshl_b64 s[14:15], s[20:21], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[14:15], s[10:11], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, s14, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s15, v5, vcc_lo
	global_load_u16 v2, v[4:5], off offset:40960
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 16, v2
.LBB3_13:
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s21, 0
	s_cbranch_scc1 .LBB3_54
; %bb.14:
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_mov_b32 s28, s3
	s_mov_b32 s3, s21
	s_mov_b32 s29, s21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s8, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v1, vcc_lo
	s_add_nc_u64 s[8:9], s[16:17], s[12:13]
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x40
	s_load_b64 s[26:27], s[0:1], 0x50
	s_load_b32 s8, s[8:9], 0x0
	s_clause 0x3
	global_load_u16 v13, v[5:6], off
	global_load_u16 v14, v[5:6], off offset:20480
	global_load_u16 v24, v[5:6], off offset:40960
	global_load_u16 v25, v[5:6], off offset:61440
	v_subrev_co_u32 v5, s30, 0x1000, v3
	v_mov_b32_e32 v6, 0
	v_cmp_gt_u32_e64 s0, 0x1800, v3
	v_cmp_lt_u32_e64 s1, 0x7ff, v3
	v_add_co_u32 v16, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s19, v1, vcc_lo
	v_add_co_u32 v18, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s5, v1, vcc_lo
	s_mul_u64 s[16:17], s[22:23], s[2:3]
	s_mov_b32 s9, s21
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s8, -1
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s20, s8, s24
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v22, 16, v13
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v23, 16, v14
	v_lshlrev_b64_e32 v[2:3], 1, v[5:6]
	v_add_co_u32 v5, vcc_lo, s6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v1, vcc_lo
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v24, 16, v24
	v_add_co_u32 v7, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s7, v3, vcc_lo
	v_add_co_u32 v9, vcc_lo, s26, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s27, v1, vcc_lo
	v_add_co_u32 v11, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s15, v1, vcc_lo
	v_add_co_u32 v20, vcc_lo, s12, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s13, v1, vcc_lo
	v_add_co_u32 v13, vcc_lo, s10, v0
	v_cmp_le_u64_e64 s6, s[20:21], s[28:29]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v25, 16, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s11, v1, vcc_lo
	s_xor_b32 s7, s30, -1
	s_mov_b32 s20, s21
	s_branch .LBB3_16
.LBB3_15:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v26, v15 :: v_dual_mov_b32 v15, v4
	s_clause 0x1
	global_store_d16_hi_b16 v[1:2], v3, off offset:20480
	global_store_b16 v[1:2], v0, off offset:40960
	v_mov_b32_e32 v4, v27
	s_add_co_i32 s20, s20, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s20, s2
	s_cbranch_scc0 .LBB3_54
.LBB3_16:                               ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[16:17], s[20:21]
	s_and_saveexec_b32 s10, s0
	s_cbranch_execz .LBB3_18
; %bb.17:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[0:1], null, 0x6000, s4, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x6000, s5, v[1:2]
	global_load_d16_b16 v0, v[0:1], off offset:12288
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s4, v[9:10]
	v_mad_co_u64_u32 v[2:3], null, 0x3000, s5, v[2:3]
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off
.LBB3_18:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_cmp_lt_u32 s20, s24
	s_cselect_b32 s10, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s25, s10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s10
	s_mov_b32 s10, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_28
; %bb.19:                               ;   in Loop: Header=BB3_16 Depth=1
	s_and_saveexec_b32 s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB3_25
; %bb.20:                               ;   in Loop: Header=BB3_16 Depth=1
	s_and_saveexec_b32 s11, s7
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s11, exec_lo, s11
	s_cbranch_execz .LBB3_22
; %bb.21:                               ;   in Loop: Header=BB3_16 Depth=1
	v_mad_co_u64_u32 v[0:1], null, 0x3000, s4, v[11:12]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s5, v[1:2]
	v_mov_b16_e32 v2.l, 0
	global_store_b16 v[0:1], v2, off offset:-8192
.LBB3_22:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s11, s11
	s_cbranch_execz .LBB3_24
; %bb.23:                               ;   in Loop: Header=BB3_16 Depth=1
	s_lshl_b64 s[12:13], s[4:5], 12
	v_mov_b16_e32 v0.l, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v20, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s13, v21, vcc_lo
	global_store_b16 v[1:2], v0, off offset:-4096
.LBB3_24:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
.LBB3_25:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB3_27
; %bb.26:                               ;   in Loop: Header=BB3_16 Depth=1
	s_lshl_b64 s[12:13], s[4:5], 12
	v_mov_b16_e32 v0.l, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v16, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s13, v17, vcc_lo
	global_store_b16 v[1:2], v0, off
.LBB3_27:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_mov_b32 s10, 0
.LBB3_28:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s10
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_46
; %bb.29:                               ;   in Loop: Header=BB3_16 Depth=1
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s10, s7
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
; %bb.30:                               ;   in Loop: Header=BB3_16 Depth=1
	v_mad_co_u64_u32 v[0:1], null, 0x6000, s4, v[7:8]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, 0x6000, s5, v[1:2]
; %bb.31:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
; %bb.32:                               ;   in Loop: Header=BB3_16 Depth=1
	s_lshl_b64 s[12:13], s[4:5], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, v18, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s13, v19, vcc_lo
; %bb.33:                               ;   in Loop: Header=BB3_16 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	global_load_d16_b16 v0, v[0:1], off
	v_fma_f32 v1, v22, v26, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v23, v15
	v_fmac_f32_e32 v1, v24, v4
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v2.l, v0.l
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 16, v2
	v_fmac_f32_e32 v1, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v2, 0xbfb8aa3b, v1
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v1
	v_fma_f32 v3, 0xbfb8aa3b, v1, -v2
	v_rndne_f32_e32 v26, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v3, 0xb2a5705f, v1 :: v_dual_sub_f32 v2, v2, v26
	v_add_f32_e32 v2, v2, v3
	v_cvt_i32_f32_e32 v3, v26
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v2, v2
	v_ldexp_f32 v2, v2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, 1.0, v2
	v_div_scale_f32 v3, null, v2, v2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v26, v3
	v_fma_f32 v28, -v3, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v28, v26
	v_div_scale_f32 v28, vcc_lo, v1, v2, v1
	v_mul_f32_e32 v29, v28, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v30, -v3, v29, v28
	v_fmac_f32_e32 v29, v30, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v29, v28
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v26, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v2, v3, v2, v1
	v_and_b32_e32 v1, 0x7f800000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execnz .LBB3_47
; %bb.34:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execnz .LBB3_48
.LBB3_35:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execnz .LBB3_49
.LBB3_36:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB3_38
.LBB3_37:                               ;   in Loop: Header=BB3_16 Depth=1
	s_lshl_b64 s[4:5], s[4:5], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, v16, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v17, vcc_lo
	global_store_d16_hi_b16 v[2:3], v1, off
.LBB3_38:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_and_b32_e32 v1, 0x7f800000, v15
	s_mov_b32 s4, exec_lo
                                        ; implicit-def: $vgpr26
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.39:                               ;   in Loop: Header=BB3_16 Depth=1
	v_bfe_u32 v1, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v26, v15, v1, 0x7fff
; %bb.40:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.41:                               ;   in Loop: Header=BB3_16 Depth=1
	v_or_b32_e32 v2, 0x10000, v15
	v_and_b32_e32 v1, 0xffff, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v26, v2, v15, vcc_lo
; %bb.42:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_add_nc_u64 s[4:5], s[20:21], s[8:9]
	v_and_b32_e32 v28, 0x7f800000, v4
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[1:2], null, 0xf000, s4, v[13:14]
	s_mov_b32 s4, exec_lo
	v_mad_co_u64_u32 v[2:3], null, 0xf000, s5, v[2:3]
                                        ; implicit-def: $vgpr3
	global_store_d16_hi_b16 v[1:2], v26, off
	v_cmpx_ne_u32_e32 0x7f800000, v28
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.43:                               ;   in Loop: Header=BB3_16 Depth=1
	v_bfe_u32 v3, v4, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v3, v4, v3, 0x7fff
; %bb.44:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB3_15
; %bb.45:                               ;   in Loop: Header=BB3_16 Depth=1
	v_and_b32_e32 v3, 0xffff, v4
	v_or_b32_e32 v26, 0x10000, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v26, v4, vcc_lo
	s_branch .LBB3_15
.LBB3_46:                               ;   in Loop: Header=BB3_16 Depth=1
	v_mov_b32_e32 v27, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v4, v27
	s_add_co_i32 s20, s20, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s20, s2
	s_cbranch_scc1 .LBB3_16
	s_branch .LBB3_54
.LBB3_47:                               ;   in Loop: Header=BB3_16 Depth=1
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB3_35
.LBB3_48:                               ;   in Loop: Header=BB3_16 Depth=1
	v_and_b32_e32 v1, 0xffff, v2
	v_or_b32_e32 v3, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v3, v2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB3_36
.LBB3_49:                               ;   in Loop: Header=BB3_16 Depth=1
	s_and_saveexec_b32 s11, s7
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s11, exec_lo, s11
	s_cbranch_execz .LBB3_51
; %bb.50:                               ;   in Loop: Header=BB3_16 Depth=1
	v_mad_co_u64_u32 v[28:29], null, 0x3000, s4, v[11:12]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[29:30], null, 0x3000, s5, v[29:30]
	global_store_d16_hi_b16 v[28:29], v1, off offset:-8192
                                        ; implicit-def: $vgpr1
.LBB3_51:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s11, s11
	s_cbranch_execz .LBB3_53
; %bb.52:                               ;   in Loop: Header=BB3_16 Depth=1
	s_lshl_b64 s[12:13], s[4:5], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, v20, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s13, v21, vcc_lo
	global_store_d16_hi_b16 v[2:3], v1, off offset:-4096
.LBB3_53:                               ;   in Loop: Header=BB3_16 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
                                        ; implicit-def: $vgpr1
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execnz .LBB3_37
	s_branch .LBB3_38
.LBB3_54:
	s_endpgm
.Lfunc_end3:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj, .Lfunc_end3-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 31
		.amdhsa_next_free_sgpr 31
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.num_vgpr, 31
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.numbered_sgpr, 31
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2364
; TotalNumSgprs: 33
; NumVgprs: 31
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 31
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x74
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[51:52], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x2800, v51
	s_cbranch_execz .LBB4_61
; %bb.1:
	s_clause 0x1
	s_load_b512 s[4:19], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x60
	s_mov_b32 s22, ttmp7
	s_mov_b64 s[20:21], 0
	s_mov_b32 s23, 0
	s_wait_kmcnt 0x0
	s_cmp_eq_u64 s[12:13], 0
	s_mov_b32 s26, s2
	s_cbranch_scc1 .LBB4_3
; %bb.2:
	s_lshl_b64 s[24:25], s[22:23], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[12:13], s[12:13], s[24:25]
	s_load_b32 s26, s[12:13], 0x0
.LBB4_3:
	s_lshl_b64 s[12:13], s[22:23], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[12:13], s[14:15], s[12:13]
	s_load_b32 s12, s[12:13], 0x0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s12, -1
	s_cselect_b32 s13, -1, 0
	s_cmp_lt_u32 s12, s3
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s25, s13, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s3, s25, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB4_8
; %bb.4:
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v64, 0
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccz .LBB4_9
.LBB4_5:
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccz .LBB4_10
.LBB4_6:
	v_mov_b32_e32 v52, 0
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_11
.LBB4_7:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, v51
	s_cbranch_execz .LBB4_12
	s_branch .LBB4_13
.LBB4_8:
	s_mov_b32 s13, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_u64 s[20:21], s[12:13], 0x7800
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v64, 0
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccnz .LBB4_5
.LBB4_9:
	v_mov_b32_e32 v52, 0
	s_lshl_b64 s[12:13], s[20:21], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_add_nc_u64 s[12:13], s[10:11], s[12:13]
	v_lshlrev_b64_e32 v[0:1], 1, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v64, 16, v0
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_6
.LBB4_10:
	v_mov_b32_e32 v52, 0
	s_lshl_b64 s[12:13], s[20:21], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[10:11], s[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[51:52]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s12, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_load_u16 v0, v[0:1], off offset:20480
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v52, 0 :: v_dual_lshlrev_b32 v63, 16, v0
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB4_7
.LBB4_11:
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v0, v51 :: v_dual_mov_b32 v1, v52
.LBB4_12:
	v_lshlrev_b64_e32 v[2:3], 1, v[51:52]
	s_lshl_b64 s[12:13], s[20:21], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[10:11], s[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_add_co_u32 v2, vcc_lo, s10, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s11, v3, vcc_lo
	global_load_u16 v2, v[2:3], off offset:40960
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v52, 16, v2
.LBB4_13:
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s21, 0
	s_cbranch_scc1 .LBB4_61
; %bb.14:
	v_lshlrev_b64_e32 v[49:50], 1, v[0:1]
	s_min_u32 s24, s26, s2
	s_mov_b32 s3, s21
	s_cmp_gt_i32 s26, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[22:23], s[22:23], s[2:3]
	s_cselect_b32 s26, -1, 0
	v_add_co_u32 v0, vcc_lo, s8, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s9, v50, vcc_lo
	s_load_b256 s[8:15], s[0:1], 0x40
	v_add_co_u32 v53, vcc_lo, s18, v49
	s_clause 0x3
	global_load_u16 v61, v[0:1], off
	global_load_u16 v62, v[0:1], off offset:20480
	global_load_u16 v72, v[0:1], off offset:40960
	global_load_u16 v73, v[0:1], off offset:61440
	v_subrev_co_u32 v0, s20, 0x1000, v51
	v_mov_b32_e32 v1, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v54, null, s19, v50, vcc_lo
	v_cmp_gt_u32_e64 s0, 0x1800, v51
	v_cmp_lt_u32_e64 s1, 0x7ff, v51
	s_xor_b32 s3, s20, -1
	s_cmp_lg_u64 s[16:17], 0
	s_mov_b32 s20, s21
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v70, 16, v61
	v_lshlrev_b64_e32 v[55:56], 1, v[0:1]
	v_add_co_u32 v0, vcc_lo, s4, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, s5, v50, vcc_lo
	s_wait_loadcnt 0x2
	v_dual_mov_b32 v16, v1 :: v_dual_lshlrev_b32 v71, 16, v62
	v_add_co_u32 v55, vcc_lo, s6, v55
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v56, null, s7, v56, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v57, vcc_lo, s14, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, s15, v50, vcc_lo
	v_add_co_u32 v59, vcc_lo, s12, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v60, null, s13, v50, vcc_lo
	v_add_co_u32 v66, vcc_lo, s10, v49
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_dual_mov_b32 v8, v1 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v10, v1 :: v_dual_mov_b32 v11, v1
	v_dual_mov_b32 v12, v1 :: v_dual_mov_b32 v13, v1
	v_dual_mov_b32 v14, v1 :: v_dual_mov_b32 v15, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, s11, v50, vcc_lo
	v_add_co_u32 v68, vcc_lo, s8, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v69, null, s9, v50, vcc_lo
	v_add_co_u32 v61, vcc_lo, s6, v49
	v_dual_mov_b32 v32, v16 :: v_dual_mov_b32 v31, v15
	v_dual_mov_b32 v48, v16 :: v_dual_mov_b32 v47, v15
	v_dual_mov_b32 v30, v14 :: v_dual_mov_b32 v29, v13
	v_dual_mov_b32 v28, v12 :: v_dual_mov_b32 v27, v11
	v_dual_mov_b32 v26, v10 :: v_dual_mov_b32 v25, v9
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v46, v14 :: v_dual_mov_b32 v45, v13
	v_dual_mov_b32 v44, v12 :: v_dual_mov_b32 v43, v11
	v_dual_mov_b32 v42, v10 :: v_dual_mov_b32 v41, v9
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v33, v1
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v72, 16, v72
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v73, 16, v73
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v62, null, s7, v50, vcc_lo
	s_cselect_b32 s6, -1, 0
	s_and_b32 s7, s26, s25
	s_branch .LBB4_17
.LBB4_15:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_mov_b32 m0, s20
	v_movreld_b32_e32 v1, v49
	v_movreld_b32_e32 v17, v50
	v_movreld_b32_e32 v33, v51
.LBB4_16:                               ;   in Loop: Header=BB4_17 Depth=1
	s_add_co_i32 s20, s20, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s20, s2
	s_cbranch_scc0 .LBB4_61
.LBB4_17:                               ; =>This Inner Loop Header: Depth=1
	s_add_nc_u64 s[4:5], s[22:23], s[20:21]
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB4_19
; %bb.18:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[49:50], null, 0x6000, s4, v[61:62]
	v_mad_co_u64_u32 v[74:75], null, 0x3000, s4, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[50:51], null, 0x6000, s5, v[50:51]
	v_mad_co_u64_u32 v[75:76], null, 0x3000, s5, v[75:76]
	global_load_d16_b16 v49, v[49:50], off offset:12288
	s_wait_loadcnt 0x0
	global_store_b16 v[74:75], v49, off
.LBB4_19:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_cmp_gt_u32 s24, s20
	s_cselect_b32 s8, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s8, s7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s8, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_29
; %bb.20:                               ;   in Loop: Header=BB4_17 Depth=1
	v_mad_co_u64_u32 v[49:50], null, 0x5000, s4, v[53:54]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[50:51], null, 0x5000, s5, v[50:51]
	v_mov_b16_e32 v51.l, 0
	global_store_b16 v[49:50], v51, off
	s_and_saveexec_b32 s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s8, exec_lo, s8
	s_cbranch_execz .LBB4_26
; %bb.21:                               ;   in Loop: Header=BB4_17 Depth=1
	s_and_saveexec_b32 s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
	s_cbranch_execz .LBB4_23
; %bb.22:                               ;   in Loop: Header=BB4_17 Depth=1
	v_mad_co_u64_u32 v[49:50], null, 0x3000, s4, v[59:60]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[50:51], null, 0x3000, s5, v[50:51]
	v_mov_b16_e32 v51.l, 0
	global_store_b16 v[49:50], v51, off offset:-8192
.LBB4_23:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
	s_cbranch_execz .LBB4_25
; %bb.24:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[10:11], s[4:5], 12
	v_mov_b16_e32 v49.l, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v50, vcc_lo, v66, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, s11, v67, vcc_lo
	global_store_b16 v[50:51], v49, off offset:-4096
.LBB4_25:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
.LBB4_26:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s8
	s_cbranch_execz .LBB4_28
; %bb.27:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[10:11], s[4:5], 12
	v_mov_b16_e32 v49.l, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v50, vcc_lo, v68, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, s11, v69, vcc_lo
	global_store_b16 v[50:51], v49, off
.LBB4_28:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_mov_b32 s8, 0
.LBB4_29:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_16
; %bb.30:                               ;   in Loop: Header=BB4_17 Depth=1
	s_and_b32 vcc_lo, exec_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB4_52
; %bb.31:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[8:9], s[4:5], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_load_b32 s8, s[8:9], 0x0
	s_cbranch_execnz .LBB4_33
.LBB4_32:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_kmcnt 0x0
	s_add_co_i32 s8, s20, -1
.LBB4_33:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s8, s20
	s_mov_b32 s9, -1
	s_cbranch_scc0 .LBB4_35
; %bb.34:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_16
	s_branch .LBB4_44
.LBB4_35:                               ;   in Loop: Header=BB4_17 Depth=1
	v_mad_co_u64_u32 v[49:50], null, 0x5000, s4, v[53:54]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[50:51], null, 0x5000, s5, v[50:51]
	v_mov_b16_e32 v51.l, 0
	global_store_b16 v[49:50], v51, off
	s_and_saveexec_b32 s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
	s_cbranch_execz .LBB4_41
; %bb.36:                               ;   in Loop: Header=BB4_17 Depth=1
	s_and_saveexec_b32 s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB4_38
; %bb.37:                               ;   in Loop: Header=BB4_17 Depth=1
	v_mad_co_u64_u32 v[49:50], null, 0x3000, s4, v[59:60]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[50:51], null, 0x3000, s5, v[50:51]
	v_mov_b16_e32 v51.l, 0
	global_store_b16 v[49:50], v51, off offset:-8192
.LBB4_38:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB4_40
; %bb.39:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[12:13], s[4:5], 12
	v_mov_b16_e32 v49.l, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v50, vcc_lo, v66, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, s13, v67, vcc_lo
	global_store_b16 v[50:51], v49, off offset:-4096
.LBB4_40:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
.LBB4_41:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
	s_cbranch_execz .LBB4_43
; %bb.42:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[10:11], s[4:5], 12
	v_mov_b16_e32 v49.l, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v50, vcc_lo, v68, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, s11, v69, vcc_lo
	global_store_b16 v[50:51], v49, off
.LBB4_43:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_cbranch_execnz .LBB4_16
.LBB4_44:                               ;   in Loop: Header=BB4_17 Depth=1
                                        ; implicit-def: $vgpr49_vgpr50
	s_and_saveexec_b32 s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.45:                               ;   in Loop: Header=BB4_17 Depth=1
	v_mad_co_u64_u32 v[49:50], null, 0x6000, s4, v[55:56]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[50:51], null, 0x6000, s5, v[50:51]
; %bb.46:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.47:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[10:11], s[4:5], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v49, vcc_lo, v0, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, s11, v65, vcc_lo
; %bb.48:                               ;   in Loop: Header=BB4_17 Depth=1
	s_or_b32 exec_lo, exec_lo, s9
	global_load_d16_b16 v74, v[49:50], off
	s_mov_b32 m0, s8
	s_cmp_lt_i32 s8, 0
	v_movrels_b32_e32 v49, v1
	s_cselect_b32 vcc_lo, -1, 0
	v_movrels_b32_e32 v50, v17
	v_movrels_b32_e32 v75, v33
	s_mov_b32 s8, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v51, v49, v64, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v49, v50, v63 :: v_dual_cndmask_b32 v50, v75, v52
	v_fma_f32 v78, v70, v51, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v78, v71, v49
	s_wait_loadcnt 0x0
	v_mov_b16_e32 v51.l, v74.l
	v_dual_fmac_f32 v78, v72, v50 :: v_dual_lshlrev_b32 v51, 16, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v78, v73, v51
	v_mul_f32_e32 v75, 0xbfb8aa3b, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v76, 0xbfb8aa3b, v78, -v75
	v_rndne_f32_e32 v77, v75
	v_sub_f32_e32 v75, v75, v77
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v78
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v76, 0xb2a5705f, v78
	v_add_f32_e32 v75, v75, v76
	v_cvt_i32_f32_e32 v76, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v75, v75
	v_ldexp_f32 v75, v75, v76
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v75, 0, v75, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v78
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v75, 0x7f800000, v75, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v75, 1.0, v75
	v_div_scale_f32 v76, null, v75, v75, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v77, v76
	v_fma_f32 v79, -v76, v77, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v77, v79, v77
	v_div_scale_f32 v79, vcc_lo, v78, v75, v78
	v_mul_f32_e32 v80, v79, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v81, -v76, v80, v79
	v_fmac_f32_e32 v80, v81, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v76, -v76, v80, v79
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v79, v76, v77, v80
	v_mad_co_u64_u32 v[76:77], null, 0x5000, s4, v[53:54]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v75, v79, v75, v78
	v_mad_co_u64_u32 v[77:78], null, 0x5000, s5, v[77:78]
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v79, 0x7f800000, v75
	global_store_b16 v[76:77], v74, off
                                        ; implicit-def: $vgpr74
	v_cmpx_ne_u32_e32 0x7f800000, v79
	s_xor_b32 s8, exec_lo, s8
	s_cbranch_execnz .LBB4_53
; %bb.49:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s8
	s_cbranch_execnz .LBB4_54
.LBB4_50:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s8, exec_lo, s8
	s_cbranch_execnz .LBB4_55
.LBB4_51:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s8
	s_cbranch_execz .LBB4_15
	s_branch .LBB4_60
.LBB4_52:                               ;   in Loop: Header=BB4_17 Depth=1
                                        ; implicit-def: $sgpr8
	s_branch .LBB4_32
.LBB4_53:                               ;   in Loop: Header=BB4_17 Depth=1
	v_bfe_u32 v74, v75, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v74, v75, v74, 0x7fff
                                        ; implicit-def: $vgpr75
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s8
	s_cbranch_execz .LBB4_50
.LBB4_54:                               ;   in Loop: Header=BB4_17 Depth=1
	v_and_b32_e32 v74, 0xffff, v75
	v_or_b32_e32 v76, 0x10000, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v74
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v74, v76, v75, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s8, exec_lo, s8
	s_cbranch_execz .LBB4_51
.LBB4_55:                               ;   in Loop: Header=BB4_17 Depth=1
	s_and_saveexec_b32 s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
	s_cbranch_execz .LBB4_57
; %bb.56:                               ;   in Loop: Header=BB4_17 Depth=1
	v_mad_co_u64_u32 v[75:76], null, 0x3000, s4, v[59:60]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[76:77], null, 0x3000, s5, v[76:77]
	global_store_d16_hi_b16 v[75:76], v74, off offset:-8192
                                        ; implicit-def: $vgpr74
.LBB4_57:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
	s_cbranch_execz .LBB4_59
; %bb.58:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[10:11], s[4:5], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v75, vcc_lo, v66, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, s11, v67, vcc_lo
	global_store_d16_hi_b16 v[75:76], v74, off offset:-4096
.LBB4_59:                               ;   in Loop: Header=BB4_17 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
                                        ; implicit-def: $vgpr74
	s_and_not1_saveexec_b32 s8, s8
	s_cbranch_execz .LBB4_15
.LBB4_60:                               ;   in Loop: Header=BB4_17 Depth=1
	s_lshl_b64 s[4:5], s[4:5], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v75, vcc_lo, v68, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, s5, v69, vcc_lo
	global_store_d16_hi_b16 v[75:76], v74, off
	s_branch .LBB4_15
.LBB4_61:
	s_endpgm
.Lfunc_end4:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj, .Lfunc_end4-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 360
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
		.amdhsa_next_free_vgpr 82
		.amdhsa_next_free_sgpr 27
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.num_vgpr, 82
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.numbered_sgpr, 27
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2636
; TotalNumSgprs: 29
; NumVgprs: 82
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 82
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x4c
	s_load_b128 s[4:7], s[0:1], 0x30
	v_mov_b32_e32 v1, 0
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[0:1], null, s2, ttmp9, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u64_e64 s[4:5], v[0:1]
	s_cbranch_execz .LBB5_8
; %bb.1:
	s_clause 0x2
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b128 s[12:15], s[0:1], 0x20
	s_load_b32 s2, s[0:1], 0x38
	s_mov_b32 s3, 0
                                        ; implicit-def: $vgpr4_vgpr5
	s_mov_b32 s0, exec_lo
	v_cmpx_ne_u32_e32 0, v1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s0
	s_cbranch_execz .LBB5_3
; %bb.2:
	s_wait_kmcnt 0x0
	s_cvt_f32_u32 s0, s2
	s_mov_b32 s16, 0x4f800000
	s_sub_nc_u64 s[18:19], 0, s[2:3]
	s_mov_b32 s21, s3
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s0, s16, 0x0, s0
	s_mov_b32 s25, s3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s0, s0
	s_mul_f32 s0, s0, 0x5f7ffffc
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s16, s0, 0x2f800000
	s_trunc_f32 s16, s16
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s0, s16, 0xcf800000, s0
	s_cvt_u32_f32 s17, s16
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s16, s0
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_u64 s[22:23], s[18:19], s[16:17]
	s_mul_hi_u32 s27, s16, s23
	s_mul_i32 s26, s16, s23
	s_mul_hi_u32 s20, s16, s22
	s_mul_i32 s24, s17, s22
	s_add_nc_u64 s[20:21], s[20:21], s[26:27]
	s_mul_hi_u32 s0, s17, s22
	s_mul_hi_u32 s28, s17, s23
	s_add_co_u32 s20, s20, s24
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_ci_u32 s24, s21, s0
	s_mul_i32 s22, s17, s23
	s_add_co_ci_u32 s23, s28, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[20:21], s[24:25], s[22:23]
	s_mov_b32 s23, s3
	s_add_co_u32 s16, s16, s20
	s_add_co_ci_u32 s17, s17, s21
	s_mov_b32 s21, s3
	s_mul_u64 s[18:19], s[18:19], s[16:17]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_hi_u32 s25, s16, s19
	s_mul_i32 s24, s16, s19
	s_mul_hi_u32 s20, s16, s18
	s_mul_i32 s3, s17, s18
	s_add_nc_u64 s[20:21], s[20:21], s[24:25]
	s_mul_hi_u32 s0, s17, s18
	s_mul_hi_u32 s26, s17, s19
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s3, s20, s3
	s_add_co_ci_u32 s22, s21, s0
	s_mul_i32 s18, s17, s19
	s_add_co_ci_u32 s19, s26, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[18:19], s[22:23], s[18:19]
	s_add_co_u32 s0, s16, s18
	s_add_co_ci_u32 s3, s17, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mul_hi_u32 v8, v0, s0
	v_mad_co_u64_u32 v[2:3], null, v0, s3, 0
	v_mad_co_u64_u32 v[4:5], null, v1, s0, 0
	v_mad_co_u64_u32 v[6:7], null, v1, s3, 0
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
	v_add_co_u32 v2, vcc_lo, v2, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, s2, v2, 0
	v_mad_co_u64_u32 v[3:4], null, s2, v4, v[3:4]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_u32 v2, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_sub_co_ci_u32_e64 v3, null, v1, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_co_u32 v4, vcc_lo, v2, s2
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v5, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e32 vcc_lo, s2, v4
	v_cmp_eq_u32_e64 s0, 0, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v6, 0, -1, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, 0, -1, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, -1, v6, vcc_lo
	v_sub_co_u32 v8, vcc_lo, v4, s2
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v9, null, 0, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e32 vcc_lo, 0, v6
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v6, -1, v7, s0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v5, v5, v9 :: v_dual_cndmask_b32 v4, v4, v8
	v_cmp_ne_u32_e32 vcc_lo, 0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_cndmask_b32 v5, v3, v5 :: v_dual_cndmask_b32 v4, v2, v4
.LBB5_3:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_5
; %bb.4:
	s_wait_kmcnt 0x0
	v_cvt_f32_u32_e32 v2, s2
	s_sub_co_i32 s1, 0, s2
	v_mov_b32_e32 v5, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_u32_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v3, s1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_hi_u32 v3, v2, v3
	v_add_nc_u32_e32 v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_hi_u32 v2, v0, v2
	v_mul_lo_u32 v2, v2, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v2, v0, v2
	v_subrev_nc_u32_e32 v3, s2, v2
	v_cmp_le_u32_e32 vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo
	v_subrev_nc_u32_e32 v3, s2, v2
	v_cmp_le_u32_e32 vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v4, v2, v3, vcc_lo
.LBB5_5:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshlrev_b64_e32 v[2:3], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_mov_b32 s0, exec_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v6, vcc_lo, s4, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s5, v3, vcc_lo
	global_load_u16 v8, v[6:7], off
	v_add_co_u32 v6, vcc_lo, s10, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s11, v5, vcc_lo
	global_load_b32 v6, v[6:7], off
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v7, 16, v8
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v7
	v_cmpx_nlt_f32_e32 0x41a00000, v6
	s_cbranch_execz .LBB5_7
; %bb.6:
	v_mul_f32_e32 v7, 0x3fb8aa3b, v6
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v6
	s_mov_b32 s1, 0x3e9b6dac
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rndne_f32_e32 v8, v7
	v_fma_f32 v9, 0x3fb8aa3b, v6, -v7
	v_sub_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v9, v6, 0x32a5705f, v9
	v_cvt_i32_f32_e32 v8, v8
	v_add_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v7, v7
	v_ldexp_f32 v7, v7, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, 0x7f800000, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, 1.0, v6
	v_frexp_mant_f32_e32 v8, v7
	v_frexp_exp_i32_f32_e32 v9, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_f32_e32 vcc_lo, 0x3f2aaaab, v8
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v8, null, 0, v9, vcc_lo
	v_add_f32_e32 v9, -1.0, v7
	v_cmp_neq_f32_e32 vcc_lo, 0x7f800000, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v10, 0, v8
	v_cvt_f32_i32_e32 v8, v8
	v_sub_f32_e32 v11, v9, v7
	v_sub_f32_e32 v9, v6, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v7, v7, v10
	v_add_f32_e32 v11, 1.0, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v12, 1.0, v7
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v9, v9, v10
	v_dual_add_f32 v10, -1.0, v7 :: v_dual_add_f32 v11, -1.0, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v13, 1.0, v10
	v_sub_f32_e32 v11, v7, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v7, v7, v13
	v_add_f32_e32 v11, v9, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v7, v9, v7
	v_add_f32_e32 v13, v12, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, v10, v7
	v_rcp_f32_e32 v9, v13
	v_sub_f32_e32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_sub_f32 v10, v10, v14 :: v_dual_add_f32 v11, v11, v12
	v_mul_f32_e32 v15, v14, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v7, v7, v10 :: v_dual_mul_f32 v16, v13, v15
	v_fma_f32 v12, v15, v13, -v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v12, v15, v11
	v_add_f32_e32 v17, v16, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v18, v14, v17
	v_sub_f32_e32 v10, v17, v16
	v_sub_f32_e32 v14, v14, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v10, v10, v12
	v_sub_f32_e32 v14, v14, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v14
	v_add_f32_e32 v7, v10, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v18, v7
	v_dual_mul_f32 v12, v9, v10 :: v_dual_sub_f32 v17, v18, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v14, v13, v12 :: v_dual_add_f32 v7, v7, v17
	v_fma_f32 v13, v12, v13, -v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v13, v12, v11
	v_add_f32_e32 v11, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v16, v10, v11
	v_sub_f32_e32 v14, v11, v14
	v_sub_f32_e32 v10, v10, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v10, v10, v11
	v_sub_f32_e32 v11, v14, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v7, v7, v10
	v_add_f32_e32 v10, v15, v12
	v_add_f32_e32 v7, v11, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v11, v10, v15
	v_add_f32_e32 v7, v16, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v11, v12, v11
	v_mul_f32_e32 v7, v9, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v11, v7
	v_add_f32_e32 v9, v10, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, v9, v9
	s_wait_alu depctr_sa_sdst(0)
	v_dual_fmaak_f32 v12, s1, v11, 0x3ecc95a3 :: v_dual_mul_f32 v13, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmaak_f32 v11, v11, v12, 0x3f2aaada :: v_dual_add_f32 v12, v9, v9
	v_sub_f32_e32 v9, v9, v10
	v_mul_f32_e32 v11, v13, v11
	v_mul_f32_e32 v13, 0x3f317218, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v7, v7, v9 :: v_dual_add_f32 v10, v12, v11
	v_add_f32_e32 v7, v7, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v9, v10, v12
	v_fma_f32 v12, 0x3f317218, v8, -v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v9, v11, v9 :: v_dual_fmamk_f32 v8, v8, 0xb102e308, v12
	v_add_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v13, v8
	v_add_f32_e32 v11, v10, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v13, v9, v13 :: v_dual_add_f32 v12, v9, v11
	v_sub_f32_e32 v10, v11, v10
	v_sub_f32_e32 v8, v8, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v14, v12, v9 :: v_dual_sub_f32 v7, v7, v10
	v_dual_sub_f32 v15, v12, v14 :: v_dual_sub_f32 v10, v11, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v11, v8, v7
	v_sub_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v9, v10, v9 :: v_dual_sub_f32 v10, v11, v8
	v_add_f32_e32 v9, v11, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v11, v11, v10
	v_sub_f32_e32 v7, v7, v10
	v_add_f32_e32 v13, v12, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v8, v8, v11
	v_sub_f32_e32 v10, v13, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v7, v7, v8 :: v_dual_sub_f32 v8, v9, v10
	v_add_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v13, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, 0x7f800000, v7, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, 0x33800000, |v6|
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v6, v7, v6, vcc_lo
.LBB5_7:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	global_load_u16 v7, v[2:3], off
	v_add_co_u32 v2, vcc_lo, s8, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s9, v5, vcc_lo
	global_load_b32 v2, v[2:3], off
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 16, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, 0xbfb8aa3b, v3
	s_wait_loadcnt 0x0
	v_cmp_ngt_f32_e64 s0, 0xc2ce8ed0, v2
	v_fma_f32 v5, 0xbfb8aa3b, v3, -v4
	v_rndne_f32_e32 v7, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v4, v4, v7
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v3
	v_fmamk_f32 v5, v3, 0xb2a5705f, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v7
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v5, 0x3fb8aa3b, v2 :: v_dual_cndmask_b32 v4, 0, v4
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fma_f32 v8, 0x3fb8aa3b, v2, -v5
	v_rndne_f32_e32 v9, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, 0x7f800000, v4, vcc_lo
	v_dual_fmamk_f32 v8, v2, 0x32a5705f, v8 :: v_dual_sub_f32 v5, v5, v9
	v_cvt_i32_f32_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v4, 1.0, v3 :: v_dual_add_f32 v5, v5, v8
	v_div_scale_f32 v3, null, v4, v4, 1.0
	v_div_scale_f32 v8, vcc_lo, 1.0, v4, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v5, v5
	v_rcp_f32_e32 v7, v3
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v5, v5, v9
	v_fma_f32 v10, -v3, v7, 1.0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v5, 0, v5, s0
	v_cmp_nlt_f32_e64 s0, 0x42b17218, v2
	v_fmac_f32_e32 v7, v10, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, 0x7f800000, v5, s0
	v_mul_f32_e32 v10, v8, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e64 v5, v6, -v5
	v_fma_f32 v11, -v3, v10, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v10, v11, v7
	v_fma_f32 v3, -v3, v10, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v7, v3, v7, v10
	v_add_co_u32 v2, vcc_lo, s12, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s13, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, s14, v0
	v_div_fixup_f32 v4, v7, v4, 1.0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_store_b32 v[2:3], v5, off
	global_store_b32 v[0:1], v4, off
.LBB5_8:
	s_endpgm
.Lfunc_end5:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj, .Lfunc_end5-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
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
		.amdhsa_next_free_vgpr 19
		.amdhsa_next_free_sgpr 29
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.num_vgpr, 19
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.numbered_sgpr, 29
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2232
; TotalNumSgprs: 31
; NumVgprs: 19
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 31
; NumVGPRsForWavesPerEU: 19
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_ ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b32 s18, s[0:1], 0x44
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[0:7], s[0:1], 0x20
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v1, v0
	s_mov_b32 s16, ttmp9
	s_mov_b32 s17, 0
	s_wait_kmcnt 0x0
	s_mul_u64 s[6:7], s[16:17], 0x1400
	s_and_b32 s19, s18, 0xffff
	s_mov_b32 s18, s17
.LBB6_1:                                ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v5, s20, s6, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s7, 0, s20
	v_lshlrev_b64_e32 v[7:8], 1, v[1:2]
	v_add_nc_u32_e32 v1, s19, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 1, v[5:6]
	v_add_co_u32 v7, vcc_lo, s8, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v8, null, s9, v8, vcc_lo
	v_add_co_u32 v9, vcc_lo, s10, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s11, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, s12, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, vcc_lo
	global_load_u16 v7, v[7:8], off
	global_load_u16 v8, v[9:10], off
	global_load_u16 v5, v[5:6], off
	v_cmp_lt_u32_e32 vcc_lo, 0x13ff, v1
	s_or_b32 s18, vcc_lo, s18
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v6, 16, v7
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 16, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v4, v6, v5 :: v_dual_lshlrev_b32 v7, 16, v8
	v_fmac_f32_e32 v3, v6, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s18
	s_cbranch_execnz .LBB6_1
; %bb.2:
	s_or_b32 exec_lo, exec_lo, s18
	v_lshlrev_b32_e32 v1, 2, v0
	s_mov_b32 s6, exec_lo
	ds_store_2addr_stride64_b32 v1, v3, v4 offset1:4
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 0x80, v0
	s_cbranch_execz .LBB6_4
; %bb.3:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:2
	ds_load_2addr_stride64_b32 v[4:5], v1 offset0:4 offset1:6
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB6_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s6, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 64, v0
	s_cbranch_execz .LBB6_6
; %bb.5:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:1
	ds_load_2addr_stride64_b32 v[4:5], v1 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB6_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s6, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB6_8
; %bb.7:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:32
	ds_load_2addr_b32 v[4:5], v4 offset1:32
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB6_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s6, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB6_10
; %bb.9:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:16
	ds_load_2addr_b32 v[4:5], v4 offset1:16
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB6_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s6, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 8, v0
	s_cbranch_execz .LBB6_12
; %bb.11:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:8
	ds_load_2addr_b32 v[4:5], v4 offset1:8
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB6_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s6, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 4, v0
	s_cbranch_execz .LBB6_14
; %bb.13:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:4
	ds_load_2addr_b32 v[4:5], v4 offset1:4
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB6_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s6, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 2, v0
	s_cbranch_execz .LBB6_16
; %bb.15:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:2
	ds_load_2addr_b32 v[4:5], v4 offset1:2
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB6_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB6_18
; %bb.17:
	v_add_nc_u32_e64 v0, 4, 0
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:4
	ds_load_2addr_stride64_b32 v[4:5], v0 offset1:4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v4, v2
	v_add_f32_e32 v2, v5, v3
	ds_store_2addr_stride64_b32 v1, v0, v2 offset1:4
.LBB6_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB6_30
; %bb.19:
	v_mov_b32_e32 v0, 0
	ds_load_b32 v0, v0
	s_wait_dscnt 0x0
	v_and_b32_e32 v1, 0x7f800000, v0
	v_readfirstlane_b32 s6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v1
	s_cbranch_vccnz .LBB6_21
; %bb.20:
	s_wait_alu depctr_sa_sdst(0)
	s_bfe_u32 s7, s6, 0x10010
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s6, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s7, 0x7fff
	s_cbranch_execz .LBB6_22
	s_branch .LBB6_23
.LBB6_21:
                                        ; implicit-def: $sgpr9
.LBB6_22:
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s7, s6, 0xffff
	s_or_b32 s8, s6, 0x10000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0
	s_cselect_b32 s9, s6, s8
.LBB6_23:
	v_mov_b32_e32 v0, 0
	ds_load_b32 v0, v0 offset:1024
	s_wait_dscnt 0x0
	v_and_b32_e32 v1, 0x7f800000, v0
	v_readfirstlane_b32 s6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v1
	s_cbranch_vccnz .LBB6_25
; %bb.24:
	s_bfe_u32 s7, s6, 0x10010
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s6, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s7, 0x7fff
	s_cbranch_execz .LBB6_26
	s_branch .LBB6_27
.LBB6_25:
                                        ; implicit-def: $sgpr8
.LBB6_26:
	s_and_b32 s7, s6, 0xffff
	s_or_b32 s8, s6, 0x10000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0
	s_cselect_b32 s8, s6, s8
.LBB6_27:
	s_lshl_b64 s[6:7], s[16:17], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]
	s_load_b32 s0, s[0:1], 0x0
	s_and_b32 s1, s9, 0xffff0000
	s_wait_kmcnt 0x0
	s_add_f32 s1, s0, s1
	s_delay_alu instid0(SALU_CYCLE_3)
	s_cmp_gt_f32 s1, 0x41a00000
	s_cbranch_scc1 .LBB6_29
; %bb.28:
	s_mul_f32 s0, s1, 0x3fb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3)
	s_xor_b32 s9, s0, 0x80000000
	s_rndne_f32 s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s9, s1, 0x3fb8aa3b, s9
	s_cmp_nlt_f32 s1, 0xc2ce8ed0
	s_sub_f32 s0, s0, s10
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s9, s1, 0x32a5705f, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s0, s0, s9
	s_cvt_i32_f32 s9, s10
	v_s_exp_f32 s0, s0
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_1)
	v_ldexp_f32 v0, s0, s9
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s0, v0
	s_cselect_b32 s0, s0, 0
	s_cmp_ngt_f32 s1, 0x42b17218
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s0, s0, 0x7f800000
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s0, 1.0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_frexp_mant_f32_e32 v0, s1
	v_frexp_exp_i32_f32_e32 v1, s1
	v_readfirstlane_b32 s9, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_readfirstlane_b32 s10, v1
	s_cmp_lt_f32 s9, 0x3f2aaaab
	s_add_f32 s9, s1, -1.0
	s_sub_co_ci_u32 s10, s10, 0
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s12, s9, s1
	s_sub_co_i32 s11, 0, s10
	s_cvt_f32_i32 s10, s10
	s_wait_alu depctr_sa_sdst(0)
	v_ldexp_f32 v0, s1, s11
	s_sub_f32 s1, s0, s9
	s_add_f32 s9, s12, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_readfirstlane_b32 s12, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s1, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_ldexp_f32 v0, s1, s11
	s_add_f32 s9, s12, 1.0
	s_add_f32 s13, s12, -1.0
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s9, -1.0
	v_readfirstlane_b32 s11, v0
	s_add_f32 s16, s13, 1.0
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s1, s12, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_sub_f32 s12, s12, s16
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s11, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_add_f32 s11, s11, s12
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s16, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_2)
	s_add_f32 s17, s13, s11
	v_s_rcp_f32 s12, s16
	s_sub_f32 s9, s9, s16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_sub_f32 s13, s13, s17
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s1, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	s_add_f32 s11, s11, s13
	s_mul_f32 s18, s17, s12
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s19, s16, s18
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s20, s19, 0x80000000
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s20, s18, s16
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_fmac_f32 s20, s18, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s9, s19, s20
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_sub_f32 s21, s17, s9
	s_sub_f32 s13, s9, s19
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s17, s17, s21
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_sub_f32 s13, s13, s20
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s9, s17, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s9, s11, s9
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s9, s13, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s11, s21, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s13, s12, s11
	s_sub_f32 s20, s21, s11
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_f32 s17, s16, s13
	s_add_f32 s9, s9, s20
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_2)
	s_xor_b32 s19, s17, 0x80000000
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s19, s13, s16
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s19, s13, s1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s1, s17, s19
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s16, s11, s1
	s_sub_f32 s17, s1, s17
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s11, s11, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s1, s11, s1
	s_sub_f32 s11, s17, s19
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_f32 s1, s9, s1
	s_add_f32 s9, s18, s13
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s11, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_sub_f32 s11, s9, s18
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s16, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_2)
	s_sub_f32 s11, s13, s11
	s_mov_b32 s13, 0x3e9b6dac
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s1, s12, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s11, s1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s12, s11, s11
	s_sub_f32 s9, s11, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1)
	s_fmaak_f32 s13, s12, s13, 0x3ecc95a3
	s_mul_f32 s16, s11, s12
	s_sub_f32 s1, s1, s9
	s_wait_alu depctr_sa_sdst(0)
	s_fmaak_f32 s12, s12, s13, 0x3f2aaada
	s_add_f32 s13, s11, s11
	s_add_f32 s1, s1, s1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s12, s16, s12
	s_mul_f32 s16, s10, 0x3f317218
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_f32 s11, s13, s12
	s_xor_b32 s17, s16, 0x80000000
	s_cmp_neq_f32 s0, 0x7f800000
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s9, s11, s13
	s_fmamk_f32 s13, s10, 0x3f317218, s17
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_f32 s9, s12, s9
	s_fmamk_f32 s10, s10, 0xb102e308, s13
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_f32 s1, s1, s9
	s_add_f32 s9, s16, s10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_f32 s12, s11, s1
	s_sub_f32 s16, s9, s16
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_f32 s13, s9, s12
	s_sub_f32 s11, s12, s11
	s_sub_f32 s10, s10, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s17, s13, s9
	s_sub_f32 s1, s1, s11
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_f32 s18, s13, s17
	s_sub_f32 s11, s12, s17
	s_add_f32 s12, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s9, s9, s18
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_f32 s9, s11, s9
	s_sub_f32 s11, s12, s10
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s9, s12, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_f32 s12, s12, s11
	s_sub_f32 s1, s1, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s16, s13, s9
	s_sub_f32 s10, s10, s12
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_f32 s11, s16, s13
	s_add_f32 s1, s1, s10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s9, s9, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s1, s1, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s1, s16, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s1, s1, 0x7f800000
	s_and_b32 s9, s0, 0x7fffffff
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_f32 s9, 0x33800000
	s_cselect_b32 s1, s0, s1
.LBB6_29:
	s_add_nc_u64 s[10:11], s[14:15], s[6:7]
	s_and_b32 s8, s8, 0xffff0000
	s_load_b32 s0, s[10:11], 0x0
	s_wait_kmcnt 0x0
	s_mul_f32 s9, s0, 0x3fb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_xor_b32 s10, s9, 0x80000000
	s_rndne_f32 s11, s9
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s10, s0, 0x3fb8aa3b, s10
	s_cmp_nlt_f32 s0, 0xc2ce8ed0
	s_sub_f32 s9, s9, s11
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s10, s0, 0x32a5705f, s10
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s0, 0x42b17218
	s_mul_f32 s0, s8, 0xbfb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s9, s9, s10
	s_cvt_i32_f32 s10, s11
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s9, s9
	s_wait_alu depctr_va_sdst(0)
	v_ldexp_f32 v0, s9, s10
	s_rndne_f32 s10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_xor_b32 s9, s0, 0x80000000
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s0, s0, s10
	s_fmamk_f32 s9, s8, 0xbfb8aa3b, s9
	s_cmp_ngt_f32 s8, 0x42ce8ed0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_fmamk_f32 s9, s8, 0xb2a5705f, s9
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s0, s0, s9
	s_cvt_i32_f32 s9, s10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s0, s0
	s_wait_alu depctr_va_sdst(0)
	v_ldexp_f32 v1, s0, s9
	s_cselect_b32 s0, -1, 0
	s_cmp_nlt_f32 s8, 0xc2b17218
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_cselect_b32 s0, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v1, 0x7f800000, v1, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v1, 1.0, v1 :: v_dual_cndmask_b32 v0, 0x7f800000, v0
	v_div_scale_f32 v2, null, v1, v1, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v0, 0x80000000, v0
	v_rcp_f32_e32 v3, v2
	v_xor_b32_e32 v2, 0x80000000, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v0, s1, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v4, v2, v3, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, s0, 1.0, v1, 1.0
	s_mov_b32 vcc_lo, s0
	s_add_nc_u64 s[0:1], s[2:3], s[6:7]
	s_add_nc_u64 s[2:3], s[4:5], s[6:7]
	v_mul_f32_e32 v5, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, v2, v5, v4
	v_fmac_f32_e32 v5, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v2, v5
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v2, v4, v3, v5
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v1, v2, v1, 1.0
	s_clause 0x1
	global_store_b32 v3, v0, s[0:1]
	global_store_b32 v3, v1, s[2:3]
.LBB6_30:
	s_endpgm
.Lfunc_end6:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_, .Lfunc_end6-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 312
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.num_vgpr, 11
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.numbered_sgpr, 22
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2808
; TotalNumSgprs: 24
; NumVgprs: 11
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2048 bytes/workgroup (compile time only)
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
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x24
	s_load_b64 s[2:3], s[0:1], 0x10
	v_mov_b32_e32 v1, 0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[0:1], null, s4, ttmp9, v[0:1]
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[0:1]
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB7_2
; %bb.1:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v3, null, s1, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	global_load_b32 v2, v[2:3], off
	s_wait_loadcnt 0x0
	global_store_b32 v[0:1], v2, off
.LBB7_2:
	s_endpgm
.Lfunc_end7:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm, .Lfunc_end7-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.num_vgpr, 4
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.numbered_sgpr, 5
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 152
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
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[8:11], s[0:1], 0x40
	s_load_b96 s[28:30], s[0:1], 0x50
	s_load_b512 s[12:27], s[0:1], 0x0
	s_mov_b32 s37, 0
	s_mov_b32 s34, ttmp9
	s_mov_b32 s43, s37
	s_mov_b32 s35, s37
	v_mov_b32_e32 v1, 0
	s_mov_b32 s39, s37
	s_wait_kmcnt 0x0
	s_cvt_f32_u32 s2, s10
	s_sub_co_i32 s3, 0, s10
	s_mov_b32 s42, s9
	s_mov_b32 s38, s8
	v_s_rcp_f32 s2, s2
	s_mul_u64 s[44:45], s[42:43], s[34:35]
	s_mov_b32 s36, s11
	s_mul_u64 s[46:47], s[44:45], s[38:39]
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s2, s2
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s3, s3, s2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s2, s11, s2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s2, s10
	s_add_co_i32 s4, s2, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s11, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s5, s3, s10
	s_cmp_ge_u32 s3, s10
	s_cselect_b32 s2, s4, s2
	s_cselect_b32 s3, s5, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s2, 1
	s_cmp_ge_u32 s3, s10
	s_mov_b32 s3, s37
	s_cselect_b32 s6, s4, s2
	s_mul_u64 s[4:5], s[42:43], s[38:39]
	s_cvt_f32_u32 s2, s6
	s_sub_co_i32 s7, 0, s6
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s2, s2
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s2, s2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s7, s2, s7
	s_add_co_i32 s2, s2, s7
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_u64_e64 s[4:5], v[0:1]
	s_cbranch_execz .LBB8_3
; %bb.1:
	s_load_b32 s0, s[0:1], 0x6c
	v_add_co_u32 v2, s1, s46, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v3, null, s47, 0, s1
	s_mov_b32 s11, s37
	v_lshlrev_b64_e32 v[6:7], 2, v[2:3]
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	v_add_co_u32 v4, vcc_lo, s22, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s23, v7, vcc_lo
	v_add_co_u32 v6, vcc_lo, s24, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s25, v7, vcc_lo
	s_wait_kmcnt 0x0
	s_and_b32 s1, s0, 0xffff
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s8, s1, 2
.LBB8_2:                                ; =>This Inner Loop Header: Depth=1
	global_load_b32 v8, v[4:5], off
	v_add_co_u32 v2, vcc_lo, v2, s1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u64_e32 vcc_lo, s[4:5], v[2:3]
	s_or_b32 s11, vcc_lo, s11
	s_wait_loadcnt 0x0
	global_store_b32 v[6:7], v8, off
	v_add_co_u32 v6, s0, v6, s8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, s0
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB8_2
.LBB8_3:
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[34:35], s[2:3]
	s_mov_b32 s23, 0
	s_cmp_eq_u32 s28, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB8_77
; %bb.4:
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s0, s1, s6
	s_lshl_b64 s[2:3], s[34:35], 2
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s0, ttmp9, s0
	s_add_co_i32 s4, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s5, s0, s6
	s_cmp_ge_u32 s0, s6
	v_lshlrev_b32_e32 v6, 2, v0
	s_cselect_b32 s1, s4, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s0, s5, s0
	s_add_co_i32 s4, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s0, s6
	s_add_nc_u64 s[18:19], s[18:19], s[2:3]
	s_cselect_b32 s40, s4, s1
	s_cmp_lg_u32 s9, 0
	v_cmp_gt_u32_e64 s0, s38, v0
	s_cselect_b32 s31, -1, 0
	s_lshl_b64 s[46:47], s[46:47], 2
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v7, 0x800, v6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[24:25], s[24:25], s[46:47]
	v_cmp_eq_u32_e64 s1, 0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, s22, s24, v6
	v_cmp_gt_u32_e64 s2, 0x80, v0
	v_cmp_gt_u32_e64 s3, 64, v0
	v_cmp_gt_u32_e64 s4, 32, v0
	v_cmp_gt_u32_e64 s5, 16, v0
	v_cmp_gt_u32_e64 s6, 8, v0
	v_cmp_gt_u32_e64 s7, 4, v0
	v_cmp_gt_u32_e64 s8, 2, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s25, 0, s22
	v_add_nc_u32_e32 v9, 0xc00, v6
	v_add_nc_u32_e32 v10, 0xc00, v6
	v_add_nc_u32_e32 v11, 0x400, v6
	v_add_nc_u32_e64 v12, 4, 0
	s_mul_u64 s[42:43], s[36:37], s[42:43]
	s_lshl_b64 s[44:45], s[44:45], 1
	s_mov_b32 s11, s23
	s_mov_b32 s41, s23
	s_add_nc_u64 s[16:17], s[16:17], s[44:45]
	s_lshl_b64 s[42:43], s[42:43], 1
	s_add_nc_u64 s[24:25], s[26:27], s[44:45]
	s_lshl_b64 s[26:27], s[38:39], 2
	s_mov_b32 s22, s23
	s_branch .LBB8_6
.LBB8_5:                                ;   in Loop: Header=BB8_6 Depth=1
	s_add_co_i32 s22, s22, 1
	s_add_nc_u64 s[16:17], s[16:17], s[42:43]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s22, s28
	s_add_nc_u64 s[24:25], s[24:25], s[42:43]
	s_cbranch_scc1 .LBB8_77
.LBB8_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB8_32 Depth 2
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	s_and_saveexec_b32 s33, s0
	s_cbranch_execz .LBB8_8
; %bb.7:                                ;   in Loop: Header=BB8_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[44:45], s[22:23], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[44:45], s[44:45], s[40:41]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[13:14], null, s44, s38, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[14:15], null, s45, s38, v[14:15]
	v_lshlrev_b64_e32 v[4:5], 1, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, vcc_lo, s12, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s13, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, s14, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s15, v5, vcc_lo
	global_load_u16 v13, v[13:14], off
	global_load_u16 v4, v[4:5], off
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 16, v13
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 16, v4
.LBB8_8:                                ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v13, v5, v5 :: v_dual_mul_f32 v14, v4, v4
	ds_store_b32 v7, v5
	ds_store_b32 v6, v13 offset:3072
	ds_store_2addr_stride64_b32 v6, v4, v14 offset1:4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s2
	s_cbranch_execz .LBB8_10
; %bb.9:                                ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_stride64_b32 v[4:5], v6 offset0:12 offset1:14
	ds_load_2addr_stride64_b32 v[13:14], v6 offset0:4 offset1:6
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v14, v13
	ds_store_2addr_stride64_b32 v6, v5, v4 offset0:4 offset1:12
.LBB8_10:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s3
	s_cbranch_execz .LBB8_12
; %bb.11:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_stride64_b32 v[4:5], v6 offset0:12 offset1:13
	ds_load_2addr_stride64_b32 v[13:14], v6 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v14, v13
	ds_store_2addr_stride64_b32 v6, v5, v4 offset0:4 offset1:12
.LBB8_12:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s4
	s_cbranch_execz .LBB8_14
; %bb.13:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_b32 v[4:5], v10 offset1:32
	ds_load_2addr_b32 v[13:14], v11 offset1:32
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v14, v13
	ds_store_2addr_stride64_b32 v6, v5, v4 offset0:4 offset1:12
.LBB8_14:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s5
	s_cbranch_execz .LBB8_16
; %bb.15:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_b32 v[4:5], v10 offset1:16
	ds_load_2addr_b32 v[13:14], v11 offset1:16
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v14, v13
	ds_store_2addr_stride64_b32 v6, v5, v4 offset0:4 offset1:12
.LBB8_16:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s6
	s_cbranch_execz .LBB8_18
; %bb.17:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_b32 v[4:5], v10 offset1:8
	ds_load_2addr_b32 v[13:14], v11 offset1:8
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v14, v13
	ds_store_2addr_stride64_b32 v6, v5, v4 offset0:4 offset1:12
.LBB8_18:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s7
	s_cbranch_execz .LBB8_20
; %bb.19:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_b32 v[4:5], v10 offset1:4
	ds_load_2addr_b32 v[13:14], v11 offset1:4
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v14, v13
	ds_store_2addr_stride64_b32 v6, v5, v4 offset0:4 offset1:12
.LBB8_20:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s8
	s_cbranch_execz .LBB8_22
; %bb.21:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_b32 v[4:5], v10 offset1:2
	ds_load_2addr_b32 v[13:14], v11 offset1:2
	s_wait_dscnt 0x0
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v14, v13
	ds_store_2addr_stride64_b32 v6, v5, v4 offset0:4 offset1:12
.LBB8_22:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s1
	s_cbranch_execz .LBB8_24
; %bb.23:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_stride64_b32 v[4:5], v12 offset0:4 offset1:12
	ds_load_2addr_stride64_b32 v[13:14], v6 offset0:4 offset1:12
	s_wait_dscnt 0x0
	v_dual_add_f32 v5, v5, v14 :: v_dual_add_f32 v4, v4, v13
	ds_store_2addr_stride64_b32 v6, v4, v5 offset0:4 offset1:12
.LBB8_24:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s0
	s_cbranch_execz .LBB8_26
; %bb.25:                               ;   in Loop: Header=BB8_6 Depth=1
	ds_load_2addr_stride64_b32 v[4:5], v8 offset0:4 offset1:12
	s_wait_dscnt 0x0
	v_readfirstlane_b32 s39, v5
	s_add_f32 s39, s30, s39
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	s_cmp_lt_f32 s39, 0x800000
	s_mul_f32 s44, s39, 0x4b800000
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s39, s44, s39
	v_readfirstlane_b32 s44, v4
	s_wait_alu depctr_sa_sdst(0)
	v_s_rsq_f32 s39, s39
	ds_load_b32 v4, v7
	ds_load_b32 v5, v6
	s_add_f32 s44, s30, s44
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	s_mul_f32 s46, s44, 0x4b800000
	s_mul_f32 s45, s39, 0x45800000
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s39, s45, s39
	s_cmp_lt_f32 s44, 0x800000
	s_cselect_b32 s44, s46, s44
	s_wait_alu depctr_sa_sdst(0)
	v_s_rsq_f32 s44, s44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s45, s44, 0x45800000
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s44, s45, s44
	s_wait_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mul_f32 v4, s39, v4 :: v_dual_mul_f32 v5, s44, v5
	ds_store_b32 v7, v4
	ds_store_b32 v6, v5
.LBB8_26:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[44:45], s[22:23], s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s1
	s_cbranch_execz .LBB8_28
; %bb.27:                               ;   in Loop: Header=BB8_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[46:47], s[44:45], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[46:47], s[18:19], s[46:47]
	global_load_b32 v4, v8, s[46:47]
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s39, v4
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v4
	s_mul_f32 s39, s39, 0x3fb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_2)
	s_xor_b32 s46, s39, 0x80000000
	s_wait_alu depctr_sa_sdst(0)
	v_fma_f32 v5, 0x3fb8aa3b, v4, s46
	s_rndne_f32 s46, s39
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s39, s39, s46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v5, 0x32a5705f, v4
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_add_f32_e32 v5, s39, v5
	s_cvt_i32_f32 s39, s46
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v5, v5
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_1)
	v_ldexp_f32 v5, v5, s39
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, 0x7f800000, v5, vcc_lo
	ds_store_b32 v8, v4 offset:4096
.LBB8_28:                               ;   in Loop: Header=BB8_6 Depth=1
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b32 vcc_lo, exec_lo, s31
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB8_5
; %bb.29:                               ;   in Loop: Header=BB8_6 Depth=1
	s_add_nc_u64 s[44:45], s[44:45], s[34:35]
	v_dual_mov_b32 v5, v3 :: v_dual_mov_b32 v4, v2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[46:47], s[44:45], 2
	s_mov_b64 s[44:45], s[24:25]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[46:47], s[20:21], s[46:47]
	s_mov_b64 s[48:49], s[16:17]
	s_mov_b32 s33, s9
	s_branch .LBB8_32
.LBB8_30:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s51
	global_store_d16_hi_b16 v8, v13, s[44:45]
.LBB8_31:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, vcc_lo, v4, s26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s27, v5, vcc_lo
	s_add_co_i32 s33, s33, -1
	s_add_nc_u64 s[48:49], s[48:49], 2
	s_add_nc_u64 s[44:45], s[44:45], 2
	s_cmp_eq_u32 s33, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB8_5
.LBB8_32:                               ;   Parent Loop BB8_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v13, 0
	s_and_saveexec_b32 s39, s0
	s_cbranch_execz .LBB8_34
; %bb.33:                               ;   in Loop: Header=BB8_32 Depth=2
	global_load_b32 v13, v[4:5], off
	ds_load_b32 v14, v6
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v13, v13, v14
.LBB8_34:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	ds_store_b32 v6, v13 offset:3072
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s2
	s_cbranch_execz .LBB8_36
; %bb.35:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_stride64_b32 v[13:14], v6 offset0:12 offset1:14
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_36:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s3
	s_cbranch_execz .LBB8_38
; %bb.37:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_stride64_b32 v[13:14], v6 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_38:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s4
	s_cbranch_execz .LBB8_40
; %bb.39:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:32
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_40:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s5
	s_cbranch_execz .LBB8_42
; %bb.41:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:16
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_42:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s6
	s_cbranch_execz .LBB8_44
; %bb.43:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:8
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_44:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s7
	s_cbranch_execz .LBB8_46
; %bb.45:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:4
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_46:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s8
	s_cbranch_execz .LBB8_48
; %bb.47:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:2
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_48:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s1
	s_cbranch_execz .LBB8_50
; %bb.49:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v9 offset1:1
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_50:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s1
	s_cbranch_execz .LBB8_52
; %bb.51:                               ;   in Loop: Header=BB8_32 Depth=2
	s_clause 0x1
	global_load_u16 v15, v8, s[48:49]
	global_load_b32 v16, v8, s[46:47]
	ds_load_2addr_stride64_b32 v[13:14], v8 offset0:12 offset1:16
	s_wait_dscnt 0x0
	v_xor_b32_e32 v14, 0x80000000, v14
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v14, v13
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v13, v16, v15
	ds_store_b32 v8, v13 offset:4100
.LBB8_52:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s0
	s_cbranch_execz .LBB8_54
; %bb.53:                               ;   in Loop: Header=BB8_32 Depth=2
	global_load_b32 v15, v[4:5], off
	ds_load_b64 v[13:14], v8 offset:4096
	ds_load_b32 v16, v6
	s_wait_dscnt 0x0
	v_mul_f32_e32 v14, v14, v16
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v13, v15
	global_store_b32 v[4:5], v14, off
.LBB8_54:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v13, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s0
	s_cbranch_execz .LBB8_56
; %bb.55:                               ;   in Loop: Header=BB8_32 Depth=2
	global_load_b32 v13, v[4:5], off
	ds_load_b32 v14, v7
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v13, v13, v14
.LBB8_56:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	ds_store_b32 v6, v13 offset:3072
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s2
	s_cbranch_execz .LBB8_58
; %bb.57:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_stride64_b32 v[13:14], v6 offset0:12 offset1:14
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_58:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s3
	s_cbranch_execz .LBB8_60
; %bb.59:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_stride64_b32 v[13:14], v6 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_60:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s4
	s_cbranch_execz .LBB8_62
; %bb.61:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:32
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_62:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s5
	s_cbranch_execz .LBB8_64
; %bb.63:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:16
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_64:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s6
	s_cbranch_execz .LBB8_66
; %bb.65:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:8
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_66:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s7
	s_cbranch_execz .LBB8_68
; %bb.67:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:4
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_68:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s8
	s_cbranch_execz .LBB8_70
; %bb.69:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v10 offset1:2
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_70:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s1
	s_cbranch_execz .LBB8_72
; %bb.71:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_2addr_b32 v[13:14], v9 offset1:1
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	ds_store_b32 v6, v13 offset:3072
.LBB8_72:                               ;   in Loop: Header=BB8_32 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s39, s1
	s_cbranch_execz .LBB8_31
; %bb.73:                               ;   in Loop: Header=BB8_32 Depth=2
	ds_load_b32 v13, v8 offset:3072
	s_mov_b32 s52, -1
	s_wait_dscnt 0x0
	v_readfirstlane_b32 s50, v13
	s_mul_f32 s50, s29, s50
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_and_b32 s51, s50, 0x7f800000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s51, 0x7f800000
                                        ; implicit-def: $sgpr51
	s_cbranch_scc1 .LBB8_75
; %bb.74:                               ;   in Loop: Header=BB8_32 Depth=2
	s_bfe_u32 s51, s50, 0x10010
	s_mov_b32 s52, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s51, s50, s51
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s51, 0x7fff
.LBB8_75:                               ;   in Loop: Header=BB8_32 Depth=2
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB8_30
; %bb.76:                               ;   in Loop: Header=BB8_32 Depth=2
	s_and_b32 s51, s50, 0xffff
	s_or_b32 s52, s50, 0x10000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s51, 0
	s_cselect_b32 s51, s50, s52
	s_branch .LBB8_30
.LBB8_77:
	s_endpgm
.Lfunc_end8:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff, .Lfunc_end8-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
		.amdhsa_group_segment_fixed_size 4104
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
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 53
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end8-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.num_vgpr, 17
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.numbered_sgpr, 53
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3800
; TotalNumSgprs: 55
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4104 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 55
; NumVGPRsForWavesPerEU: 17
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x34
	s_load_b64 s[8:9], s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[2:3], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e64 s8, v2
	s_cbranch_execz .LBB9_22
; %bb.1:
	s_lshl_b32 s10, ttmp7, 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s10, s9
	s_cbranch_scc1 .LBB9_22
; %bb.2:
	s_load_b256 s[0:7], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	s_cmp_eq_u32 s10, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	v_lshlrev_b64_e32 v[4:5], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_wait_kmcnt 0x0
	v_add_co_u32 v8, vcc_lo, s2, v0
	v_add_co_ci_u32_e64 v9, null, s3, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_add_co_u32 v4, vcc_lo, s2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, s2, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo
	v_add_co_u32 v2, vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo
	s_mov_b32 s3, 0
	s_mov_b32 s2, s8
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s11, s3
	s_cselect_b32 s8, -1, 0
	s_add_nc_u64 s[12:13], s[10:11], -3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s14, s8, exec_lo
	s_cselect_b32 s15, 0, s13
	s_cselect_b32 s14, 0, s12
	s_add_nc_u64 s[16:17], s[10:11], -1
	s_mul_u64 s[18:19], s[14:15], s[2:3]
	s_cselect_b32 s13, s5, s1
	s_cselect_b32 s12, s4, s0
	s_cselect_b32 s14, 1, -2
	s_cselect_b32 s15, 0, -1
	s_cselect_b32 s17, 0, s17
	s_cselect_b32 s16, 2, s16
	s_lshl_b64 s[18:19], s[18:19], 1
	v_add_co_u32 v13, vcc_lo, s0, v0
	s_add_nc_u64 s[20:21], s[14:15], s[10:11]
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_clause 0x2
	global_load_u16 v10, v[8:9], off
	global_load_u16 v11, v[4:5], off
	global_load_u16 v12, v[6:7], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v1, vcc_lo
	s_mul_u64 s[20:21], s[20:21], s[2:3]
	v_add_co_u32 v4, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s19, v1, vcc_lo
	s_lshl_b64 s[18:19], s[20:21], 1
	s_mul_u64 s[16:17], s[16:17], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_lshl_b64 s[16:17], s[16:17], 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v6, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s19, v1, vcc_lo
	s_add_nc_u64 s[16:17], s[12:13], s[16:17]
	global_load_u16 v15, v[4:5], off
	s_mul_u64 s[18:19], s[10:11], s[2:3]
	v_add_co_u32 v8, vcc_lo, s16, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[16:17], s[18:19], 1
	global_load_u16 v6, v[6:7], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v13, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v14, vcc_lo
	global_load_u16 v7, v[8:9], off
	global_load_u16 v8, v[4:5], off
	global_load_u16 v2, v[2:3], off
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v10, 16, v10
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v11, 16, v11
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fma_f32 v3, v10, v3, 0
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 16, v6
	v_dual_fmac_f32 v3, v11, v6 :: v_dual_lshlrev_b32 v12, 16, v12
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v7, 16, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 16, v8
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 16, v2
	v_fmac_f32_e32 v3, v12, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v13, v6
	v_mul_f32_e32 v2, 0xbfb8aa3b, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, 0xbfb8aa3b, v3, -v2
	v_rndne_f32_e32 v7, v2
	v_sub_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v6, v3, 0xb2a5705f, v6
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v3
	v_add_f32_e32 v2, v2, v6
	v_cvt_i32_f32_e32 v6, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v2, v2
	v_ldexp_f32 v2, v2, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, 1.0, v2
	v_div_scale_f32 v6, null, v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v7, v6
	v_fma_f32 v8, -v6, v7, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v8, v7
	v_div_scale_f32 v8, vcc_lo, v3, v2, v3
	v_mul_f32_e32 v9, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v9, v8
	v_fmac_f32_e32 v9, v14, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v6, v9, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v6, v6, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v6, v2, v3
	v_and_b32_e32 v2, 0x7f800000, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s11, vcc_lo
	s_xor_b32 s11, exec_lo, s11
; %bb.3:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
; %bb.4:
	s_and_not1_saveexec_b32 s11, s11
; %bb.5:
	v_and_b32_e32 v2, 0xffff, v3
	v_or_b32_e32 v6, 0x10000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v6, v3, vcc_lo
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s11
	v_add_co_u32 v3, vcc_lo, s6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v1, vcc_lo
	s_or_b32 s6, s10, 1
	v_add_co_u32 v6, vcc_lo, v3, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v2, off
	s_cbranch_scc1 .LBB9_22
; %bb.7:
	s_mov_b32 s7, 0
	s_cmp_lt_u32 s6, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[6:7], -3
	s_cselect_b32 s5, s5, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s19, 0, s19
	s_cselect_b32 s18, 1, s18
	s_cselect_b32 s4, s4, s0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[18:19], s[18:19], s[2:3]
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[14:15], s[18:19], 1
	s_mul_u64 s[6:7], s[6:7], s[2:3]
	s_add_nc_u64 s[4:5], s[4:5], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[12:13], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s5, v1, vcc_lo
	global_load_u16 v14, v[2:3], off
	s_add_nc_u64 s[4:5], s[0:1], s[16:17]
	s_lshl_b64 s[0:1], s[2:3], 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	global_load_u16 v8, v[8:9], off
	v_add_co_u32 v4, vcc_lo, v4, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	s_clause 0x1
	global_load_u16 v9, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v8, 16, v8
	s_wait_loadcnt 0x1
	v_dual_fmac_f32 v14, v11, v8 :: v_dual_lshlrev_b32 v9, 16, v9
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v8, 16, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v9
	v_fmac_f32_e32 v14, v13, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v8, 0xbfb8aa3b, v14
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	v_fma_f32 v9, 0xbfb8aa3b, v14, -v8
	v_rndne_f32_e32 v15, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v9, v14, 0xb2a5705f, v9 :: v_dual_sub_f32 v8, v8, v15
	v_add_f32_e32 v8, v8, v9
	v_cvt_i32_f32_e32 v9, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v8, v8
	v_ldexp_f32 v8, v8, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, 0x7f800000, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v8, 1.0, v8
	v_div_scale_f32 v9, null, v8, v8, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v15, v9
	v_fma_f32 v16, -v9, v15, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v15
	v_div_scale_f32 v16, vcc_lo, v14, v8, v14
	v_mul_f32_e32 v17, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v9, v17, v16
	v_fmac_f32_e32 v17, v18, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v9, -v9, v17, v16
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v9, v9, v15, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v9, v9, v8, v14
	v_and_b32_e32 v8, 0x7f800000, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v8
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b32 s6, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
; %bb.8:
	v_bfe_u32 v8, v9, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v8, v9, v8, 0x7fff
                                        ; implicit-def: $vgpr9
; %bb.9:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
; %bb.10:
	v_and_b32_e32 v8, 0xffff, v9
	v_or_b32_e32 v14, 0x10000, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v14, v9, vcc_lo
; %bb.11:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s6, s10, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v8, off
	s_cbranch_scc1 .LBB9_22
; %bb.12:
	s_mov_b32 s7, 0
	s_and_b32 s8, s8, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[6:7], -3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s7, 0, s7
	s_cselect_b32 s6, 2, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[6:7], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s3, v1, vcc_lo
	s_add_nc_u64 s[2:3], s[4:5], s[0:1]
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[8:9], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s3, v1, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[8:9], off
	global_load_u16 v17, v[4:5], off
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v15, v10, v15, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v14, 16, v14
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v15, v11, v14 :: v_dual_lshlrev_b32 v14, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v12, v16
	v_fmac_f32_e32 v15, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v15
	v_fma_f32 v16, 0xbfb8aa3b, v15, -v14
	v_rndne_f32_e32 v17, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmamk_f32 v16, v15, 0xb2a5705f, v16
	v_sub_f32_e32 v14, v14, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v14, v14, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v15
	v_exp_f32_e32 v14, v14
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v14, v14, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v14, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v14, 0x7f800000, v14, vcc_lo
	v_add_f32_e32 v14, 1.0, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v16, null, v14, v14, v15
	v_rcp_f32_e32 v17, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v16, v17, 1.0
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v15, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v19, v18, v17
	v_fma_f32 v20, -v16, v19, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v20, v17
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v16, v16, v17, v19
	v_div_fixup_f32 v15, v16, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v14, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.13:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.14:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.15:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB9_22
; %bb.17:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v8, v[8:9], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, v4, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v5, vcc_lo
	s_clause 0x1
	global_load_u16 v0, v[0:1], off
	global_load_u16 v1, v[2:3], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v2, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v2, v10, v2, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	v_lshlrev_b32_e32 v3, 16, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v11, v3
	v_fmac_f32_e32 v2, v12, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v13, v1
	v_mul_f32_e32 v0, 0xbfb8aa3b, v2
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v1, 0xbfb8aa3b, v2, -v0
	v_rndne_f32_e32 v3, v0
	v_dual_fmamk_f32 v1, v2, 0xb2a5705f, v1 :: v_dual_sub_f32 v0, v0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v0, v0, v1
	v_cvt_i32_f32_e32 v1, v3
	v_exp_f32_e32 v0, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v0, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0x7f800000, v0, vcc_lo
	v_add_f32_e32 v0, 1.0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v1, null, v0, v0, v2
	v_rcp_f32_e32 v3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v1, v3, 1.0
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, vcc_lo, v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, v4, v3
	v_fma_f32 v8, -v1, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v8, v3
	v_fma_f32 v1, -v1, v5, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v3, v5
	v_div_fixup_f32 v1, v1, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s2
; %bb.18:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.19:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s2
; %bb.20:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_co_u32 v1, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s1, v7, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
.LBB9_22:
	s_endpgm
.Lfunc_end9:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end9-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
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
		.amdhsa_next_free_vgpr 21
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end9-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_vgpr, 21
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.numbered_sgpr, 22
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2764
; TotalNumSgprs: 24
; NumVgprs: 21
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 21
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x34
	s_load_b64 s[8:9], s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[2:3], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e64 s8, v2
	s_cbranch_execz .LBB10_42
; %bb.1:
	s_lshl_b32 s10, ttmp7, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s10, s9
	s_cbranch_scc1 .LBB10_42
; %bb.2:
	s_load_b256 s[0:7], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	s_cmp_eq_u32 s10, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	v_lshlrev_b64_e32 v[4:5], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_wait_kmcnt 0x0
	v_add_co_u32 v8, vcc_lo, s2, v0
	v_add_co_ci_u32_e64 v9, null, s3, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_add_co_u32 v4, vcc_lo, s2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, s2, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo
	v_add_co_u32 v2, vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo
	s_mov_b32 s3, 0
	s_mov_b32 s2, s8
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s11, s3
	s_cselect_b32 s8, -1, 0
	s_add_nc_u64 s[12:13], s[10:11], -3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s14, s8, exec_lo
	s_cselect_b32 s15, 0, s13
	s_cselect_b32 s14, 0, s12
	s_add_nc_u64 s[16:17], s[10:11], -1
	s_mul_u64 s[18:19], s[14:15], s[2:3]
	s_cselect_b32 s13, s5, s1
	s_cselect_b32 s12, s4, s0
	s_cselect_b32 s14, 1, -2
	s_cselect_b32 s15, 0, -1
	s_cselect_b32 s17, 0, s17
	s_cselect_b32 s16, 2, s16
	s_lshl_b64 s[18:19], s[18:19], 1
	v_add_co_u32 v13, vcc_lo, s0, v0
	s_add_nc_u64 s[20:21], s[14:15], s[10:11]
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_clause 0x2
	global_load_u16 v10, v[8:9], off
	global_load_u16 v11, v[4:5], off
	global_load_u16 v12, v[6:7], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v1, vcc_lo
	s_mul_u64 s[20:21], s[20:21], s[2:3]
	v_add_co_u32 v4, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s19, v1, vcc_lo
	s_lshl_b64 s[18:19], s[20:21], 1
	s_mul_u64 s[16:17], s[16:17], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_lshl_b64 s[16:17], s[16:17], 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v6, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s19, v1, vcc_lo
	s_add_nc_u64 s[16:17], s[12:13], s[16:17]
	global_load_u16 v15, v[4:5], off
	s_mul_u64 s[18:19], s[10:11], s[2:3]
	v_add_co_u32 v8, vcc_lo, s16, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[16:17], s[18:19], 1
	global_load_u16 v6, v[6:7], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v13, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v14, vcc_lo
	global_load_u16 v7, v[8:9], off
	global_load_u16 v8, v[4:5], off
	global_load_u16 v2, v[2:3], off
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v10, 16, v10
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v11, 16, v11
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fma_f32 v3, v10, v3, 0
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 16, v6
	v_dual_fmac_f32 v3, v11, v6 :: v_dual_lshlrev_b32 v12, 16, v12
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v7, 16, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 16, v8
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 16, v2
	v_fmac_f32_e32 v3, v12, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v13, v6
	v_mul_f32_e32 v2, 0xbfb8aa3b, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, 0xbfb8aa3b, v3, -v2
	v_rndne_f32_e32 v7, v2
	v_sub_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v6, v3, 0xb2a5705f, v6
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v3
	v_add_f32_e32 v2, v2, v6
	v_cvt_i32_f32_e32 v6, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v2, v2
	v_ldexp_f32 v2, v2, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, 1.0, v2
	v_div_scale_f32 v6, null, v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v7, v6
	v_fma_f32 v8, -v6, v7, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v8, v7
	v_div_scale_f32 v8, vcc_lo, v3, v2, v3
	v_mul_f32_e32 v9, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v9, v8
	v_fmac_f32_e32 v9, v14, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v6, v9, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v6, v6, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v6, v2, v3
	v_and_b32_e32 v2, 0x7f800000, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s11, vcc_lo
	s_xor_b32 s11, exec_lo, s11
; %bb.3:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
; %bb.4:
	s_and_not1_saveexec_b32 s11, s11
; %bb.5:
	v_and_b32_e32 v2, 0xffff, v3
	v_or_b32_e32 v6, 0x10000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v6, v3, vcc_lo
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s11
	v_add_co_u32 v3, vcc_lo, s6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v1, vcc_lo
	s_or_b32 s6, s10, 1
	v_add_co_u32 v6, vcc_lo, v3, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v2, off
	s_cbranch_scc1 .LBB10_42
; %bb.7:
	s_mov_b32 s7, 0
	s_cmp_lt_u32 s6, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[6:7], -3
	s_cselect_b32 s5, s5, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s19, 0, s19
	s_cselect_b32 s18, 1, s18
	s_cselect_b32 s4, s4, s0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[18:19], s[18:19], s[2:3]
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[14:15], s[18:19], 1
	s_mul_u64 s[6:7], s[6:7], s[2:3]
	s_add_nc_u64 s[4:5], s[4:5], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[12:13], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s5, v1, vcc_lo
	s_add_nc_u64 s[4:5], s[0:1], s[16:17]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_lshl_b64 s[0:1], s[2:3], 1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	global_load_u16 v15, v[8:9], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, v4, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v5, vcc_lo
	s_clause 0x1
	global_load_u16 v4, v[2:3], off
	global_load_u16 v5, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 16, v4
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v5, 16, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v4
	v_fmac_f32_e32 v14, v13, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, 0xbfb8aa3b, v14
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	v_fma_f32 v5, 0xbfb8aa3b, v14, -v4
	v_rndne_f32_e32 v15, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v5, v14, 0xb2a5705f, v5 :: v_dual_sub_f32 v4, v4, v15
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, 1.0, v4
	v_div_scale_f32 v5, null, v4, v4, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v15, v5
	v_fma_f32 v16, -v5, v15, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v15
	v_div_scale_f32 v16, vcc_lo, v14, v4, v14
	v_mul_f32_e32 v17, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v5, v17, v16
	v_fmac_f32_e32 v17, v18, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v5, v17, v16
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v5, v5, v15, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v5, v5, v4, v14
	v_and_b32_e32 v4, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b32 s6, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
; %bb.8:
	v_bfe_u32 v4, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v4, v5, v4, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.9:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
; %bb.10:
	v_and_b32_e32 v4, 0xffff, v5
	v_or_b32_e32 v14, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v14, v5, vcc_lo
; %bb.11:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s6, s10, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v4, off
	s_cbranch_scc1 .LBB10_42
; %bb.12:
	s_mov_b32 s7, 0
	s_and_b32 s8, s8, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[6:7], -3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s7, 0, s7
	s_cselect_b32 s6, 2, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[6:7], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	s_add_nc_u64 s[2:3], s[4:5], s[0:1]
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v15, v10, v15, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v14, 16, v14
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v15, v11, v14 :: v_dual_lshlrev_b32 v14, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v12, v16
	v_fmac_f32_e32 v15, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v15
	v_fma_f32 v16, 0xbfb8aa3b, v15, -v14
	v_rndne_f32_e32 v17, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmamk_f32 v16, v15, 0xb2a5705f, v16
	v_sub_f32_e32 v14, v14, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v14, v14, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v15
	v_exp_f32_e32 v14, v14
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v14, v14, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v14, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v14, 0x7f800000, v14, vcc_lo
	v_add_f32_e32 v14, 1.0, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v16, null, v14, v14, v15
	v_rcp_f32_e32 v17, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v16, v17, 1.0
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v15, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v19, v18, v17
	v_fma_f32 v20, -v16, v19, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v20, v17
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v16, v16, v17, v19
	v_div_fixup_f32 v15, v16, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v14, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.13:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.14:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.15:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB10_42
; %bb.17:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.18:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.19:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.20:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB10_42
; %bb.22:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.23:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.24:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.25:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB10_42
; %bb.27:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.28:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.29:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.30:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB10_42
; %bb.32:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.33:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.35:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB10_42
; %bb.37:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v4, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v0, v[0:1], off
	global_load_u16 v1, v[2:3], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v2, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v2, v10, v2, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	v_lshlrev_b32_e32 v3, 16, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v11, v3
	v_fmac_f32_e32 v2, v12, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v13, v1
	v_mul_f32_e32 v0, 0xbfb8aa3b, v2
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v1, 0xbfb8aa3b, v2, -v0
	v_rndne_f32_e32 v3, v0
	v_dual_fmamk_f32 v1, v2, 0xb2a5705f, v1 :: v_dual_sub_f32 v0, v0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v0, v0, v1
	v_cvt_i32_f32_e32 v1, v3
	v_exp_f32_e32 v0, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v0, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0x7f800000, v0, vcc_lo
	v_add_f32_e32 v0, 1.0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v1, null, v0, v0, v2
	v_rcp_f32_e32 v3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v1, v3, 1.0
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, vcc_lo, v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, v4, v3
	v_fma_f32 v8, -v1, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v8, v3
	v_fma_f32 v1, -v1, v5, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v3, v5
	v_div_fixup_f32 v1, v1, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s2
; %bb.38:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.39:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s2
; %bb.40:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_co_u32 v1, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s1, v7, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
.LBB10_42:
	s_endpgm
.Lfunc_end10:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end10-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
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
		.amdhsa_next_free_vgpr 21
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end10-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_vgpr, 21
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.numbered_sgpr, 22
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4860
; TotalNumSgprs: 24
; NumVgprs: 21
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 21
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x34
	s_load_b64 s[8:9], s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[2:3], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e64 s8, v2
	s_cbranch_execz .LBB11_82
; %bb.1:
	s_lshl_b32 s10, ttmp7, 4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s10, s9
	s_cbranch_scc1 .LBB11_82
; %bb.2:
	s_load_b256 s[0:7], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	s_cmp_eq_u32 s10, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	v_lshlrev_b64_e32 v[4:5], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_wait_kmcnt 0x0
	v_add_co_u32 v8, vcc_lo, s2, v0
	v_add_co_ci_u32_e64 v9, null, s3, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_add_co_u32 v4, vcc_lo, s2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, s2, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo
	v_add_co_u32 v2, vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo
	s_mov_b32 s3, 0
	s_mov_b32 s2, s8
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s11, s3
	s_cselect_b32 s8, -1, 0
	s_add_nc_u64 s[12:13], s[10:11], -3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s14, s8, exec_lo
	s_cselect_b32 s15, 0, s13
	s_cselect_b32 s14, 0, s12
	s_add_nc_u64 s[16:17], s[10:11], -1
	s_mul_u64 s[18:19], s[14:15], s[2:3]
	s_cselect_b32 s13, s5, s1
	s_cselect_b32 s12, s4, s0
	s_cselect_b32 s14, 1, -2
	s_cselect_b32 s15, 0, -1
	s_cselect_b32 s17, 0, s17
	s_cselect_b32 s16, 2, s16
	s_lshl_b64 s[18:19], s[18:19], 1
	v_add_co_u32 v13, vcc_lo, s0, v0
	s_add_nc_u64 s[20:21], s[14:15], s[10:11]
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_clause 0x2
	global_load_u16 v10, v[8:9], off
	global_load_u16 v11, v[4:5], off
	global_load_u16 v12, v[6:7], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v1, vcc_lo
	s_mul_u64 s[20:21], s[20:21], s[2:3]
	v_add_co_u32 v4, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s19, v1, vcc_lo
	s_lshl_b64 s[18:19], s[20:21], 1
	s_mul_u64 s[16:17], s[16:17], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_lshl_b64 s[16:17], s[16:17], 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v6, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s19, v1, vcc_lo
	s_add_nc_u64 s[16:17], s[12:13], s[16:17]
	global_load_u16 v15, v[4:5], off
	s_mul_u64 s[18:19], s[10:11], s[2:3]
	v_add_co_u32 v8, vcc_lo, s16, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[16:17], s[18:19], 1
	global_load_u16 v6, v[6:7], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v13, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v14, vcc_lo
	global_load_u16 v7, v[8:9], off
	global_load_u16 v8, v[4:5], off
	global_load_u16 v2, v[2:3], off
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v10, 16, v10
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v11, 16, v11
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fma_f32 v3, v10, v3, 0
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 16, v6
	v_dual_fmac_f32 v3, v11, v6 :: v_dual_lshlrev_b32 v12, 16, v12
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v7, 16, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 16, v8
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 16, v2
	v_fmac_f32_e32 v3, v12, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v13, v6
	v_mul_f32_e32 v2, 0xbfb8aa3b, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, 0xbfb8aa3b, v3, -v2
	v_rndne_f32_e32 v7, v2
	v_sub_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v6, v3, 0xb2a5705f, v6
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v3
	v_add_f32_e32 v2, v2, v6
	v_cvt_i32_f32_e32 v6, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v2, v2
	v_ldexp_f32 v2, v2, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, 1.0, v2
	v_div_scale_f32 v6, null, v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v7, v6
	v_fma_f32 v8, -v6, v7, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v8, v7
	v_div_scale_f32 v8, vcc_lo, v3, v2, v3
	v_mul_f32_e32 v9, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v9, v8
	v_fmac_f32_e32 v9, v14, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v6, v9, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v6, v6, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v6, v2, v3
	v_and_b32_e32 v2, 0x7f800000, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s11, vcc_lo
	s_xor_b32 s11, exec_lo, s11
; %bb.3:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
; %bb.4:
	s_and_not1_saveexec_b32 s11, s11
; %bb.5:
	v_and_b32_e32 v2, 0xffff, v3
	v_or_b32_e32 v6, 0x10000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v6, v3, vcc_lo
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s11
	v_add_co_u32 v3, vcc_lo, s6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v1, vcc_lo
	s_or_b32 s6, s10, 1
	v_add_co_u32 v6, vcc_lo, v3, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v2, off
	s_cbranch_scc1 .LBB11_82
; %bb.7:
	s_mov_b32 s7, 0
	s_cmp_lt_u32 s6, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[6:7], -3
	s_cselect_b32 s5, s5, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s19, 0, s19
	s_cselect_b32 s18, 1, s18
	s_cselect_b32 s4, s4, s0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[18:19], s[18:19], s[2:3]
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[14:15], s[18:19], 1
	s_mul_u64 s[6:7], s[6:7], s[2:3]
	s_add_nc_u64 s[4:5], s[4:5], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[12:13], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s5, v1, vcc_lo
	s_add_nc_u64 s[4:5], s[0:1], s[16:17]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_lshl_b64 s[0:1], s[2:3], 1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	global_load_u16 v15, v[8:9], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, v4, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v5, vcc_lo
	s_clause 0x1
	global_load_u16 v4, v[2:3], off
	global_load_u16 v5, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 16, v4
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v5, 16, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v4
	v_fmac_f32_e32 v14, v13, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, 0xbfb8aa3b, v14
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	v_fma_f32 v5, 0xbfb8aa3b, v14, -v4
	v_rndne_f32_e32 v15, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v5, v14, 0xb2a5705f, v5 :: v_dual_sub_f32 v4, v4, v15
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, 1.0, v4
	v_div_scale_f32 v5, null, v4, v4, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v15, v5
	v_fma_f32 v16, -v5, v15, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v15
	v_div_scale_f32 v16, vcc_lo, v14, v4, v14
	v_mul_f32_e32 v17, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v5, v17, v16
	v_fmac_f32_e32 v17, v18, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v5, v17, v16
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v5, v5, v15, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v5, v5, v4, v14
	v_and_b32_e32 v4, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b32 s6, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
; %bb.8:
	v_bfe_u32 v4, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v4, v5, v4, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.9:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
; %bb.10:
	v_and_b32_e32 v4, 0xffff, v5
	v_or_b32_e32 v14, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v14, v5, vcc_lo
; %bb.11:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s6, s10, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v4, off
	s_cbranch_scc1 .LBB11_82
; %bb.12:
	s_mov_b32 s7, 0
	s_and_b32 s8, s8, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[6:7], -3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s7, 0, s7
	s_cselect_b32 s6, 2, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[6:7], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	s_add_nc_u64 s[2:3], s[4:5], s[0:1]
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v15, v10, v15, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v14, 16, v14
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v15, v11, v14 :: v_dual_lshlrev_b32 v14, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v12, v16
	v_fmac_f32_e32 v15, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v15
	v_fma_f32 v16, 0xbfb8aa3b, v15, -v14
	v_rndne_f32_e32 v17, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmamk_f32 v16, v15, 0xb2a5705f, v16
	v_sub_f32_e32 v14, v14, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v14, v14, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v15
	v_exp_f32_e32 v14, v14
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v14, v14, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v14, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v14, 0x7f800000, v14, vcc_lo
	v_add_f32_e32 v14, 1.0, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v16, null, v14, v14, v15
	v_rcp_f32_e32 v17, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v16, v17, 1.0
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v15, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v19, v18, v17
	v_fma_f32 v20, -v16, v19, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v20, v17
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v16, v16, v17, v19
	v_div_fixup_f32 v15, v16, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v14, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.13:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.14:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.15:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.17:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.18:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.19:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.20:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.22:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.23:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.24:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.25:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.27:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.28:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.29:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.30:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.32:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.33:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.35:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.37:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.38:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.39:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.40:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.42:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.43:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.45:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 9
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.47:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.48:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.49:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.50:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 10
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.52:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.53:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.55:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 11
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.57:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.58:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.59:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.60:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 12
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.62:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.63:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.64:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.65:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.67:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.68:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.69:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.70:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 14
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.72:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.73:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.74:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.75:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB11_82
; %bb.77:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v9, vcc_lo
	global_load_u16 v4, v[4:5], off
	global_load_u16 v0, v[0:1], off
	global_load_u16 v1, v[2:3], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v2, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v2, v10, v2, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	v_lshlrev_b32_e32 v3, 16, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v11, v3
	v_fmac_f32_e32 v2, v12, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v13, v1
	v_mul_f32_e32 v0, 0xbfb8aa3b, v2
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v1, 0xbfb8aa3b, v2, -v0
	v_rndne_f32_e32 v3, v0
	v_dual_fmamk_f32 v1, v2, 0xb2a5705f, v1 :: v_dual_sub_f32 v0, v0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v0, v0, v1
	v_cvt_i32_f32_e32 v1, v3
	v_exp_f32_e32 v0, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v0, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0x7f800000, v0, vcc_lo
	v_add_f32_e32 v0, 1.0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v1, null, v0, v0, v2
	v_rcp_f32_e32 v3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v1, v3, 1.0
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, vcc_lo, v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, v4, v3
	v_fma_f32 v8, -v1, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v8, v3
	v_fma_f32 v1, -v1, v5, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v3, v5
	v_div_fixup_f32 v1, v1, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s2
; %bb.78:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.79:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s2
; %bb.80:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_co_u32 v1, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s1, v7, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
.LBB11_82:
	s_endpgm
.Lfunc_end11:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end11-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
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
		.amdhsa_next_free_vgpr 21
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end11-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_vgpr, 21
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.numbered_sgpr, 22
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9004
; TotalNumSgprs: 24
; NumVgprs: 21
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 21
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x34
	s_load_b64 s[8:9], s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[2:3], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e64 s8, v2
	s_cbranch_execz .LBB12_162
; %bb.1:
	s_lshl_b32 s10, ttmp7, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s10, s9
	s_cbranch_scc1 .LBB12_162
; %bb.2:
	s_load_b256 s[0:7], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	s_cmp_eq_u32 s10, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	v_lshlrev_b64_e32 v[4:5], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 1, v[2:3]
	v_add_nc_u32_e32 v2, s8, v2
	s_wait_kmcnt 0x0
	v_add_co_u32 v8, vcc_lo, s2, v0
	v_add_co_ci_u32_e64 v9, null, s3, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_add_co_u32 v4, vcc_lo, s2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, s2, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo
	v_add_co_u32 v2, vcc_lo, s2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo
	s_mov_b32 s3, 0
	s_mov_b32 s2, s8
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s11, s3
	s_cselect_b32 s8, -1, 0
	s_add_nc_u64 s[12:13], s[10:11], -3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s14, s8, exec_lo
	s_cselect_b32 s15, 0, s13
	s_cselect_b32 s14, 0, s12
	s_add_nc_u64 s[16:17], s[10:11], -1
	s_mul_u64 s[18:19], s[14:15], s[2:3]
	s_cselect_b32 s13, s5, s1
	s_cselect_b32 s12, s4, s0
	s_cselect_b32 s14, 1, -2
	s_cselect_b32 s15, 0, -1
	s_cselect_b32 s17, 0, s17
	s_cselect_b32 s16, 2, s16
	s_lshl_b64 s[18:19], s[18:19], 1
	v_add_co_u32 v13, vcc_lo, s0, v0
	s_add_nc_u64 s[20:21], s[14:15], s[10:11]
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_clause 0x2
	global_load_u16 v10, v[8:9], off
	global_load_u16 v11, v[4:5], off
	global_load_u16 v12, v[6:7], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v1, vcc_lo
	s_mul_u64 s[20:21], s[20:21], s[2:3]
	v_add_co_u32 v4, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s19, v1, vcc_lo
	s_lshl_b64 s[18:19], s[20:21], 1
	s_mul_u64 s[16:17], s[16:17], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[12:13], s[18:19]
	s_lshl_b64 s[16:17], s[16:17], 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v6, vcc_lo, s18, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s19, v1, vcc_lo
	s_add_nc_u64 s[16:17], s[12:13], s[16:17]
	global_load_u16 v15, v[4:5], off
	s_mul_u64 s[18:19], s[10:11], s[2:3]
	v_add_co_u32 v8, vcc_lo, s16, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[16:17], s[18:19], 1
	global_load_u16 v6, v[6:7], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v13, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v14, vcc_lo
	global_load_u16 v7, v[8:9], off
	global_load_u16 v8, v[4:5], off
	global_load_u16 v2, v[2:3], off
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v10, 16, v10
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v11, 16, v11
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fma_f32 v3, v10, v3, 0
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v6, 16, v6
	v_dual_fmac_f32 v3, v11, v6 :: v_dual_lshlrev_b32 v12, 16, v12
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v7, 16, v7
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v6, 16, v8
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v13, 16, v2
	v_fmac_f32_e32 v3, v12, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v13, v6
	v_mul_f32_e32 v2, 0xbfb8aa3b, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, 0xbfb8aa3b, v3, -v2
	v_rndne_f32_e32 v7, v2
	v_sub_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v6, v3, 0xb2a5705f, v6
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v3
	v_add_f32_e32 v2, v2, v6
	v_cvt_i32_f32_e32 v6, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v2, v2
	v_ldexp_f32 v2, v2, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, 1.0, v2
	v_div_scale_f32 v6, null, v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v7, v6
	v_fma_f32 v8, -v6, v7, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v8, v7
	v_div_scale_f32 v8, vcc_lo, v3, v2, v3
	v_mul_f32_e32 v9, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v9, v8
	v_fmac_f32_e32 v9, v14, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v6, v9, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v6, v6, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v6, v2, v3
	v_and_b32_e32 v2, 0x7f800000, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s11, vcc_lo
	s_xor_b32 s11, exec_lo, s11
; %bb.3:
	v_bfe_u32 v2, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v2, v3, v2, 0x7fff
                                        ; implicit-def: $vgpr3
; %bb.4:
	s_and_not1_saveexec_b32 s11, s11
; %bb.5:
	v_and_b32_e32 v2, 0xffff, v3
	v_or_b32_e32 v6, 0x10000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v6, v3, vcc_lo
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s11
	v_add_co_u32 v3, vcc_lo, s6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v1, vcc_lo
	s_or_b32 s6, s10, 1
	v_add_co_u32 v6, vcc_lo, v3, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v2, off
	s_cbranch_scc1 .LBB12_162
; %bb.7:
	s_mov_b32 s7, 0
	s_cmp_lt_u32 s6, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[6:7], -3
	s_cselect_b32 s5, s5, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s19, 0, s19
	s_cselect_b32 s18, 1, s18
	s_cselect_b32 s4, s4, s0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[18:19], s[18:19], s[2:3]
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[14:15], s[18:19], 1
	s_mul_u64 s[6:7], s[6:7], s[2:3]
	s_add_nc_u64 s[4:5], s[4:5], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[12:13], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s5, v1, vcc_lo
	s_add_nc_u64 s[4:5], s[0:1], s[16:17]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_lshl_b64 s[0:1], s[2:3], 1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s5, v1, vcc_lo
	global_load_u16 v15, v[8:9], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, v4, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v5, vcc_lo
	s_clause 0x1
	global_load_u16 v4, v[2:3], off
	global_load_u16 v5, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 16, v4
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v5, 16, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v4
	v_fmac_f32_e32 v14, v13, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, 0xbfb8aa3b, v14
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	v_fma_f32 v5, 0xbfb8aa3b, v14, -v4
	v_rndne_f32_e32 v15, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v5, v14, 0xb2a5705f, v5 :: v_dual_sub_f32 v4, v4, v15
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, 1.0, v4
	v_div_scale_f32 v5, null, v4, v4, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v15, v5
	v_fma_f32 v16, -v5, v15, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v15
	v_div_scale_f32 v16, vcc_lo, v14, v4, v14
	v_mul_f32_e32 v17, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v5, v17, v16
	v_fmac_f32_e32 v17, v18, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v5, v17, v16
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v5, v5, v15, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v5, v5, v4, v14
	v_and_b32_e32 v4, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b32 s6, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
; %bb.8:
	v_bfe_u32 v4, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v4, v5, v4, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.9:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
; %bb.10:
	v_and_b32_e32 v4, 0xffff, v5
	v_or_b32_e32 v14, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v14, v5, vcc_lo
; %bb.11:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s6, s10, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s6, s9
	global_store_d16_hi_b16 v[6:7], v4, off
	s_cbranch_scc1 .LBB12_162
; %bb.12:
	s_mov_b32 s7, 0
	s_and_b32 s8, s8, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[6:7], -3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s7, 0, s7
	s_cselect_b32 s6, 2, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[6:7], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	s_add_nc_u64 s[2:3], s[4:5], s[0:1]
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v15, v10, v15, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v14, 16, v14
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v15, v11, v14 :: v_dual_lshlrev_b32 v14, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v12, v16
	v_fmac_f32_e32 v15, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v15
	v_fma_f32 v16, 0xbfb8aa3b, v15, -v14
	v_rndne_f32_e32 v17, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmamk_f32 v16, v15, 0xb2a5705f, v16
	v_sub_f32_e32 v14, v14, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v14, v14, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v15
	v_exp_f32_e32 v14, v14
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v14, v14, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v14, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v14, 0x7f800000, v14, vcc_lo
	v_add_f32_e32 v14, 1.0, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v16, null, v14, v14, v15
	v_rcp_f32_e32 v17, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v16, v17, 1.0
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v15, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v19, v18, v17
	v_fma_f32 v20, -v16, v19, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v20, v17
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v16, v16, v17, v19
	v_div_fixup_f32 v15, v16, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v14, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.13:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.14:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.15:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.17:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.18:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.19:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.20:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.22:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.23:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.24:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.25:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.27:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.28:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.29:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.30:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.32:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.33:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.35:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.37:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.38:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.39:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.40:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.42:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x1
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.43:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.45:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 9
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.47:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[2:3], off
	global_load_u16 v15, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.48:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.49:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.50:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 10
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.52:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	s_clause 0x1
	global_load_u16 v14, v[4:5], off
	global_load_u16 v15, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.53:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.55:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 11
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.57:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.58:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.59:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.60:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 12
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.62:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.63:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.64:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.65:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.67:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.68:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.69:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.70:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 14
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.72:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.73:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.74:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.75:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.77:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.78:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.79:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.80:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.82:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.83:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.84:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.85:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 17
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.87:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.88:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.89:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.90:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 18
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.92:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.93:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.94:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.95:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 19
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.97:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.98:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.99:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.100:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 20
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.102:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.103:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.104:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.105:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 21
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.107:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.108:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.109:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.110:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.111:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 22
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.112:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.113:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.114:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.115:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 23
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.117:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.118:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.119:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.120:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 24
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.122:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.123:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.124:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.125:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 25
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.127:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.128:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.129:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.130:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 26
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.132:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.133:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.134:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.135:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 27
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.137:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.138:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.139:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.140:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.141:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 28
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.142:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.143:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.144:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.145:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 29
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.147:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v2, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[4:5], off
	global_load_u16 v16, v[2:3], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.148:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.149:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.150:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.151:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 30
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.152:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[4:5], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	global_load_u16 v15, v[2:3], off
	global_load_u16 v16, v[4:5], off
	global_load_u16 v17, v[8:9], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fma_f32 v14, v10, v14, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v16, 16, v16
	v_lshlrev_b32_e32 v15, 16, v15
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v14, v11, v15 :: v_dual_lshlrev_b32 v15, 16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v12, v16
	v_fmac_f32_e32 v14, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v14
	v_fma_f32 v16, 0xbfb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v16, v14, 0xb2a5705f, v16 :: v_dual_sub_f32 v15, v15, v17
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, 1.0, v15
	v_div_scale_f32 v16, null, v15, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v16
	v_fma_f32 v18, -v16, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_div_scale_f32 v18, vcc_lo, v14, v15, v14
	v_mul_f32_e32 v19, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v16, v19, v18
	v_fmac_f32_e32 v19, v20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v19, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v16, v15, v14
	v_and_b32_e32 v14, 0x7f800000, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.153:
	v_bfe_u32 v14, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v15, v14, 0x7fff
                                        ; implicit-def: $vgpr15
; %bb.154:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.155:
	v_and_b32_e32 v14, 0xffff, v15
	v_or_b32_e32 v16, 0x10000, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v16, v15, vcc_lo
; %bb.156:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_co_u32 v6, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_or_b32 s4, s10, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s4, s9
	global_store_d16_hi_b16 v[6:7], v14, off
	s_cbranch_scc1 .LBB12_162
; %bb.157:
	s_add_nc_u64 s[2:3], s[2:3], s[0:1]
	global_load_u16 v14, v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, v8, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v9, vcc_lo
	global_load_u16 v4, v[4:5], off
	global_load_u16 v0, v[0:1], off
	global_load_u16 v1, v[2:3], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v2, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v2, v10, v2, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	v_lshlrev_b32_e32 v3, 16, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v11, v3
	v_fmac_f32_e32 v2, v12, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v13, v1
	v_mul_f32_e32 v0, 0xbfb8aa3b, v2
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v1, 0xbfb8aa3b, v2, -v0
	v_rndne_f32_e32 v3, v0
	v_dual_fmamk_f32 v1, v2, 0xb2a5705f, v1 :: v_dual_sub_f32 v0, v0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v0, v0, v1
	v_cvt_i32_f32_e32 v1, v3
	v_exp_f32_e32 v0, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v0, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0x7f800000, v0, vcc_lo
	v_add_f32_e32 v0, 1.0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v1, null, v0, v0, v2
	v_rcp_f32_e32 v3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v1, v3, 1.0
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, vcc_lo, v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, v4, v3
	v_fma_f32 v8, -v1, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v8, v3
	v_fma_f32 v1, -v1, v5, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v3, v5
	v_div_fixup_f32 v1, v1, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s2
; %bb.158:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.159:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s2
; %bb.160:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.161:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_co_u32 v1, vcc_lo, v6, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s1, v7, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
.LBB12_162:
	s_endpgm
.Lfunc_end12:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end12-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
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
		.amdhsa_next_free_vgpr 21
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end12-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_vgpr, 21
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.numbered_sgpr, 22
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 17260
; TotalNumSgprs: 24
; NumVgprs: 21
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 21
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_,comdat
	.globl	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_ ; -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.p2align	8
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_: ; @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x4c
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[13:14], null, ttmp9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x2800, v13
	s_cbranch_execz .LBB13_179
; %bb.1:
	s_load_b512 s[0:15], s[0:1], 0x0
	v_mov_b32_e32 v14, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 1, v[13:14]
	s_wait_kmcnt 0x0
	v_add_co_u32 v4, vcc_lo, s4, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s5, v8, vcc_lo
	v_add_co_u32 v16, vcc_lo, s0, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s1, v8, vcc_lo
	s_clause 0x3
	global_load_d16_b16 v3, v[4:5], off
	global_load_d16_b16 v2, v[4:5], off offset:20480
	global_load_d16_b16 v1, v[4:5], off offset:40960
	global_load_d16_b16 v0, v[4:5], off offset:61440
	v_subrev_co_u32 v4, s5, 0x1000, v13
	v_mov_b32_e32 v5, v14
	v_add_co_u32 v9, vcc_lo, s6, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s7, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]
	s_lshl_b32 s4, ttmp7, 2
	s_xor_b32 s16, s5, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s4, 2
	s_mov_b32 s0, -1
	v_add_co_u32 v11, vcc_lo, s2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s3, v5, vcc_lo
                                        ; implicit-def: $vgpr4_vgpr5
	s_cbranch_scc1 .LBB13_9
; %bb.2:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB13_14
.LBB13_3:
	global_load_d16_b16 v4, v[4:5], off
	s_cmp_gt_i32 s4, 1
	s_cbranch_scc0 .LBB13_15
.LBB13_4:
	s_add_co_i32 s0, s4, -2
	s_mov_b32 s1, 0
                                        ; implicit-def: $vgpr5_vgpr6
	s_and_saveexec_b32 s5, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
; %bb.5:
	v_mad_co_u64_u32 v[5:6], null, 0x6000, s0, v[11:12]
; %bb.6:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
; %bb.7:
	s_lshl_b64 s[0:1], s[0:1], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, vcc_lo, v16, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s1, v17, vcc_lo
; %bb.8:
	s_or_b32 exec_lo, exec_lo, s5
	s_cbranch_execz .LBB13_16
	s_branch .LBB13_17
.LBB13_9:
	s_add_co_i32 s0, s4, -3
	s_mov_b32 s1, 0
                                        ; implicit-def: $vgpr4_vgpr5
	s_and_saveexec_b32 s5, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
; %bb.10:
	v_mad_co_u64_u32 v[4:5], null, 0x6000, s0, v[11:12]
; %bb.11:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
; %bb.12:
	s_lshl_b64 s[0:1], s[0:1], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, vcc_lo, v16, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s1, v17, vcc_lo
; %bb.13:
	s_or_b32 exec_lo, exec_lo, s5
	s_cbranch_execnz .LBB13_3
.LBB13_14:
	v_mad_co_i64_i32 v[4:5], null, 0x5000, s4, v[9:10]
	global_load_d16_b16 v4, v[4:5], off
	s_cmp_gt_i32 s4, 1
	s_cbranch_scc1 .LBB13_4
.LBB13_15:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB13_16:
	s_or_b32 s0, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[5:6], null, 0x5000, s0, v[9:10]
.LBB13_17:
	global_load_d16_b16 v5, v[5:6], off
	s_cmp_gt_i32 s4, 0
	s_mov_b32 s1, 0
	s_cbranch_scc0 .LBB13_23
; %bb.18:
	s_add_co_i32 s0, s4, -1
                                        ; implicit-def: $vgpr14_vgpr15
	s_and_saveexec_b32 s5, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
; %bb.19:
	v_mad_co_u64_u32 v[14:15], null, 0x6000, s0, v[11:12]
; %bb.20:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
; %bb.21:
	s_lshl_b64 s[0:1], s[0:1], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v14, vcc_lo, v16, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s1, v17, vcc_lo
; %bb.22:
	s_or_b32 exec_lo, exec_lo, s5
	s_cbranch_execz .LBB13_24
	s_branch .LBB13_25
.LBB13_23:
                                        ; implicit-def: $vgpr14_vgpr15
.LBB13_24:
	s_or_b32 s0, s4, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[14:15], null, 0x5000, s0, v[9:10]
.LBB13_25:
	global_load_d16_b16 v6, v[14:15], off
	s_cmp_gt_i32 s4, -1
	s_mov_b32 s0, -1
	s_cselect_b32 s7, -1, 0
                                        ; implicit-def: $vgpr14_vgpr15
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB13_31
; %bb.26:
	s_mov_b32 s5, 0
                                        ; implicit-def: $vgpr14_vgpr15
	s_and_saveexec_b32 s0, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.27:
	v_mad_co_u64_u32 v[14:15], null, 0x6000, s4, v[11:12]
; %bb.28:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.29:
	s_lshl_b64 s[18:19], s[4:5], 13
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_co_u32 v14, vcc_lo, v16, s18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s19, v17, vcc_lo
; %bb.30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB13_31:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB13_33
; %bb.32:
	s_or_b32 s0, s4, 3
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[14:15], null, 0x5000, s0, v[9:10]
.LBB13_33:
	global_load_u16 v14, v[14:15], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 16, v4
	v_lshlrev_b32_e32 v22, 16, v2
	v_lshlrev_b32_e32 v21, 16, v3
	v_lshlrev_b32_e32 v24, 16, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v2, v21, v4, 0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v4, 16, v6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 16, v14
	v_lshlrev_b32_e32 v23, 16, v1
	v_lshlrev_b32_e32 v3, 16, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v22, v3
	v_fmac_f32_e32 v2, v23, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v24, v0
	v_mul_f32_e32 v0, 0xbfb8aa3b, v2
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v1, 0xbfb8aa3b, v2, -v0
	v_rndne_f32_e32 v3, v0
	v_dual_fmamk_f32 v1, v2, 0xb2a5705f, v1 :: v_dual_sub_f32 v0, v0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v0, v0, v1
	v_cvt_i32_f32_e32 v1, v3
	v_exp_f32_e32 v0, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v0, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0x7f800000, v0, vcc_lo
	v_add_f32_e32 v0, 1.0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v1, null, v0, v0, v2
	v_rcp_f32_e32 v3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v1, v3, 1.0
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, vcc_lo, v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, v4, v3
	v_fma_f32 v6, -v1, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v6, v3
	v_fma_f32 v1, -v1, v5, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v3, v5
	v_div_fixup_f32 v1, v1, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.34:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.35:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.36:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v3, vcc_lo, s12, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s13, v8, vcc_lo
	v_add_co_u32 v15, vcc_lo, s10, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s11, v8, vcc_lo
	v_add_co_u32 v19, vcc_lo, s8, v7
	v_cmp_lt_u32_e64 s0, 0x7ff, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, s9, v8, vcc_lo
	s_mov_b32 s5, 0
	s_and_saveexec_b32 s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB13_43
; %bb.38:
	s_and_saveexec_b32 s6, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB13_40
; %bb.39:
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s4, v[3:4]
	global_store_d16_hi_b16 v[1:2], v0, off offset:-8192
                                        ; implicit-def: $vgpr0
.LBB13_40:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB13_42
; %bb.41:
	s_lshl_b64 s[8:9], s[4:5], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v15, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s9, v18, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off offset:-4096
.LBB13_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
                                        ; implicit-def: $vgpr0
.LBB13_43:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB13_45
; %bb.44:
	s_lshl_b64 s[8:9], s[4:5], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v19, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s9, v20, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
.LBB13_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_co_u32 v5, vcc_lo, s14, v7
	v_cmp_gt_u32_e64 s1, 0x1800, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s15, v8, vcc_lo
	s_and_saveexec_b32 s6, s1
	s_cbranch_execz .LBB13_47
; %bb.46:
	s_mul_u64 s[8:9], s[4:5], 0x6000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[2:3], s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s8, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s9, v8, vcc_lo
	global_load_d16_b16 v0, v[0:1], off offset:12288
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s4, v[5:6]
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off
.LBB13_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_or_b32 s6, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s6, 3
	s_cbranch_scc1 .LBB13_53
; %bb.48:
	s_add_co_i32 s8, s4, -2
	s_mov_b32 s9, 0
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s10, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
; %bb.49:
	v_mad_co_u64_u32 v[0:1], null, 0x6000, s8, v[11:12]
; %bb.50:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
; %bb.51:
	s_lshl_b64 s[8:9], s[8:9], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, v16, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s9, v17, vcc_lo
; %bb.52:
	s_or_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB13_54
	s_branch .LBB13_55
.LBB13_53:
                                        ; implicit-def: $vgpr0_vgpr1
.LBB13_54:
	v_mad_co_i64_i32 v[0:1], null, 0x5000, s6, v[9:10]
.LBB13_55:
	global_load_d16_b16 v0, v[0:1], off
	s_cmp_lt_i32 s6, 2
	s_cbranch_scc1 .LBB13_61
; %bb.56:
	s_add_co_i32 s8, s4, -1
	s_mov_b32 s9, 0
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b32 s10, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
; %bb.57:
	v_mad_co_u64_u32 v[1:2], null, 0x6000, s8, v[11:12]
; %bb.58:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
; %bb.59:
	s_lshl_b64 s[8:9], s[8:9], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v16, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s9, v17, vcc_lo
; %bb.60:
	s_or_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB13_62
	s_branch .LBB13_63
.LBB13_61:
                                        ; implicit-def: $vgpr1_vgpr2
.LBB13_62:
	s_or_b32 s8, s4, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x5000, s8, v[9:10]
.LBB13_63:
	global_load_d16_b16 v1, v[1:2], off
	v_cndmask_b32_e64 v25, 0, 1, s7
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB13_69
; %bb.64:
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b32 s7, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
; %bb.65:
	v_mad_co_u64_u32 v[13:14], null, 0x6000, s4, v[11:12]
; %bb.66:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
; %bb.67:
	s_lshl_b64 s[8:9], s[4:5], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v13, vcc_lo, v16, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s9, v17, vcc_lo
; %bb.68:
	s_or_b32 exec_lo, exec_lo, s7
	s_cbranch_execz .LBB13_70
	s_branch .LBB13_71
.LBB13_69:
                                        ; implicit-def: $vgpr13_vgpr14
.LBB13_70:
	s_or_b32 s7, s4, 3
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[13:14], null, 0x5000, s7, v[9:10]
.LBB13_71:
	global_load_d16_b16 v2, v[13:14], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_77
; %bb.72:
	s_mov_b32 s7, 0
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b32 s8, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s8, exec_lo, s8
; %bb.73:
	v_mad_co_u64_u32 v[13:14], null, 0x6000, s6, v[11:12]
; %bb.74:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s8
; %bb.75:
	s_lshl_b64 s[10:11], s[6:7], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v13, vcc_lo, v16, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s11, v17, vcc_lo
; %bb.76:
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB13_78
	s_branch .LBB13_79
.LBB13_77:
                                        ; implicit-def: $vgpr13_vgpr14
.LBB13_78:
	s_add_co_i32 s7, s4, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[13:14], null, 0x5000, s7, v[9:10]
.LBB13_79:
	global_load_u16 v13, v[13:14], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v2, 16, v2
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v0, v21, v0, 0
	v_fmac_f32_e32 v0, v22, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v0, v23, v2 :: v_dual_lshlrev_b32 v1, 16, v13
	v_fmac_f32_e32 v0, v24, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v1, 0xbfb8aa3b, v0
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v0
	v_fma_f32 v2, 0xbfb8aa3b, v0, -v1
	v_rndne_f32_e32 v13, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v2, v0, 0xb2a5705f, v2 :: v_dual_sub_f32 v1, v1, v13
	v_add_f32_e32 v1, v1, v2
	v_cvt_i32_f32_e32 v2, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v1, v1
	v_ldexp_f32 v1, v1, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x7f800000, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 1.0, v1
	v_div_scale_f32 v2, null, v1, v1, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v13, v2
	v_fma_f32 v14, -v2, v13, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v13, v14, v13
	v_div_scale_f32 v14, vcc_lo, v0, v1, v0
	v_mul_f32_e32 v26, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v27, -v2, v26, v14
	v_fmac_f32_e32 v26, v27, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v2, v26, v14
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v13, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v2, v1, v0
	v_and_b32_e32 v0, 0x7f800000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
; %bb.80:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.81:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
; %bb.82:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_mov_b32 s7, 0
	s_and_saveexec_b32 s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s8, exec_lo, s8
	s_cbranch_execnz .LBB13_92
; %bb.84:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s8, s8
	s_cbranch_execnz .LBB13_97
.LBB13_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s1
	s_cbranch_execnz .LBB13_98
.LBB13_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_or_b32 s8, s4, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s8, 3
	s_cbranch_scc1 .LBB13_99
.LBB13_87:
	s_add_co_i32 s10, s4, -1
	s_mov_b32 s11, 0
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s9, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.88:
	v_mad_co_u64_u32 v[0:1], null, 0x6000, s10, v[11:12]
; %bb.89:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.90:
	s_lshl_b64 s[10:11], s[10:11], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, v16, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s11, v17, vcc_lo
; %bb.91:
	s_or_b32 exec_lo, exec_lo, s9
	s_cbranch_execz .LBB13_100
	s_branch .LBB13_101
.LBB13_92:
	s_and_saveexec_b32 s9, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
	s_cbranch_execz .LBB13_94
; %bb.93:
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s6, v[3:4]
	global_store_d16_hi_b16 v[1:2], v0, off offset:-8192
                                        ; implicit-def: $vgpr0
.LBB13_94:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
	s_cbranch_execz .LBB13_96
; %bb.95:
	s_lshl_b64 s[10:11], s[6:7], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v15, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s11, v18, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off offset:-4096
.LBB13_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
                                        ; implicit-def: $vgpr0
	s_and_not1_saveexec_b32 s8, s8
	s_cbranch_execz .LBB13_85
.LBB13_97:
	s_lshl_b64 s[10:11], s[6:7], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v19, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s11, v20, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s1
	s_cbranch_execz .LBB13_86
.LBB13_98:
	s_mul_u64 s[10:11], s[6:7], 0x6000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[2:3], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s10, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s11, v8, vcc_lo
	global_load_d16_b16 v0, v[0:1], off offset:12288
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s6, v[5:6]
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off
	s_or_b32 exec_lo, exec_lo, s8
	s_or_b32 s8, s4, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s8, 3
	s_cbranch_scc0 .LBB13_87
.LBB13_99:
                                        ; implicit-def: $vgpr0_vgpr1
.LBB13_100:
	v_mad_co_i64_i32 v[0:1], null, 0x5000, s8, v[9:10]
.LBB13_101:
	global_load_d16_b16 v0, v[0:1], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_107
; %bb.102:
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b32 s9, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.103:
	v_mad_co_u64_u32 v[1:2], null, 0x6000, s4, v[11:12]
; %bb.104:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.105:
	s_lshl_b64 s[10:11], s[4:5], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v16, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s11, v17, vcc_lo
; %bb.106:
	s_or_b32 exec_lo, exec_lo, s9
	s_cbranch_execz .LBB13_108
	s_branch .LBB13_109
.LBB13_107:
                                        ; implicit-def: $vgpr1_vgpr2
.LBB13_108:
	s_or_b32 s9, s4, 3
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x5000, s9, v[9:10]
.LBB13_109:
	global_load_d16_b16 v1, v[1:2], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_115
; %bb.110:
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b32 s9, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.111:
	v_mad_co_u64_u32 v[13:14], null, 0x6000, s6, v[11:12]
; %bb.112:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.113:
	s_lshl_b64 s[10:11], s[6:7], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v13, vcc_lo, v16, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s11, v17, vcc_lo
; %bb.114:
	s_or_b32 exec_lo, exec_lo, s9
	s_cbranch_execz .LBB13_116
	s_branch .LBB13_117
.LBB13_115:
                                        ; implicit-def: $vgpr13_vgpr14
.LBB13_116:
	s_add_co_i32 s9, s4, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[13:14], null, 0x5000, s9, v[9:10]
.LBB13_117:
	global_load_d16_b16 v2, v[13:14], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_123
; %bb.118:
	s_mov_b32 s9, 0
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b32 s10, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
; %bb.119:
	v_mad_co_u64_u32 v[13:14], null, 0x6000, s8, v[11:12]
; %bb.120:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
; %bb.121:
	s_lshl_b64 s[12:13], s[8:9], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v13, vcc_lo, v16, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s13, v17, vcc_lo
; %bb.122:
	s_or_b32 exec_lo, exec_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB13_124
	s_branch .LBB13_125
.LBB13_123:
                                        ; implicit-def: $vgpr13_vgpr14
.LBB13_124:
	s_add_co_i32 s9, s4, 5
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[13:14], null, 0x5000, s9, v[9:10]
.LBB13_125:
	global_load_u16 v13, v[13:14], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v2, 16, v2
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v0, v21, v0, 0
	v_fmac_f32_e32 v0, v22, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v0, v23, v2 :: v_dual_lshlrev_b32 v1, 16, v13
	v_fmac_f32_e32 v0, v24, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v1, 0xbfb8aa3b, v0
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v0
	v_fma_f32 v2, 0xbfb8aa3b, v0, -v1
	v_rndne_f32_e32 v13, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v2, v0, 0xb2a5705f, v2 :: v_dual_sub_f32 v1, v1, v13
	v_add_f32_e32 v1, v1, v2
	v_cvt_i32_f32_e32 v2, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v1, v1
	v_ldexp_f32 v1, v1, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x7f800000, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 1.0, v1
	v_div_scale_f32 v2, null, v1, v1, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v13, v2
	v_fma_f32 v14, -v2, v13, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v13, v14, v13
	v_div_scale_f32 v14, vcc_lo, v0, v1, v0
	v_mul_f32_e32 v26, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v27, -v2, v26, v14
	v_fmac_f32_e32 v26, v27, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v2, v26, v14
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v13, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v2, v1, v0
	v_and_b32_e32 v0, 0x7f800000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s9, exec_lo, s9
; %bb.126:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.127:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s9, s9
; %bb.128:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_mov_b32 s9, 0
	s_and_saveexec_b32 s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execnz .LBB13_138
; %bb.130:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execnz .LBB13_143
.LBB13_131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s1
	s_cbranch_execnz .LBB13_144
.LBB13_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_145
.LBB13_133:
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s10, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
; %bb.134:
	v_mad_co_u64_u32 v[0:1], null, 0x6000, s4, v[11:12]
; %bb.135:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
; %bb.136:
	s_lshl_b64 s[12:13], s[4:5], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, v16, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s13, v17, vcc_lo
; %bb.137:
	s_or_b32 exec_lo, exec_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_or_b32 s10, s4, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB13_146
	s_branch .LBB13_147
.LBB13_138:
	s_and_saveexec_b32 s11, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s11, exec_lo, s11
	s_cbranch_execz .LBB13_140
; %bb.139:
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s8, v[3:4]
	global_store_d16_hi_b16 v[1:2], v0, off offset:-8192
                                        ; implicit-def: $vgpr0
.LBB13_140:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s11, s11
	s_cbranch_execz .LBB13_142
; %bb.141:
	s_lshl_b64 s[12:13], s[8:9], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v15, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s13, v18, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off offset:-4096
.LBB13_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
                                        ; implicit-def: $vgpr0
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB13_131
.LBB13_143:
	s_lshl_b64 s[12:13], s[8:9], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v19, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s13, v20, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
	s_or_b32 exec_lo, exec_lo, s10
	s_and_saveexec_b32 s10, s1
	s_cbranch_execz .LBB13_132
.LBB13_144:
	s_mul_u64 s[12:13], s[8:9], 0x6000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[2:3], s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s12, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s13, v8, vcc_lo
	global_load_d16_b16 v0, v[0:1], off offset:12288
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s8, v[5:6]
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off
	s_or_b32 exec_lo, exec_lo, s10
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccz .LBB13_133
.LBB13_145:
                                        ; implicit-def: $vgpr0_vgpr1
	s_or_b32 s10, s4, 3
.LBB13_146:
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[0:1], null, 0x5000, s10, v[9:10]
.LBB13_147:
	global_load_d16_b16 v0, v[0:1], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_153
; %bb.148:
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b32 s5, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
; %bb.149:
	v_mad_co_u64_u32 v[1:2], null, 0x6000, s6, v[11:12]
; %bb.150:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
; %bb.151:
	s_lshl_b64 s[6:7], s[6:7], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v16, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s7, v17, vcc_lo
; %bb.152:
	s_or_b32 exec_lo, exec_lo, s5
	s_cbranch_execz .LBB13_154
	s_branch .LBB13_155
.LBB13_153:
                                        ; implicit-def: $vgpr1_vgpr2
.LBB13_154:
	s_add_co_i32 s5, s4, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[1:2], null, 0x5000, s5, v[9:10]
.LBB13_155:
	global_load_d16_b16 v1, v[1:2], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_161
; %bb.156:
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b32 s5, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
; %bb.157:
	v_mad_co_u64_u32 v[13:14], null, 0x6000, s8, v[11:12]
; %bb.158:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
; %bb.159:
	s_lshl_b64 s[6:7], s[8:9], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v13, vcc_lo, v16, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s7, v17, vcc_lo
; %bb.160:
	s_or_b32 exec_lo, exec_lo, s5
	s_cbranch_execz .LBB13_162
	s_branch .LBB13_163
.LBB13_161:
                                        ; implicit-def: $vgpr13_vgpr14
.LBB13_162:
	s_add_co_i32 s5, s4, 5
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[13:14], null, 0x5000, s5, v[9:10]
.LBB13_163:
	global_load_d16_b16 v2, v[13:14], off
	v_cmp_ne_u32_e32 vcc_lo, 1, v25
	s_cbranch_vccnz .LBB13_169
; %bb.164:
	s_mov_b32 s11, 0
                                        ; implicit-def: $vgpr13_vgpr14
	s_and_saveexec_b32 s5, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
; %bb.165:
	v_mad_co_u64_u32 v[13:14], null, 0x6000, s10, v[11:12]
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr17
; %bb.166:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
; %bb.167:
	s_lshl_b64 s[6:7], s[10:11], 13
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v13, vcc_lo, v16, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s7, v17, vcc_lo
; %bb.168:
	s_or_b32 exec_lo, exec_lo, s5
	s_cbranch_execz .LBB13_170
	s_branch .LBB13_171
.LBB13_169:
                                        ; implicit-def: $vgpr13_vgpr14
.LBB13_170:
	s_add_co_i32 s4, s4, 6
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[13:14], null, 0x5000, s4, v[9:10]
.LBB13_171:
	global_load_u16 v9, v[13:14], off
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v2, 16, v2
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v0, v21, v0, 0
	v_fmac_f32_e32 v0, v22, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v0, v23, v2 :: v_dual_lshlrev_b32 v1, 16, v9
	v_fmac_f32_e32 v0, v24, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v1, 0xbfb8aa3b, v0
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v0
	v_fma_f32 v2, 0xbfb8aa3b, v0, -v1
	v_rndne_f32_e32 v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmamk_f32 v2, v0, 0xb2a5705f, v2 :: v_dual_sub_f32 v1, v1, v9
	v_add_f32_e32 v1, v1, v2
	v_cvt_i32_f32_e32 v2, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v1, v1
	v_ldexp_f32 v1, v1, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x7f800000, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 1.0, v1
	v_div_scale_f32 v2, null, v1, v1, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v9, v2
	v_fma_f32 v10, -v2, v9, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v9
	v_div_scale_f32 v10, vcc_lo, v0, v1, v0
	v_mul_f32_e32 v11, v10, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v12, -v2, v11, v10
	v_fmac_f32_e32 v11, v12, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v2, v11, v10
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v2, v1, v0
	v_and_b32_e32 v0, 0x7f800000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
; %bb.172:
	v_bfe_u32 v0, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v1, v0, 0x7fff
                                        ; implicit-def: $vgpr1
; %bb.173:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
; %bb.174:
	v_and_b32_e32 v0, 0xffff, v1
	v_or_b32_e32 v2, 0x10000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v2, v1, vcc_lo
; %bb.175:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_mov_b32 s11, 0
	s_and_saveexec_b32 s4, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s4
	s_cbranch_execnz .LBB13_180
; %bb.176:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execnz .LBB13_185
.LBB13_177:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB13_179
.LBB13_178:
	s_mul_u64 s[0:1], s[10:11], 0x6000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s0, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v8, vcc_lo
	global_load_d16_b16 v0, v[0:1], off offset:12288
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s10, v[5:6]
	s_wait_loadcnt 0x0
	global_store_b16 v[1:2], v0, off
.LBB13_179:
	s_endpgm
.LBB13_180:
	s_and_saveexec_b32 s4, s16
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB13_182
; %bb.181:
	v_mad_co_u64_u32 v[1:2], null, 0x3000, s10, v[3:4]
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr18
	global_store_d16_hi_b16 v[1:2], v0, off offset:-8192
                                        ; implicit-def: $vgpr0
.LBB13_182:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB13_184
; %bb.183:
	s_lshl_b64 s[6:7], s[10:11], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v15, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s7, v18, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off offset:-4096
.LBB13_184:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr0
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB13_177
.LBB13_185:
	s_lshl_b64 s[4:5], s[10:11], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v19, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s5, v20, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz .LBB13_178
	s_branch .LBB13_179
.Lfunc_end13:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_, .Lfunc_end13-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 28
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end13-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_,"axG",@progbits,_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.num_vgpr, 28
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.numbered_sgpr, 20
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5024
; TotalNumSgprs: 22
; NumVgprs: 28
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 28
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_ad5f8cfca7299000,@object ; @__hip_cuid_ad5f8cfca7299000
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ad5f8cfca7299000
__hip_cuid_ad5f8cfca7299000:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ad5f8cfca7299000, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ad5f8cfca7299000
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
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     24
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
    .max_flat_workgroup_size: 256
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     14
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
    .max_flat_workgroup_size: 256
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     13
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
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
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
    .max_flat_workgroup_size: 1024
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     31
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
        .value_kind:     hidden_block_count_x
      - .offset:         108
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         112
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         116
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         118
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         120
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         122
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         124
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         126
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         144
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         152
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         160
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         168
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 360
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     82
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
      - .offset:         48
        .size:           8
        .value_kind:     by_value
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
    .max_flat_workgroup_size: 1024
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     19
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
        .value_kind:     hidden_block_count_x
      - .offset:         60
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         68
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         70
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         72
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         74
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         76
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         78
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         120
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 2048
    .kernarg_segment_align: 8
    .kernarg_segment_size: 312
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_.kd
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
      - .offset:         16
        .size:           8
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
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
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
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
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
    .group_segment_fixed_size: 4104
    .kernarg_segment_align: 8
    .kernarg_segment_size: 352
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
    .private_segment_fixed_size: 0
    .sgpr_count:     55
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
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
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     21
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
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     21
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
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     21
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
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     21
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
    .name:           _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     28
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx1201

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.att_syntax
	.file	"gdn_ops.hip"
	.text
	.globl	_ZN6ninfer3ops5r97003gdn23causal_conv1d_silu_bf16EPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn23causal_conv1d_silu_bf16EPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t
	.prefalign	4, .Lfunc_end0, nop
	.type	_ZN6ninfer3ops5r97003gdn23causal_conv1d_silu_bf16EPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn23causal_conv1d_silu_bf16EPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn23causal_conv1d_silu_bf16EPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$168, %rsp
	.cfi_def_cfa_offset 224
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	232(%rsp), %r10
	movl	224(%rsp), %ebx
	movq	%rsi, %xmm0
	movq	%rdi, %xmm1
	punpcklqdq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0]
	movq	%rcx, %xmm0
	movq	%rdx, %xmm2
	punpcklqdq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0]
	testq	%r8, %r8
	setne	%al
	testl	%r9d, %r9d
	setne	%r11b
	testl	%ebx, %ebx
	sete	%bpl
	testq	%r10, %r10
	setne	%r14b
	pxor	%xmm0, %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	shufps	$221, %xmm2, %xmm0              # xmm0 = xmm0[1,3],xmm2[1,3]
	shufps	$136, %xmm2, %xmm1              # xmm1 = xmm1[0,2],xmm2[0,2]
	andps	%xmm0, %xmm1
	movmskps	%xmm1, %r15d
	testl	%r15d, %r15d
	sete	%r15b
	andb	%r11b, %r14b
	andb	%al, %r14b
	andb	%r15b, %r14b
	xorb	$1, %r14b
	movl	$1, %eax
	orb	%bpl, %r14b
	jne	.LBB0_8
# %bb.1:
	cmpl	$64, %ebx
	jb	.LBB0_3
# %bb.2:
.Ltmp8:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x10
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rbx
	.cfi_adjust_cfa_offset 8
	callq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp9:                                 # EH_LABEL
	jmp	.LBB0_8
.LBB0_3:
	movq	%rdi, %r14
	movq	%rsi, %r12
	movq	%rdx, %r15
	movq	%rcx, %rbp
	movq	%r8, %r13
	movl	%r9d, 12(%rsp)                  # 4-byte Spill
	movl	%r9d, %eax
	addq	$255, %rax
	shrq	$8, %rax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	%r10, %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.4:
	testl	%eax, %eax
	jne	.LBB0_7
# %bb.5:
	movq	%r14, 104(%rsp)
	movq	%r12, 96(%rsp)
	movq	%r15, 88(%rsp)
	movq	%rbp, 80(%rsp)
	movq	%r13, 72(%rsp)
	movl	12(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 20(%rsp)
	movl	%ebx, 16(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 160(%rsp)
.Ltmp2:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp3:                                 # EH_LABEL
# %bb.6:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp4:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp5:                                 # EH_LABEL
.LBB0_7:
.Ltmp6:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp7:                                 # EH_LABEL
.LBB0_8:
	addq	$168, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_9:
	.cfi_def_cfa_offset 224
.Ltmp10:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end0:
	.size	_ZN6ninfer3ops5r97003gdn23causal_conv1d_silu_bf16EPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t, .Lfunc_end0-_ZN6ninfer3ops5r97003gdn23causal_conv1d_silu_bf16EPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Ltmp8-.Lfunc_begin0           # >> Call Site 1 <<
	.uleb128 .Ltmp7-.Ltmp8                  #   Call between .Ltmp8 and .Ltmp7
	.uleb128 .Ltmp10-.Lfunc_begin0          #     jumps to .Ltmp10
	.byte	1                               #   On action: 1
.Lcst_end0:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase0:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end1, nop     # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$168, %rsp
	.cfi_def_cfa_offset 224
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r9d, %r14d
	movq	%r8, %rbx
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
	movq	%rdx, %r13
	movq	%rsi, %r12
	movq	%rdi, %r15
	movq	232(%rsp), %r9
	movl	224(%rsp), %ebp
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leal	255(%r14), %edi
	shrl	$8, %edi
	leal	3(%rbp), %eax
	shrl	$2, %eax
	shlq	$32, %rax
	orq	%rax, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	je	.LBB1_1
# %bb.2:
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB1_8
	jmp	.LBB1_3
.LBB1_1:
	movq	%r15, 88(%rsp)
	movq	%r12, 80(%rsp)
	movq	%r13, 72(%rsp)
	movq	%rbx, 32(%rsp)
	movl	%r14d, 12(%rsp)
	movl	%ebp, 100(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	32(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	100(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB1_8
.LBB1_3:
	cmpl	$3, %ebp
	jb	.LBB1_5
# %bb.4:
	addl	$-3, %ebp
	movl	%r14d, %eax
	imulq	%rax, %rbp
	leaq	(%r15,%rbp,2), %rsi
	addq	%rax, %rax
	leaq	(%rax,%rax,2), %rdx
	movq	104(%rsp), %rdi                 # 8-byte Reload
	movl	$3, %ecx
	movq	232(%rsp), %r8
	callq	hipMemcpyAsync@PLT
	jmp	.LBB1_8
.LBB1_5:
	movl	%r14d, %eax
	addq	$255, %rax
	shrq	$8, %rax
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leaq	(%rdx,%rax), %rdi
	addq	$-256, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	232(%rsp), %r9
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB1_7
# %bb.6:
	movq	%r15, 88(%rsp)
	movq	%r13, 80(%rsp)
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	movl	%r14d, 16(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB1_7:
	callq	hipGetLastError@PLT
.LBB1_8:
	addq	$168, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t, .Lfunc_end1-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t
	.cfi_endproc
                                        # -- End function
	.section	.text.__clang_call_terminate,"axG",@progbits,__clang_call_terminate,comdat
	.hidden	__clang_call_terminate          # -- Begin function __clang_call_terminate
	.weak	__clang_call_terminate
	.prefalign	4, .Lfunc_end2, nop
	.type	__clang_call_terminate,@function
__clang_call_terminate:                 # @__clang_call_terminate
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__cxa_begin_catch@PLT
	callq	_ZSt9terminatev@PLT
.Lfunc_end2:
	.size	__clang_call_terminate, .Lfunc_end2-__clang_call_terminate
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end3, nop     # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end3:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj, .Lfunc_end3-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.cfi_endproc
                                        # -- End function
	.globl	_ZN6ninfer3ops5r97003gdn42causal_conv1d_silu_incumbent_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn42causal_conv1d_silu_incumbent_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t
	.prefalign	4, .Lfunc_end4, nop
	.type	_ZN6ninfer3ops5r97003gdn42causal_conv1d_silu_incumbent_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn42causal_conv1d_silu_incumbent_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn42causal_conv1d_silu_incumbent_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception1
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$152, %rsp
	.cfi_def_cfa_offset 208
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r9d, %r10d
	movq	%rcx, %r14
	movq	%rdx, %r15
	movq	%rsi, %r12
	movq	%rdi, %r13
	movq	216(%rsp), %r9
	movq	%rsi, %xmm0
	movq	%rdi, %xmm1
	punpcklqdq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0]
	movq	%rcx, %xmm0
	movq	%rdx, %xmm2
	punpcklqdq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0]
	testq	%r8, %r8
	setne	%al
	testl	%r10d, %r10d
	setne	%cl
	cmpl	$0, 208(%rsp)
	sete	%dl
	testq	%r9, %r9
	setne	%sil
	pxor	%xmm0, %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	shufps	$221, %xmm2, %xmm0              # xmm0 = xmm0[1,3],xmm2[1,3]
	shufps	$136, %xmm2, %xmm1              # xmm1 = xmm1[0,2],xmm2[0,2]
	andps	%xmm0, %xmm1
	movmskps	%xmm1, %edi
	testl	%edi, %edi
	sete	%dil
	andb	%cl, %sil
	andb	%al, %sil
	andb	%dil, %sil
	xorb	$1, %sil
	movl	$1, %eax
	orb	%dl, %sil
	jne	.LBB4_6
# %bb.1:
	movq	%r8, %rbx
	movl	%r10d, %ebp
	movl	%r10d, %eax
	addq	$255, %rax
	shrq	$8, %rax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp11:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
.Ltmp12:                                # EH_LABEL
# %bb.2:
	testl	%eax, %eax
	jne	.LBB4_5
# %bb.3:
	movq	%r13, 88(%rsp)
	movq	%r12, 80(%rsp)
	movq	%r15, 72(%rsp)
	movq	%r14, 64(%rsp)
	movq	%rbx, 56(%rsp)
	movl	%ebp, 4(%rsp)
	movl	208(%rsp), %eax
	movl	%eax, (%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	movq	%rsp, %rax
	movq	%rax, 144(%rsp)
.Ltmp13:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp14:                                # EH_LABEL
# %bb.4:
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
.Ltmp15:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp16:                                # EH_LABEL
.LBB4_5:
.Ltmp17:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp18:                                # EH_LABEL
.LBB4_6:
	addq	$152, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB4_7:
	.cfi_def_cfa_offset 208
.Ltmp19:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end4:
	.size	_ZN6ninfer3ops5r97003gdn42causal_conv1d_silu_incumbent_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t, .Lfunc_end4-_ZN6ninfer3ops5r97003gdn42causal_conv1d_silu_incumbent_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table4:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase1-.Lttbaseref1
.Lttbaseref1:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Ltmp11-.Lfunc_begin1          # >> Call Site 1 <<
	.uleb128 .Ltmp18-.Ltmp11                #   Call between .Ltmp11 and .Ltmp18
	.uleb128 .Ltmp19-.Lfunc_begin1          #     jumps to .Ltmp19
	.byte	1                               #   On action: 1
.Lcst_end1:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase1:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_ZN6ninfer3ops5r97003gdn40causal_conv1d_silu_prefill_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjjP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn40causal_conv1d_silu_prefill_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjjP12ihipStream_t
	.prefalign	4, .Lfunc_end5, nop
	.type	_ZN6ninfer3ops5r97003gdn40causal_conv1d_silu_prefill_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjjP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn40causal_conv1d_silu_prefill_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjjP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn40causal_conv1d_silu_prefill_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjjP12ihipStream_t
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception2
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$168, %rsp
	.cfi_def_cfa_offset 224
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r9d, %ebx
	movq	%rdx, %r15
	movq	%rdi, %r14
	movl	232(%rsp), %edx
	leal	-1(%rdx), %edi
	movl	%edx, %r9d
	xorl	%edi, %r9d
	cmpl	%edi, %r9d
	jbe	.LBB5_2
# %bb.1:
	rep		bsfl	%edx, %eax
	addl	$-2, %eax
	movb	$1, %r10b
	cmpl	$4, %eax
	jb	.LBB5_3
.LBB5_2:
	xorl	%r10d, %r10d
.LBB5_3:
	movq	%rsi, %xmm0
	movq	%r14, %xmm1
	punpcklqdq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0]
	movq	%rcx, %xmm0
	movq	%r15, %xmm2
	punpcklqdq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0]
	pxor	%xmm0, %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	shufps	$221, %xmm2, %xmm0              # xmm0 = xmm0[1,3],xmm2[1,3]
	shufps	$136, %xmm2, %xmm1              # xmm1 = xmm1[0,2],xmm2[0,2]
	andps	%xmm0, %xmm1
	movmskps	%xmm1, %r11d
	movl	$1, %eax
	testl	%r11d, %r11d
	jne	.LBB5_48
# %bb.4:
	testq	%r8, %r8
	je	.LBB5_48
# %bb.5:
	movq	240(%rsp), %r11
	testq	%r11, %r11
	je	.LBB5_48
# %bb.6:
	testl	%ebx, %ebx
	je	.LBB5_48
# %bb.7:
	movl	224(%rsp), %ebp
	testl	%ebp, %ebp
	je	.LBB5_48
# %bb.8:
	testb	%r10b, %r10b
	je	.LBB5_48
# %bb.9:
	cmpl	%edi, %r9d
	jbe	.LBB5_48
# %bb.10:
	rep		bsfl	%edx, %edx
	addl	$-2, %edx
	cmpl	$3, %edx
	ja	.LBB5_48
# %bb.11:
	leaq	.LJTI5_0(%rip), %rax
	movslq	(%rax,%rdx,4), %rdx
	addq	%rax, %rdx
	jmpq	*%rdx
.LBB5_12:
.Ltmp66:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	%r14, %rdi
	movq	%r15, %rdx
	movl	%ebx, %r9d
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133launch_causal_conv1d_silu_prefillILj4EEE10hipError_tPK12hip_bfloat16S8_S8_PS6_S9_jjP12ihipStream_t
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp67:                                # EH_LABEL
	jmp	.LBB5_48
.LBB5_24:
	movq	%rsi, %r12
	movq	%r8, %r13
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
	leal	255(%rbx), %edi
	shrl	$8, %edi
	leal	15(%rbp), %eax
	shrl	$4, %eax
	shlq	$32, %rax
	orq	%rax, %rdi
.Ltmp34:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	%r11, %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp35:                                # EH_LABEL
# %bb.25:
	testl	%eax, %eax
	jne	.LBB5_28
# %bb.26:
	movq	%r14, 88(%rsp)
	movq	%r12, 80(%rsp)
	movq	%r15, 72(%rsp)
	movq	%r13, 32(%rsp)
	movl	%ebx, 12(%rsp)
	movl	%ebp, 100(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	32(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	100(%rsp), %rax
	movq	%rax, 152(%rsp)
.Ltmp36:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp37:                                # EH_LABEL
# %bb.27:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp38:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp39:                                # EH_LABEL
.LBB5_28:
.Ltmp40:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp41:                                # EH_LABEL
# %bb.29:
	testl	%eax, %eax
	jne	.LBB5_48
# %bb.30:
	cmpl	$2, %ebp
	ja	.LBB5_42
# %bb.31:
	movl	%ebx, %eax
	addq	$255, %rax
	shrq	$8, %rax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp42:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	240(%rsp), %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp43:                                # EH_LABEL
# %bb.32:
	testl	%eax, %eax
	jne	.LBB5_47
# %bb.33:
	movq	%r14, 88(%rsp)
	movq	%r15, 80(%rsp)
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	movl	%ebx, 16(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
.Ltmp44:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp45:                                # EH_LABEL
# %bb.34:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp46:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp47:                                # EH_LABEL
	jmp	.LBB5_47
.LBB5_35:
	movq	%rsi, %r12
	movq	%r8, %r13
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
	leal	255(%rbx), %edi
	shrl	$8, %edi
	leal	31(%rbp), %eax
	shrl	$5, %eax
	shlq	$32, %rax
	orq	%rax, %rdi
.Ltmp20:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	%r11, %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp21:                                # EH_LABEL
# %bb.36:
	testl	%eax, %eax
	jne	.LBB5_39
# %bb.37:
	movq	%r14, 88(%rsp)
	movq	%r12, 80(%rsp)
	movq	%r15, 72(%rsp)
	movq	%r13, 32(%rsp)
	movl	%ebx, 12(%rsp)
	movl	%ebp, 100(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	32(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	100(%rsp), %rax
	movq	%rax, 152(%rsp)
.Ltmp22:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp23:                                # EH_LABEL
# %bb.38:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp24:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp25:                                # EH_LABEL
.LBB5_39:
.Ltmp26:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp27:                                # EH_LABEL
# %bb.40:
	testl	%eax, %eax
	jne	.LBB5_48
# %bb.41:
	cmpl	$3, %ebp
	jae	.LBB5_42
# %bb.43:
	movl	%ebx, %eax
	addq	$255, %rax
	shrq	$8, %rax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp28:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	240(%rsp), %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp29:                                # EH_LABEL
# %bb.44:
	testl	%eax, %eax
	jne	.LBB5_47
# %bb.45:
	movq	%r14, 88(%rsp)
	movq	%r15, 80(%rsp)
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	movl	%ebx, 16(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
.Ltmp30:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp31:                                # EH_LABEL
# %bb.46:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp32:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp33:                                # EH_LABEL
	jmp	.LBB5_47
.LBB5_13:
	movq	%rsi, %r12
	movq	%r8, %r13
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
	leal	255(%rbx), %edi
	shrl	$8, %edi
	leal	7(%rbp), %eax
	shrl	$3, %eax
	shlq	$32, %rax
	orq	%rax, %rdi
.Ltmp48:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	%r11, %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp49:                                # EH_LABEL
# %bb.14:
	testl	%eax, %eax
	jne	.LBB5_17
# %bb.15:
	movq	%r14, 88(%rsp)
	movq	%r12, 80(%rsp)
	movq	%r15, 72(%rsp)
	movq	%r13, 32(%rsp)
	movl	%ebx, 12(%rsp)
	movl	%ebp, 100(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	32(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	100(%rsp), %rax
	movq	%rax, 152(%rsp)
.Ltmp50:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp51:                                # EH_LABEL
# %bb.16:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp52:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp53:                                # EH_LABEL
.LBB5_17:
.Ltmp54:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp55:                                # EH_LABEL
# %bb.18:
	testl	%eax, %eax
	jne	.LBB5_48
# %bb.19:
	cmpl	$2, %ebp
	jbe	.LBB5_20
.LBB5_42:
	addl	$-3, %ebp
	movl	%ebx, %eax
	imulq	%rax, %rbp
	leaq	(%r14,%rbp,2), %rsi
	addq	%rax, %rax
	leaq	(%rax,%rax,2), %rdx
.Ltmp64:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	104(%rsp), %rdi                 # 8-byte Reload
	movl	$3, %ecx
	movq	240(%rsp), %r8
	callq	hipMemcpyAsync@PLT
.Ltmp65:                                # EH_LABEL
	jmp	.LBB5_48
.LBB5_20:
	movl	%ebx, %eax
	addq	$255, %rax
	shrq	$8, %rax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp56:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	240(%rsp), %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp57:                                # EH_LABEL
# %bb.21:
	testl	%eax, %eax
	jne	.LBB5_47
# %bb.22:
	movq	%r14, 88(%rsp)
	movq	%r15, 80(%rsp)
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	movl	%ebx, 16(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 144(%rsp)
.Ltmp58:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp59:                                # EH_LABEL
# %bb.23:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp60:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp61:                                # EH_LABEL
.LBB5_47:
.Ltmp62:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp63:                                # EH_LABEL
.LBB5_48:
	addq	$168, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB5_49:
	.cfi_def_cfa_offset 224
.Ltmp68:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end5:
	.size	_ZN6ninfer3ops5r97003gdn40causal_conv1d_silu_prefill_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjjP12ihipStream_t, .Lfunc_end5-_ZN6ninfer3ops5r97003gdn40causal_conv1d_silu_prefill_qualificationEPK12hip_bfloat16S5_S5_PS3_S6_jjjP12ihipStream_t
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.LJTI5_0:
	.long	.LBB5_12-.LJTI5_0
	.long	.LBB5_13-.LJTI5_0
	.long	.LBB5_24-.LJTI5_0
	.long	.LBB5_35-.LJTI5_0
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table5:
.Lexception2:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase2-.Lttbaseref2
.Lttbaseref2:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.uleb128 .Ltmp66-.Lfunc_begin2          # >> Call Site 1 <<
	.uleb128 .Ltmp63-.Ltmp66                #   Call between .Ltmp66 and .Ltmp63
	.uleb128 .Ltmp68-.Lfunc_begin2          #     jumps to .Ltmp68
	.byte	1                               #   On action: 1
.Lcst_end2:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase2:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_ZN6ninfer3ops5r97003gdn49projection_conv_prefill_p2048_direct_scatter_bf16EPK12hip_bfloat16S5_S5_S5_PS3_S6_S6_S6_S6_jP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn49projection_conv_prefill_p2048_direct_scatter_bf16EPK12hip_bfloat16S5_S5_S5_PS3_S6_S6_S6_S6_jP12ihipStream_t
	.prefalign	4, .Lfunc_end6, nop
	.type	_ZN6ninfer3ops5r97003gdn49projection_conv_prefill_p2048_direct_scatter_bf16EPK12hip_bfloat16S5_S5_S5_PS3_S6_S6_S6_S6_jP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn49projection_conv_prefill_p2048_direct_scatter_bf16EPK12hip_bfloat16S5_S5_S5_PS3_S6_S6_S6_S6_jP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn49projection_conv_prefill_p2048_direct_scatter_bf16EPK12hip_bfloat16S5_S5_S5_PS3_S6_S6_S6_S6_jP12ihipStream_t
.Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception3
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$344, %rsp                      # imm = 0x158
	.cfi_def_cfa_offset 400
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, %rbp
	movq	%r9, %xmm0
	movq	%r8, %xmm1
	punpcklqdq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0]
	movq	%rsi, %xmm0
	movq	%rdi, %xmm2
	punpcklqdq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0]
	movq	%rcx, %xmm0
	movq	%rdx, %xmm3
	punpcklqdq	%xmm0, %xmm3            # xmm3 = xmm3[0],xmm0[0]
	pxor	%xmm0, %xmm0
	pcmpeqd	%xmm0, %xmm3
	pshufd	$177, %xmm3, %xmm4              # xmm4 = xmm3[1,0,3,2]
	pand	%xmm3, %xmm4
	pcmpeqd	%xmm0, %xmm2
	pshufd	$177, %xmm2, %xmm3              # xmm3 = xmm2[1,0,3,2]
	pand	%xmm2, %xmm3
	packssdw	%xmm4, %xmm3
	pcmpeqd	%xmm0, %xmm1
	pshufd	$177, %xmm1, %xmm2              # xmm2 = xmm1[1,0,3,2]
	pcmpeqd	400(%rsp), %xmm0
	pand	%xmm1, %xmm2
	pshufd	$177, %xmm0, %xmm1              # xmm1 = xmm0[1,0,3,2]
	pand	%xmm0, %xmm1
	packssdw	%xmm1, %xmm2
	packssdw	%xmm2, %xmm3
	pmovmskb	%xmm3, %ecx
	movl	$1, %eax
	testl	$43690, %ecx                    # imm = 0xAAAA
	jne	.LBB6_32
# %bb.1:
	cmpq	$0, 416(%rsp)
	je	.LBB6_32
# %bb.2:
	cmpq	$0, 432(%rsp)
	je	.LBB6_32
# %bb.3:
	cmpl	$2048, 424(%rsp)                # imm = 0x800
	jne	.LBB6_32
# %bb.4:
	movq	%r9, %r13
	movq	%r8, %rbx
	movq	%rdx, %r12
	movq	%rdi, %r15
	movq	408(%rsp), %rcx
	movq	400(%rsp), %rdx
	leaq	16777216(%rdi), %rax
	movq	%rdi, 200(%rsp)
	movq	%rax, 208(%rsp)
	leaq	50331648(%rsi), %rax
	movq	%rsi, 72(%rsp)                  # 8-byte Spill
	movq	%rsi, 216(%rsp)
	leaq	224(%rsp), %rsi
	movq	%rax, 224(%rsp)
	leaq	81920(%r12), %rax
	movq	%r12, 232(%rsp)
	movq	%rax, 240(%rsp)
	leaq	61440(%rbp), %rax
	movq	%rbp, 248(%rsp)
	movq	%rax, 256(%rsp)
	leaq	61440(%r8), %rax
	movq	%r8, 264(%rsp)
	movq	%rax, 272(%rsp)
	leaq	8388608(%r9), %rax
	movq	%r9, 280(%rsp)
	movq	%rax, 288(%rsp)
	leaq	8388608(%rdx), %rax
	movq	%rdx, 296(%rsp)
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	%rax, 304(%rsp)
	leaq	25165824(%rcx), %rax
	movq	%rcx, 312(%rsp)
	movq	%rax, %r8
	movq	%rax, 320(%rsp)
	movq	416(%rsp), %rax
	leaq	25165824(%rax), %rcx
	movq	%rax, 328(%rsp)
	movq	%rcx, %r14
	movq	%rcx, 336(%rsp)
	movl	$8, %edi
	xorl	%r10d, %r10d
	movq	%rbp, %r11
	xorq	%rbx, %r11
	jmp	.LBB6_7
	.p2align	4
.LBB6_6:                                #   in Loop: Header=BB6_7 Depth=1
	incq	%r10
	decq	%rdi
	addq	$16, %rsi
	cmpq	$9, %r10
	je	.LBB6_21
.LBB6_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB6_10 Depth 2
	cmpq	$7, %r10
	ja	.LBB6_6
# %bb.8:                                #   in Loop: Header=BB6_7 Depth=1
	movq	%r10, %rax
	xorq	$3, %rax
	movq	%r10, %rcx
	shlq	$4, %rcx
	orq	%r11, %rax
	leaq	200(%rsp,%rcx), %rdx
	jne	.LBB6_9
# %bb.13:                               #   in Loop: Header=BB6_7 Depth=1
	movl	$1, %eax
	leaq	8388608(%r13), %rcx
	cmpq	%rcx, (%rdx)
	jae	.LBB6_15
# %bb.14:                               #   in Loop: Header=BB6_7 Depth=1
	cmpq	%r13, 8(%rdx)
	ja	.LBB6_32
.LBB6_15:                               #   in Loop: Header=BB6_7 Depth=1
	movq	80(%rsp), %rcx                  # 8-byte Reload
	cmpq	%rcx, (%rdx)
	jae	.LBB6_17
# %bb.16:                               #   in Loop: Header=BB6_7 Depth=1
	movq	400(%rsp), %rcx
	cmpq	%rcx, 8(%rdx)
	ja	.LBB6_32
.LBB6_17:                               #   in Loop: Header=BB6_7 Depth=1
	cmpq	%r8, (%rdx)
	jae	.LBB6_19
# %bb.18:                               #   in Loop: Header=BB6_7 Depth=1
	movq	408(%rsp), %rcx
	cmpq	%rcx, 8(%rdx)
	ja	.LBB6_32
.LBB6_19:                               #   in Loop: Header=BB6_7 Depth=1
	cmpq	%r14, (%rdx)
	jae	.LBB6_6
# %bb.20:                               #   in Loop: Header=BB6_7 Depth=1
	movq	416(%rsp), %rcx
	cmpq	%rcx, 8(%rdx)
	jbe	.LBB6_6
	jmp	.LBB6_32
	.p2align	4
.LBB6_9:                                #   in Loop: Header=BB6_7 Depth=1
	movq	(%rdx), %rax
	movq	8(%rdx), %rdx
	movq	%rsi, %rcx
	movq	%rdi, %r9
	jmp	.LBB6_10
	.p2align	4
.LBB6_5:                                #   in Loop: Header=BB6_10 Depth=2
	addq	$16, %rcx
	decq	%r9
	je	.LBB6_6
.LBB6_10:                               #   Parent Loop BB6_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	cmpq	(%rcx), %rax
	jae	.LBB6_5
# %bb.11:                               #   in Loop: Header=BB6_10 Depth=2
	cmpq	%rdx, -8(%rcx)
	jae	.LBB6_5
# %bb.12:
	movl	$1, %eax
.LBB6_32:
	addq	$344, %rsp                      # imm = 0x158
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB6_21:
	.cfi_def_cfa_offset 400
.Ltmp69:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$2199023255592, %rdi            # imm = 0x20000000028
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	432(%rsp), %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp70:                                # EH_LABEL
	movq	72(%rsp), %r14                  # 8-byte Reload
# %bb.22:
	testl	%eax, %eax
	jne	.LBB6_25
# %bb.23:
	movq	%r15, 64(%rsp)
	movq	%r14, 56(%rsp)
	movq	%r12, 48(%rsp)
	movq	%rbp, 8(%rsp)
	movq	%r13, (%rsp)
	movq	400(%rsp), %rax
	movq	%rax, 120(%rsp)
	movq	408(%rsp), %rax
	movq	%rax, 112(%rsp)
	movq	416(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 152(%rsp)
	movq	%rsp, %rax
	movq	%rax, 160(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp71:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	96(%rsp), %rdx
	leaq	88(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp72:                                # EH_LABEL
# %bb.24:
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
.Ltmp73:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	88(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	104(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp74:                                # EH_LABEL
.LBB6_25:
.Ltmp75:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp76:                                # EH_LABEL
# %bb.26:
	testl	%eax, %eax
	jne	.LBB6_32
# %bb.27:
.Ltmp77:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967336, %rdi               # imm = 0x100000028
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	432(%rsp), %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp78:                                # EH_LABEL
# %bb.28:
	testl	%eax, %eax
	jne	.LBB6_31
# %bb.29:
	movq	%r15, 64(%rsp)
	movq	%r14, 56(%rsp)
	movq	%rbx, 48(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 144(%rsp)
.Ltmp79:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp80:                                # EH_LABEL
# %bb.30:
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
.Ltmp81:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp82:                                # EH_LABEL
.LBB6_31:
.Ltmp83:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp84:                                # EH_LABEL
	jmp	.LBB6_32
.LBB6_33:
.Ltmp85:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end6:
	.size	_ZN6ninfer3ops5r97003gdn49projection_conv_prefill_p2048_direct_scatter_bf16EPK12hip_bfloat16S5_S5_S5_PS3_S6_S6_S6_S6_jP12ihipStream_t, .Lfunc_end6-_ZN6ninfer3ops5r97003gdn49projection_conv_prefill_p2048_direct_scatter_bf16EPK12hip_bfloat16S5_S5_S5_PS3_S6_S6_S6_S6_jP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table6:
.Lexception3:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase3-.Lttbaseref3
.Lttbaseref3:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end3-.Lcst_begin3
.Lcst_begin3:
	.uleb128 .Ltmp69-.Lfunc_begin3          # >> Call Site 1 <<
	.uleb128 .Ltmp84-.Ltmp69                #   Call between .Ltmp69 and .Ltmp84
	.uleb128 .Ltmp85-.Lfunc_begin3          #     jumps to .Ltmp85
	.byte	1                               #   On action: 1
.Lcst_end3:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase3:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end7, nop     # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%r9, 48(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end7:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_, .Lfunc_end7-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end8, nop     # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end8:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_, .Lfunc_end8-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.cfi_endproc
                                        # -- End function
	.globl	_ZN6ninfer3ops5r97003gdn29projection_conv_snapshot_bf16EPK12hip_bfloat16S5_S5_PS3_PKiS8_S8_S6_S6_S6_S6_jjjP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn29projection_conv_snapshot_bf16EPK12hip_bfloat16S5_S5_PS3_PKiS8_S8_S6_S6_S6_S6_jjjP12ihipStream_t
	.prefalign	4, .Lfunc_end9, nop
	.type	_ZN6ninfer3ops5r97003gdn29projection_conv_snapshot_bf16EPK12hip_bfloat16S5_S5_PS3_PKiS8_S8_S6_S6_S6_S6_jjjP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn29projection_conv_snapshot_bf16EPK12hip_bfloat16S5_S5_PS3_PKiS8_S8_S6_S6_S6_S6_jjjP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn29projection_conv_snapshot_bf16EPK12hip_bfloat16S5_S5_PS3_PKiS8_S8_S6_S6_S6_S6_jjjP12ihipStream_t
.Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception4
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$248, %rsp
	.cfi_def_cfa_offset 304
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, %rbx
	movq	%rcx, %r15
	movq	%rdx, %r12
	movq	%rsi, %r13
	movq	%rdi, %rbp
	movl	352(%rsp), %edi
	movl	344(%rsp), %edx
	testl	%edx, %edx
	sete	%al
	leal	-1(%rdi), %ecx
	cmpl	$4, %ecx
	setae	%cl
	orb	%al, %cl
	jne	.LBB9_1
# %bb.2:
	cmpl	$1, %edi
	sete	%al
	cmpl	$17, %edx
	setb	%dl
	orb	%al, %dl
	cmpl	$0, 360(%rsp)
	setne	%cl
	andb	%dl, %cl
	jmp	.LBB9_3
.LBB9_1:
	xorl	%ecx, %ecx
.LBB9_3:
	movq	%r13, %xmm1
	movq	%rbp, %xmm0
	punpcklqdq	%xmm1, %xmm0            # xmm0 = xmm0[0],xmm1[0]
	movq	%r15, %xmm1
	movq	%r12, %xmm2
	punpcklqdq	%xmm1, %xmm2            # xmm2 = xmm2[0],xmm1[0]
	movq	304(%rsp), %xmm1                # xmm1 = mem[0],zero
	movq	%rbx, %xmm3
	punpcklqdq	%xmm1, %xmm3            # xmm3 = xmm3[0],xmm1[0]
	movq	320(%rsp), %xmm1                # xmm1 = mem[0],zero
	movq	312(%rsp), %xmm4                # xmm4 = mem[0],zero
	punpcklqdq	%xmm1, %xmm4            # xmm4 = xmm4[0],xmm1[0]
	pxor	%xmm1, %xmm1
	pcmpeqd	%xmm1, %xmm4
	pshufd	$177, %xmm4, %xmm5              # xmm5 = xmm4[1,0,3,2]
	pand	%xmm4, %xmm5
	pcmpeqd	%xmm1, %xmm3
	pshufd	$177, %xmm3, %xmm4              # xmm4 = xmm3[1,0,3,2]
	pand	%xmm3, %xmm4
	packssdw	%xmm5, %xmm4
	pcmpeqd	%xmm1, %xmm2
	pshufd	$177, %xmm2, %xmm3              # xmm3 = xmm2[1,0,3,2]
	pand	%xmm2, %xmm3
	pcmpeqd	%xmm1, %xmm0
	pshufd	$177, %xmm0, %xmm1              # xmm1 = xmm0[1,0,3,2]
	pand	%xmm0, %xmm1
	packssdw	%xmm3, %xmm1
	packssdw	%xmm4, %xmm1
	pmovmskb	%xmm1, %edx
	movl	$1, %eax
	testl	$43690, %edx                    # imm = 0xAAAA
	jne	.LBB9_13
# %bb.4:
	cmpq	$0, 328(%rsp)
	je	.LBB9_13
# %bb.5:
	cmpq	$0, 336(%rsp)
	je	.LBB9_13
# %bb.6:
	movq	368(%rsp), %r9
	testq	%r9, %r9
	je	.LBB9_13
# %bb.7:
	testb	%cl, %cl
	je	.LBB9_13
# %bb.8:
	movq	%r8, %r14
	shlq	$32, %rdi
	orq	$40, %rdi
.Ltmp86:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
.Ltmp87:                                # EH_LABEL
# %bb.9:
	testl	%eax, %eax
	movl	360(%rsp), %eax
	jne	.LBB9_12
# %bb.10:
	movq	%rbp, 136(%rsp)
	movq	%r13, 128(%rsp)
	movq	%r12, 120(%rsp)
	movq	%r15, 112(%rsp)
	movq	%r14, 104(%rsp)
	movq	%rbx, 96(%rsp)
	movq	304(%rsp), %rcx
	movq	%rcx, 88(%rsp)
	movq	312(%rsp), %rcx
	movq	%rcx, 80(%rsp)
	movq	320(%rsp), %rcx
	movq	%rcx, 72(%rsp)
	movq	328(%rsp), %rcx
	movq	%rcx, 64(%rsp)
	movq	336(%rsp), %rcx
	movq	%rcx, 56(%rsp)
	movl	344(%rsp), %ecx
	movl	%ecx, 4(%rsp)
	movl	%eax, (%rsp)
	leaq	136(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 232(%rsp)
	movq	%rsp, %rax
	movq	%rax, 240(%rsp)
.Ltmp88:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp89:                                # EH_LABEL
# %bb.11:
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
.Ltmp90:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj(%rip), %rdi
	leaq	144(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp91:                                # EH_LABEL
.LBB9_12:
.Ltmp92:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp93:                                # EH_LABEL
.LBB9_13:
	addq	$248, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB9_14:
	.cfi_def_cfa_offset 304
.Ltmp94:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end9:
	.size	_ZN6ninfer3ops5r97003gdn29projection_conv_snapshot_bf16EPK12hip_bfloat16S5_S5_PS3_PKiS8_S8_S6_S6_S6_S6_jjjP12ihipStream_t, .Lfunc_end9-_ZN6ninfer3ops5r97003gdn29projection_conv_snapshot_bf16EPK12hip_bfloat16S5_S5_PS3_PKiS8_S8_S6_S6_S6_S6_jjjP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table9:
.Lexception4:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase4-.Lttbaseref4
.Lttbaseref4:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end4-.Lcst_begin4
.Lcst_begin4:
	.uleb128 .Ltmp86-.Lfunc_begin4          # >> Call Site 1 <<
	.uleb128 .Ltmp93-.Ltmp86                #   Call between .Ltmp86 and .Ltmp93
	.uleb128 .Ltmp94-.Lfunc_begin4          #     jumps to .Ltmp94
	.byte	1                               #   On action: 1
.Lcst_end4:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase4:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_ZN6ninfer3ops5r97003gdn27projection_conv_record_bf16EPK12hip_bfloat16S5_S5_S5_PKiS7_S7_PS3_S8_S8_S8_S8_jjjP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn27projection_conv_record_bf16EPK12hip_bfloat16S5_S5_S5_PKiS7_S7_PS3_S8_S8_S8_S8_jjjP12ihipStream_t
	.prefalign	4, .Lfunc_end10, nop
	.type	_ZN6ninfer3ops5r97003gdn27projection_conv_record_bf16EPK12hip_bfloat16S5_S5_S5_PKiS7_S7_PS3_S8_S8_S8_S8_jjjP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn27projection_conv_record_bf16EPK12hip_bfloat16S5_S5_S5_PKiS7_S7_PS3_S8_S8_S8_S8_jjjP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn27projection_conv_record_bf16EPK12hip_bfloat16S5_S5_S5_PKiS7_S7_PS3_S8_S8_S8_S8_jjjP12ihipStream_t
.Lfunc_begin5:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception5
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 336
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, %r15
	movq	344(%rsp), %xmm0                # xmm0 = mem[0],zero
	movq	%r9, %xmm1
	punpcklqdq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0]
	movq	%rsi, %xmm0
	movq	%rdi, %xmm2
	punpcklqdq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0]
	movq	%rcx, %xmm0
	movq	%rdx, %xmm3
	punpcklqdq	%xmm0, %xmm3            # xmm3 = xmm3[0],xmm0[0]
	pxor	%xmm0, %xmm0
	pcmpeqd	%xmm0, %xmm3
	pshufd	$177, %xmm3, %xmm4              # xmm4 = xmm3[1,0,3,2]
	pand	%xmm3, %xmm4
	pcmpeqd	%xmm0, %xmm2
	pshufd	$177, %xmm2, %xmm3              # xmm3 = xmm2[1,0,3,2]
	pand	%xmm2, %xmm3
	packssdw	%xmm4, %xmm3
	pcmpeqd	%xmm0, %xmm1
	pshufd	$177, %xmm1, %xmm2              # xmm2 = xmm1[1,0,3,2]
	pcmpeqd	352(%rsp), %xmm0
	pand	%xmm1, %xmm2
	pshufd	$177, %xmm0, %xmm1              # xmm1 = xmm0[1,0,3,2]
	pand	%xmm0, %xmm1
	packssdw	%xmm1, %xmm2
	packssdw	%xmm2, %xmm3
	pmovmskb	%xmm3, %ecx
	movl	$1, %eax
	testl	$43690, %ecx                    # imm = 0xAAAA
	jne	.LBB10_12
# %bb.1:
	cmpq	$0, 368(%rsp)
	je	.LBB10_12
# %bb.2:
	cmpq	$0, 376(%rsp)
	je	.LBB10_12
# %bb.3:
	movq	%r9, %rbx
	movq	408(%rsp), %r9
	testq	%r9, %r9
	je	.LBB10_12
# %bb.4:
	movl	384(%rsp), %ecx
	addl	$-2, %ecx
	cmpl	$14, %ecx
	ja	.LBB10_12
# %bb.5:
	movq	%rdi, %rbp
	movl	392(%rsp), %edi
	leal	-1(%rdi), %ecx
	cmpl	$3, %ecx
	ja	.LBB10_12
# %bb.6:
	cmpl	$0, 400(%rsp)
	je	.LBB10_12
# %bb.7:
	movq	%rdx, %r12
	movq	%rsi, %r13
	movq	%r8, %r14
	shlq	$32, %rdi
	orq	$40, %rdi
.Ltmp95:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
.Ltmp96:                                # EH_LABEL
# %bb.8:
	testl	%eax, %eax
	jne	.LBB10_11
# %bb.9:
	movq	360(%rsp), %rax
	movq	352(%rsp), %rcx
	movq	336(%rsp), %rdx
	movq	%rbp, 152(%rsp)
	movq	%r13, 144(%rsp)
	movq	%r12, 136(%rsp)
	movq	%r15, 128(%rsp)
	movq	%r14, 120(%rsp)
	movq	%rbx, 112(%rsp)
	movq	%rdx, 104(%rsp)
	movq	344(%rsp), %rdx
	movq	%rdx, 96(%rsp)
	movq	%rcx, 88(%rsp)
	movq	%rax, 80(%rsp)
	movq	368(%rsp), %rax
	movq	%rax, 72(%rsp)
	movq	376(%rsp), %rax
	movq	%rax, 64(%rsp)
	movl	384(%rsp), %eax
	movl	%eax, 12(%rsp)
	movl	400(%rsp), %eax
	movl	%eax, 8(%rsp)
	leaq	152(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	136(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 256(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 264(%rsp)
.Ltmp97:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp98:                                # EH_LABEL
# %bb.10:
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
.Ltmp99:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj(%rip), %rdi
	leaq	160(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp100:                               # EH_LABEL
.LBB10_11:
.Ltmp101:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp102:                               # EH_LABEL
.LBB10_12:
	addq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB10_13:
	.cfi_def_cfa_offset 336
.Ltmp103:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end10:
	.size	_ZN6ninfer3ops5r97003gdn27projection_conv_record_bf16EPK12hip_bfloat16S5_S5_S5_PKiS7_S7_PS3_S8_S8_S8_S8_jjjP12ihipStream_t, .Lfunc_end10-_ZN6ninfer3ops5r97003gdn27projection_conv_record_bf16EPK12hip_bfloat16S5_S5_S5_PKiS7_S7_PS3_S8_S8_S8_S8_jjjP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table10:
.Lexception5:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase5-.Lttbaseref5
.Lttbaseref5:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end5-.Lcst_begin5
.Lcst_begin5:
	.uleb128 .Ltmp95-.Lfunc_begin5          # >> Call Site 1 <<
	.uleb128 .Ltmp102-.Ltmp95               #   Call between .Ltmp95 and .Ltmp102
	.uleb128 .Ltmp103-.Lfunc_begin5         #     jumps to .Ltmp103
	.byte	1                               #   On action: 1
.Lcst_end5:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase5:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_ZN6ninfer3ops5r97003gdn18control_gates_bf16EPK12hip_bfloat16S5_PKfS7_PfS8_jjP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn18control_gates_bf16EPK12hip_bfloat16S5_PKfS7_PfS8_jjP12ihipStream_t
	.prefalign	4, .Lfunc_end11, nop
	.type	_ZN6ninfer3ops5r97003gdn18control_gates_bf16EPK12hip_bfloat16S5_PKfS7_PfS8_jjP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn18control_gates_bf16EPK12hip_bfloat16S5_PKfS7_PfS8_jjP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn18control_gates_bf16EPK12hip_bfloat16S5_PKfS7_PfS8_jjP12ihipStream_t
.Lfunc_begin6:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception6
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$200, %rsp
	.cfi_def_cfa_offset 256
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, %r11
	movq	%rcx, %r15
	movq	%rdx, %r12
	movq	%rsi, %r13
	movq	%rdi, %rbp
	movl	256(%rsp), %edx
	movl	264(%rsp), %ebx
	testq	%rbx, %rbx
	setne	%al
	testq	%rdx, %rdx
	setne	%cl
	movq	272(%rsp), %r9
	imulq	%rdx, %rbx
	movq	%rsi, %xmm0
	movq	%rdi, %xmm1
	punpcklqdq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0]
	movq	%r15, %xmm0
	movq	%r12, %xmm2
	punpcklqdq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0]
	testq	%r8, %r8
	setne	%dl
	testq	%r11, %r11
	setne	%sil
	leaq	-1(%rbx), %rdi
	movabsq	$1099511627520, %r10            # imm = 0xFFFFFFFF00
	cmpq	%r10, %rdi
	setb	%dil
	testq	%r9, %r9
	setne	%r14b
	pxor	%xmm0, %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	shufps	$221, %xmm2, %xmm0              # xmm0 = xmm0[1,3],xmm2[1,3]
	shufps	$136, %xmm2, %xmm1              # xmm1 = xmm1[0,2],xmm2[0,2]
	andps	%xmm0, %xmm1
	movmskps	%xmm1, %r10d
	testl	%r10d, %r10d
	sete	%r10b
	andb	%sil, %r14b
	andb	%dl, %r14b
	andb	%r10b, %r14b
	andb	%al, %cl
	andb	%r14b, %cl
	andb	%dil, %cl
	movl	$1, %eax
	cmpb	$1, %cl
	jne	.LBB11_6
# %bb.1:
	movq	%r8, 16(%rsp)                   # 8-byte Spill
	movq	%r11, %r14
	leaq	255(%rbx), %rax
	shrq	$8, %rax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp104:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
.Ltmp105:                               # EH_LABEL
# %bb.2:
	testl	%eax, %eax
	jne	.LBB11_5
# %bb.3:
	movq	%rbp, 120(%rsp)
	movq	%r13, 112(%rsp)
	movq	%r12, 104(%rsp)
	movq	%r15, 96(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 88(%rsp)
	movq	%r14, 80(%rsp)
	movq	%rbx, 72(%rsp)
	movl	256(%rsp), %eax
	movl	%eax, 12(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp106:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp107:                               # EH_LABEL
# %bb.4:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp108:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp109:                               # EH_LABEL
.LBB11_5:
.Ltmp110:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp111:                               # EH_LABEL
.LBB11_6:
	addq	$200, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB11_7:
	.cfi_def_cfa_offset 256
.Ltmp112:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end11:
	.size	_ZN6ninfer3ops5r97003gdn18control_gates_bf16EPK12hip_bfloat16S5_PKfS7_PfS8_jjP12ihipStream_t, .Lfunc_end11-_ZN6ninfer3ops5r97003gdn18control_gates_bf16EPK12hip_bfloat16S5_PKfS7_PfS8_jjP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table11:
.Lexception6:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase6-.Lttbaseref6
.Lttbaseref6:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end6-.Lcst_begin6
.Lcst_begin6:
	.uleb128 .Ltmp104-.Lfunc_begin6         # >> Call Site 1 <<
	.uleb128 .Ltmp111-.Ltmp104              #   Call between .Ltmp104 and .Ltmp111
	.uleb128 .Ltmp112-.Lfunc_begin6         #     jumps to .Ltmp112
	.byte	1                               #   On action: 1
.Lcst_end6:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase6:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_ZN6ninfer3ops5r97003gdn25bf16_projected_control_t1EPK12hip_bfloat16S5_S5_PKfS7_PfS8_P12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn25bf16_projected_control_t1EPK12hip_bfloat16S5_S5_PKfS7_PfS8_P12ihipStream_t
	.prefalign	4, .Lfunc_end12, nop
	.type	_ZN6ninfer3ops5r97003gdn25bf16_projected_control_t1EPK12hip_bfloat16S5_S5_PKfS7_PfS8_P12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn25bf16_projected_control_t1EPK12hip_bfloat16S5_S5_PKfS7_PfS8_P12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn25bf16_projected_control_t1EPK12hip_bfloat16S5_S5_PKfS7_PfS8_P12ihipStream_t
.Lfunc_begin7:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception7
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$200, %rsp
	.cfi_def_cfa_offset 256
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, %xmm0
	movq	%r8, %xmm1
	punpcklqdq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0]
	movq	%rsi, %xmm0
	movq	%rdi, %xmm2
	punpcklqdq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0]
	movq	%rcx, %xmm0
	movq	%rdx, %xmm3
	punpcklqdq	%xmm0, %xmm3            # xmm3 = xmm3[0],xmm0[0]
	pxor	%xmm0, %xmm0
	pcmpeqd	%xmm0, %xmm3
	pshufd	$177, %xmm3, %xmm4              # xmm4 = xmm3[1,0,3,2]
	pand	%xmm3, %xmm4
	pcmpeqd	%xmm0, %xmm2
	pshufd	$177, %xmm2, %xmm3              # xmm3 = xmm2[1,0,3,2]
	pand	%xmm2, %xmm3
	packssdw	%xmm4, %xmm3
	pcmpeqd	%xmm0, %xmm1
	pshufd	$177, %xmm1, %xmm2              # xmm2 = xmm1[1,0,3,2]
	pcmpeqd	256(%rsp), %xmm0
	pand	%xmm1, %xmm2
	pshufd	$177, %xmm0, %xmm1              # xmm1 = xmm0[1,0,3,2]
	pand	%xmm0, %xmm1
	packssdw	%xmm1, %xmm2
	packssdw	%xmm2, %xmm3
	pmovmskb	%xmm3, %r10d
	movl	$1, %eax
	testl	$43690, %r10d                   # imm = 0xAAAA
	jne	.LBB12_27
# %bb.1:
	leaq	10240(%rdi), %rbp
	leaq	491520(%rsi), %r13
	cmpq	%rdi, %r13
	seta	%r10b
	cmpq	%rsi, %rbp
	seta	%r11b
	testb	%r11b, %r10b
	jne	.LBB12_27
# %bb.2:
	leaq	491520(%rdx), %r12
	cmpq	%rdi, %r12
	seta	%r10b
	cmpq	%rdx, %rbp
	seta	%r11b
	testb	%r11b, %r10b
	jne	.LBB12_27
# %bb.3:
	leaq	192(%rcx), %r14
	cmpq	%rdi, %r14
	seta	%r10b
	cmpq	%rcx, %rbp
	seta	%r11b
	testb	%r11b, %r10b
	jne	.LBB12_27
# %bb.4:
	leaq	192(%r8), %r10
	cmpq	%rdi, %r10
	seta	%r11b
	cmpq	%r8, %rbp
	seta	%bl
	testb	%bl, %r11b
	jne	.LBB12_27
# %bb.5:
	leaq	192(%r9), %r11
	cmpq	%rdi, %r11
	seta	%bl
	cmpq	%r9, %rbp
	seta	%r15b
	testb	%r15b, %bl
	jne	.LBB12_27
# %bb.6:
	movq	256(%rsp), %rbx
	leaq	192(%rbx), %r15
	cmpq	%rdi, %r15
	seta	%r15b
	cmpq	%rbx, %rbp
	seta	%bpl
	testb	%bpl, %r15b
	jne	.LBB12_27
# %bb.7:
	cmpq	%rsi, %r12
	seta	%bpl
	cmpq	%rdx, %r13
	seta	%r15b
	testb	%r15b, %bpl
	jne	.LBB12_27
# %bb.8:
	cmpq	%rsi, %r14
	seta	%bpl
	cmpq	%rcx, %r13
	seta	%r15b
	testb	%r15b, %bpl
	jne	.LBB12_27
# %bb.9:
	cmpq	%rsi, %r10
	seta	%bpl
	cmpq	%r8, %r13
	seta	%r15b
	testb	%r15b, %bpl
	jne	.LBB12_27
# %bb.10:
	cmpq	%rsi, %r11
	seta	%bpl
	cmpq	%r9, %r13
	seta	%r15b
	testb	%r15b, %bpl
	jne	.LBB12_27
# %bb.11:
	leaq	192(%rbx), %r15
	cmpq	%rsi, %r15
	seta	%bpl
	cmpq	%rbx, %r13
	seta	%r15b
	testb	%r15b, %bpl
	jne	.LBB12_27
# %bb.12:
	cmpq	%rdx, %r14
	seta	%bpl
	cmpq	%rcx, %r12
	seta	%r15b
	testb	%r15b, %bpl
	jne	.LBB12_27
# %bb.13:
	movq	%rdi, 24(%rsp)                  # 8-byte Spill
	movq	%rsi, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, %r13
	cmpq	%rdx, %r10
	seta	%cl
	cmpq	%r8, %r12
	seta	%sil
	testb	%sil, %cl
	jne	.LBB12_27
# %bb.14:
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
	cmpq	%rdx, %r11
	seta	%cl
	movq	%r9, 8(%rsp)                    # 8-byte Spill
	cmpq	%r9, %r12
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.15:
	leaq	192(%rbx), %rcx
	cmpq	16(%rsp), %rcx                  # 8-byte Folded Reload
	seta	%cl
	cmpq	%rbx, %r12
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.16:
	movq	%r8, %rbp
	cmpq	%r13, %r10
	seta	%cl
	cmpq	%r8, %r14
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.17:
	cmpq	%r13, %r11
	seta	%cl
	cmpq	8(%rsp), %r14                   # 8-byte Folded Reload
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.18:
	leaq	192(%rbx), %rcx
	cmpq	%r13, %rcx
	seta	%cl
	cmpq	%rbx, %r14
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.19:
	cmpq	%rbp, %r11
	seta	%cl
	cmpq	8(%rsp), %r10                   # 8-byte Folded Reload
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.20:
	leaq	192(%rbx), %rcx
	cmpq	%rbp, %rcx
	seta	%cl
	cmpq	%rbx, %r10
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.21:
	leaq	192(%rbx), %rcx
	cmpq	8(%rsp), %rcx                   # 8-byte Folded Reload
	seta	%cl
	cmpq	%rbx, %r11
	seta	%dl
	testb	%dl, %cl
	jne	.LBB12_27
# %bb.22:
.Ltmp113:                               # EH_LABEL
	movq	264(%rsp), %r9
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967344, %rdi               # imm = 0x100000030
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
.Ltmp114:                               # EH_LABEL
# %bb.23:
	testl	%eax, %eax
	jne	.LBB12_26
# %bb.24:
	movq	24(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 136(%rsp)
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 128(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 120(%rsp)
	movq	%r13, 112(%rsp)
	movq	%rbp, 104(%rsp)
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%rax, 96(%rsp)
	movq	%rbx, 88(%rsp)
	leaq	136(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 192(%rsp)
.Ltmp115:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	72(%rsp), %rdi
	leaq	56(%rsp), %rsi
	leaq	48(%rsp), %rdx
	leaq	40(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp116:                               # EH_LABEL
# %bb.25:
	movq	72(%rsp), %rsi
	movl	80(%rsp), %edx
	movq	56(%rsp), %rcx
	movl	64(%rsp), %r8d
.Ltmp117:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_(%rip), %rdi
	leaq	144(%rsp), %r9
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	56(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp118:                               # EH_LABEL
.LBB12_26:
.Ltmp119:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp120:                               # EH_LABEL
.LBB12_27:
	addq	$200, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB12_28:
	.cfi_def_cfa_offset 256
.Ltmp121:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end12:
	.size	_ZN6ninfer3ops5r97003gdn25bf16_projected_control_t1EPK12hip_bfloat16S5_S5_PKfS7_PfS8_P12ihipStream_t, .Lfunc_end12-_ZN6ninfer3ops5r97003gdn25bf16_projected_control_t1EPK12hip_bfloat16S5_S5_PKfS7_PfS8_P12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table12:
.Lexception7:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase7-.Lttbaseref7
.Lttbaseref7:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end7-.Lcst_begin7
.Lcst_begin7:
	.uleb128 .Ltmp113-.Lfunc_begin7         # >> Call Site 1 <<
	.uleb128 .Ltmp120-.Ltmp113              #   Call between .Ltmp113 and .Ltmp120
	.uleb128 .Ltmp121-.Lfunc_begin7         #     jumps to .Ltmp121
	.byte	1                               #   On action: 1
.Lcst_end7:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase7:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_ZN6ninfer3ops5r97003gdn15copy_state_fp32EPKfPfmP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn15copy_state_fp32EPKfPfmP12ihipStream_t
	.prefalign	4, .Lfunc_end13, nop
	.type	_ZN6ninfer3ops5r97003gdn15copy_state_fp32EPKfPfmP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn15copy_state_fp32EPKfPfmP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn15copy_state_fp32EPKfPfmP12ihipStream_t
.Lfunc_begin8:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception8
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$112, %rsp
	.cfi_def_cfa_offset 144
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movl	$1, %eax
	testq	%rcx, %rcx
	je	.LBB13_10
# %bb.1:
	movq	%rsi, %rbx
	movq	%rdi, %r14
	cmpq	%rsi, %rdi
	je	.LBB13_10
# %bb.2:
	testq	%r14, %r14
	je	.LBB13_10
# %bb.3:
	testq	%rbx, %rbx
	je	.LBB13_10
# %bb.4:
	movq	%rcx, %r9
	leaq	-1(%rdx), %rcx
	movabsq	$1099511627519, %rsi            # imm = 0xFFFFFFFEFF
	cmpq	%rsi, %rcx
	ja	.LBB13_10
# %bb.5:
	movq	%rdx, %r15
	leaq	255(%rdx), %rax
	shrq	$8, %rax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp122:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
.Ltmp123:                               # EH_LABEL
# %bb.6:
	testl	%eax, %eax
	jne	.LBB13_9
# %bb.7:
	movq	%r14, 72(%rsp)
	movq	%rbx, 64(%rsp)
	movq	%r15, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
.Ltmp124:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp125:                               # EH_LABEL
# %bb.8:
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
.Ltmp126:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp127:                               # EH_LABEL
.LBB13_9:
.Ltmp128:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp129:                               # EH_LABEL
.LBB13_10:
	addq	$112, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB13_11:
	.cfi_def_cfa_offset 144
.Ltmp130:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end13:
	.size	_ZN6ninfer3ops5r97003gdn15copy_state_fp32EPKfPfmP12ihipStream_t, .Lfunc_end13-_ZN6ninfer3ops5r97003gdn15copy_state_fp32EPKfPfmP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table13:
.Lexception8:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase8-.Lttbaseref8
.Lttbaseref8:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end8-.Lcst_begin8
.Lcst_begin8:
	.uleb128 .Ltmp122-.Lfunc_begin8         # >> Call Site 1 <<
	.uleb128 .Ltmp129-.Ltmp122              #   Call between .Ltmp122 and .Ltmp129
	.uleb128 .Ltmp130-.Lfunc_begin8         #     jumps to .Ltmp130
	.byte	1                               #   On action: 1
.Lcst_end8:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase8:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_ZN6ninfer3ops5r97003gdn19recurrent_bf16_fp32EPK12hip_bfloat16S5_S5_PKfS7_S7_PfPS3_jjjjjffP12ihipStream_t # -- Begin function _ZN6ninfer3ops5r97003gdn19recurrent_bf16_fp32EPK12hip_bfloat16S5_S5_PKfS7_S7_PfPS3_jjjjjffP12ihipStream_t
	.prefalign	4, .Lfunc_end14, nop
	.type	_ZN6ninfer3ops5r97003gdn19recurrent_bf16_fp32EPK12hip_bfloat16S5_S5_PKfS7_S7_PfPS3_jjjjjffP12ihipStream_t,@function
_ZN6ninfer3ops5r97003gdn19recurrent_bf16_fp32EPK12hip_bfloat16S5_S5_PKfS7_S7_PfPS3_jjjjjffP12ihipStream_t: # @_ZN6ninfer3ops5r97003gdn19recurrent_bf16_fp32EPK12hip_bfloat16S5_S5_PKfS7_S7_PfPS3_jjjjjffP12ihipStream_t
.Lfunc_begin9:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception9
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 336
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r14
	movl	376(%rsp), %ebp
	movl	368(%rsp), %r11d
	movl	352(%rsp), %eax
	decl	%eax
	cmpl	$256, %eax                      # imm = 0x100
	setae	%al
	cmpl	$0, 360(%rsp)
	sete	%dl
	testl	%r11d, %r11d
	sete	%r10b
	orb	%dl, %r10b
	orb	%al, %r10b
	testl	%ebp, %ebp
	sete	%al
	orb	%r10b, %al
	jne	.LBB14_1
# %bb.2:
	movl	%ebp, %eax
	xorl	%edx, %edx
	divl	%r11d
	testl	%edx, %edx
	sete	%al
	cmpl	$0, 384(%rsp)
	setne	%dl
	andb	%al, %dl
	jmp	.LBB14_3
.LBB14_1:
	xorl	%edx, %edx
.LBB14_3:
	movq	%rsi, %xmm2
	movq	%rdi, %xmm7
	punpcklqdq	%xmm2, %xmm7            # xmm7 = xmm7[0],xmm2[0]
	movq	%rcx, %xmm2
	movq	%r14, %xmm6
	punpcklqdq	%xmm2, %xmm6            # xmm6 = xmm6[0],xmm2[0]
	movq	%r9, %xmm2
	movq	%r8, %xmm3
	punpcklqdq	%xmm2, %xmm3            # xmm3 = xmm3[0],xmm2[0]
	movq	344(%rsp), %xmm2                # xmm2 = mem[0],zero
	movq	336(%rsp), %xmm4                # xmm4 = mem[0],zero
	punpcklqdq	%xmm2, %xmm4            # xmm4 = xmm4[0],xmm2[0]
	movd	%xmm0, %eax
	addl	%eax, %eax
	cmpl	$-16777216, %eax                # imm = 0xFF000000
	setb	%r10b
	pxor	%xmm2, %xmm2
	pcmpeqd	%xmm2, %xmm4
	pshufd	$177, %xmm4, %xmm5              # xmm5 = xmm4[1,0,3,2]
	pand	%xmm4, %xmm5
	pcmpeqd	%xmm2, %xmm3
	pshufd	$177, %xmm3, %xmm4              # xmm4 = xmm3[1,0,3,2]
	pand	%xmm3, %xmm4
	packssdw	%xmm5, %xmm4
	pcmpeqd	%xmm2, %xmm6
	pshufd	$177, %xmm6, %xmm3              # xmm3 = xmm6[1,0,3,2]
	pand	%xmm6, %xmm3
	pcmpeqd	%xmm2, %xmm7
	pshufd	$177, %xmm7, %xmm2              # xmm2 = xmm7[1,0,3,2]
	pand	%xmm7, %xmm2
	packssdw	%xmm3, %xmm2
	packssdw	%xmm4, %xmm2
	pmovmskb	%xmm2, %r11d
	movl	$1, %eax
	testl	$43690, %r11d                   # imm = 0xAAAA
	jne	.LBB14_12
# %bb.4:
	testb	%r10b, %r10b
	je	.LBB14_12
# %bb.5:
	testb	%dl, %dl
	je	.LBB14_12
# %bb.6:
	movq	392(%rsp), %r10
	movd	%xmm1, %edx
	testl	%edx, %edx
	sets	%r11b
	movl	%edx, %ebx
	andl	$2147483647, %ebx               # imm = 0x7FFFFFFF
	addl	$-8388608, %ebx                 # imm = 0xFF800000
	cmpl	$2130706432, %ebx               # imm = 0x7F000000
	setae	%bl
	orb	%r11b, %bl
	decl	%edx
	cmpl	$8388607, %edx                  # imm = 0x7FFFFF
	setae	%dl
	andb	%bl, %dl
	testq	%r10, %r10
	sete	%r11b
	orb	%dl, %r11b
	jne	.LBB14_12
# %bb.7:
	movd	%xmm0, 12(%rsp)                 # 4-byte Folded Spill
	movd	%xmm1, 16(%rsp)                 # 4-byte Folded Spill
	movq	%rdi, %r13
	movl	%ebp, %eax
	movq	%rsi, %rbp
	movq	%rcx, %r12
	movq	%r8, %r15
	movq	%r9, %rbx
	movl	%eax, %eax
	movabsq	$4294967296, %rdi               # imm = 0x100000000
	orq	%rax, %rdi
.Ltmp131:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	movq	%r10, %r9
	callq	__hipPushCallConfiguration@PLT
.Ltmp132:                               # EH_LABEL
# %bb.8:
	testl	%eax, %eax
	movd	16(%rsp), %xmm0                 # 4-byte Folded Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movd	12(%rsp), %xmm1                 # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jne	.LBB14_11
# %bb.9:
	movq	%r13, 152(%rsp)
	movq	%rbp, 144(%rsp)
	movq	%r14, 136(%rsp)
	movq	%r12, 128(%rsp)
	movq	%r15, 120(%rsp)
	movq	%rbx, 112(%rsp)
	movq	336(%rsp), %rax
	movq	%rax, 104(%rsp)
	movq	344(%rsp), %rax
	movq	%rax, 96(%rsp)
	movl	352(%rsp), %eax
	movl	%eax, 44(%rsp)
	movl	360(%rsp), %eax
	movl	%eax, 40(%rsp)
	movl	368(%rsp), %eax
	movl	%eax, 36(%rsp)
	movl	376(%rsp), %eax
	movl	%eax, 32(%rsp)
	movl	384(%rsp), %eax
	movl	%eax, 28(%rsp)
	movd	%xmm1, 24(%rsp)
	movd	%xmm0, 20(%rsp)
	leaq	152(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	136(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	44(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	36(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	32(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	28(%rsp), %rax
	movq	%rax, 256(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 264(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 272(%rsp)
.Ltmp133:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	80(%rsp), %rdi
	leaq	64(%rsp), %rsi
	leaq	56(%rsp), %rdx
	leaq	48(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp134:                               # EH_LABEL
# %bb.10:
	movq	80(%rsp), %rsi
	movl	88(%rsp), %edx
	movq	64(%rsp), %rcx
	movl	72(%rsp), %r8d
.Ltmp135:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff(%rip), %rdi
	leaq	160(%rsp), %r9
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp136:                               # EH_LABEL
.LBB14_11:
.Ltmp137:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp138:                               # EH_LABEL
.LBB14_12:
	addq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB14_13:
	.cfi_def_cfa_offset 336
.Ltmp139:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end14:
	.size	_ZN6ninfer3ops5r97003gdn19recurrent_bf16_fp32EPK12hip_bfloat16S5_S5_PKfS7_S7_PfPS3_jjjjjffP12ihipStream_t, .Lfunc_end14-_ZN6ninfer3ops5r97003gdn19recurrent_bf16_fp32EPK12hip_bfloat16S5_S5_PKfS7_S7_PfPS3_jjjjjffP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table14:
.Lexception9:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase9-.Lttbaseref9
.Lttbaseref9:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end9-.Lcst_begin9
.Lcst_begin9:
	.uleb128 .Ltmp131-.Lfunc_begin9         # >> Call Site 1 <<
	.uleb128 .Ltmp138-.Ltmp131              #   Call between .Ltmp131 and .Ltmp138
	.uleb128 .Ltmp139-.Lfunc_begin9         #     jumps to .Ltmp139
	.byte	1                               #   On action: 1
.Lcst_end9:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase9:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end15, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end15:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end15-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end16, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	movl	%r8d, (%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	movq	%rsp, %rax
	movq	%rax, 112(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end16:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj, .Lfunc_end16-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end17, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end17:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end17-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end18, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end18:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end18-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end19, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end19:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj, .Lfunc_end19-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end20, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.cfi_startproc
# %bb.0:
	subq	$200, %rsp
	.cfi_def_cfa_offset 208
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%r9, 48(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	208(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	216(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	224(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	240(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	248(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	256(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$216, %rsp
	.cfi_adjust_cfa_offset -216
	retq
.Lfunc_end20:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj, .Lfunc_end20-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end21, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.cfi_startproc
# %bb.0:
	subq	$216, %rsp
	.cfi_def_cfa_offset 224
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%r9, 48(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	224(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	240(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	248(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	256(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	264(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	272(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	280(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$232, %rsp
	.cfi_adjust_cfa_offset -232
	retq
.Lfunc_end21:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj, .Lfunc_end21-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end22, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%r9, 48(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end22:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj, .Lfunc_end22-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end23, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%r9, 48(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end23:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_, .Lfunc_end23-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end24, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end24:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm, .Lfunc_end24-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end25, nop    # -- Begin function _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff,@function
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff: # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.cfi_startproc
# %bb.0:
	subq	$232, %rsp
	.cfi_def_cfa_offset 240
	movq	%rdi, 104(%rsp)
	movq	%rsi, 96(%rsp)
	movq	%rdx, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movq	%r8, 72(%rsp)
	movq	%r9, 64(%rsp)
	movss	%xmm0, 12(%rsp)
	movss	%xmm1, 8(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	240(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	248(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	256(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	264(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	272(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	280(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	288(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$248, %rsp
	.cfi_adjust_cfa_offset -248
	retq
.Lfunc_end25:
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff, .Lfunc_end25-_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end26, nop    # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_ad5f8cfca7299000(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB26_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_ad5f8cfca7299000(%rip)
.LBB26_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rsi
	leaq	.L__unnamed_5(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rsi
	leaq	.L__unnamed_6(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj(%rip), %rsi
	leaq	.L__unnamed_7(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_(%rip), %rsi
	leaq	.L__unnamed_8(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj(%rip), %rsi
	leaq	.L__unnamed_9(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj(%rip), %rsi
	leaq	.L__unnamed_10(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj(%rip), %rsi
	leaq	.L__unnamed_11(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_(%rip), %rsi
	leaq	.L__unnamed_12(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm(%rip), %rsi
	leaq	.L__unnamed_13(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff(%rip), %rsi
	leaq	.L__unnamed_14(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	leaq	__hip_module_dtor(%rip), %rdi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit@PLT                      # TAILCALL
.Lfunc_end26:
	.size	__hip_module_ctor, .Lfunc_end26-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end27, nop    # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_ad5f8cfca7299000(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB27_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_ad5f8cfca7299000(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB27_2:
	retq
.Lfunc_end27:
	.size	__hip_module_dtor, .Lfunc_end27-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.section	.data.rel.ro,"aw",@progbits
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm, 8

	.type	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff,@object # @_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.p2align	3, 0x0
_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff:
	.quad	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.size	_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj"
	.size	.L__unnamed_1, 98

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_"
	.size	.L__unnamed_2, 106

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj"
	.size	.L__unnamed_3, 110

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj"
	.size	.L__unnamed_4, 123

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj"
	.size	.L__unnamed_5, 110

	.type	.L__unnamed_6,@object           # @5
.L__unnamed_6:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj"
	.size	.L__unnamed_6, 111

	.type	.L__unnamed_7,@object           # @6
.L__unnamed_7:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj"
	.size	.L__unnamed_7, 111

	.type	.L__unnamed_8,@object           # @7
.L__unnamed_8:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_"
	.size	.L__unnamed_8, 132

	.type	.L__unnamed_9,@object           # @8
.L__unnamed_9:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj"
	.size	.L__unnamed_9, 122

	.type	.L__unnamed_10,@object          # @9
.L__unnamed_10:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj"
	.size	.L__unnamed_10, 123

	.type	.L__unnamed_11,@object          # @10
.L__unnamed_11:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj"
	.size	.L__unnamed_11, 94

	.type	.L__unnamed_12,@object          # @11
.L__unnamed_12:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_"
	.size	.L__unnamed_12, 107

	.type	.L__unnamed_13,@object          # @12
.L__unnamed_13:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm"
	.size	.L__unnamed_13, 64

	.type	.L__unnamed_14,@object          # @13
.L__unnamed_14:
	.asciz	"_ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff"
	.size	.L__unnamed_14, 102

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_ad5f8cfca7299000
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_ad5f8cfca7299000,@object # @__hip_gpubin_handle_ad5f8cfca7299000
	.local	__hip_gpubin_handle_ad5f8cfca7299000
	.comm	__hip_gpubin_handle_ad5f8cfca7299000,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_ad5f8cfca7299000,@object # @__hip_cuid_ad5f8cfca7299000
	.bss
	.globl	__hip_cuid_ad5f8cfca7299000
__hip_cuid_ad5f8cfca7299000:
	.byte	0                               # 0x0
	.size	__hip_cuid_ad5f8cfca7299000, 1

	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_140__device_stub__causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_160__device_stub__projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156__device_stub__projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_171__device_stub__causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_148__device_stub__causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_146__device_stub__projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_144__device_stub__projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_135__device_stub__control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_147__device_stub__bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__copy_fp32_kernelEPKfPfm
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131__device_stub__recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_125causal_conv1d_silu_kernelEPK12hip_bfloat16S6_S6_PS4_S7_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_145projection_conv_prefill_direct_scatter_kernelILj4EEEvPK12hip_bfloat16S7_S7_S7_PS5_S8_S8_S8_
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_141projection_prefill_publish_history_kernelEPK12hip_bfloat16S6_PS4_
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj4EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_156causal_conv1d_publish_short_history_qualification_kernelEPK12hip_bfloat16S6_PS4_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj8EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj16EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_133causal_conv1d_silu_prefill_kernelILj32EEEvPK12hip_bfloat16S7_S7_PS5_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_131projection_conv_snapshot_kernelEPK12hip_bfloat16S6_S6_PS4_PKiS9_S9_S7_S7_S7_S7_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_129projection_conv_record_kernelEPK12hip_bfloat16S6_S6_S6_PKiS8_S8_PS4_S9_S9_S9_S9_jj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_120control_gates_kernelEPK12hip_bfloat16S6_PKfS8_PfS9_mj
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_132bf16_projected_control_t1_kernelEPK12hip_bfloat16S6_S6_PKfS8_PfS9_
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116copy_fp32_kernelEPKfPfm
	.addrsig_sym _ZN6ninfer3ops5r97003gdn12_GLOBAL__N_116recurrent_kernelEPK12hip_bfloat16S6_S6_PKfS8_S8_PfPS4_jjjjjff
	.addrsig_sym __hip_fatbin_ad5f8cfca7299000
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_ad5f8cfca7299000

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
