
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx1201
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_,"axG",@progbits,_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_,comdat
	.globl	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_ ; -- Begin function _ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.p2align	8
	.type	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_,@function
_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_: ; @_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s10, s[0:1], 0x24
	s_load_b64 s[0:1], s[0:1], 0x10
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, v0
	s_mov_b32 s2, ttmp9
	s_mov_b32 s3, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v3, v2
	s_mul_u64 s[8:9], s[2:3], 0x2800
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[6:7], s[6:7], s[8:9]
	s_and_b32 s9, s10, 0xffff
	s_mov_b32 s8, s3
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_lshlrev_b64_e32 v[4:5], 1, v[1:2]
	v_add_nc_u32_e32 v1, s9, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, vcc_lo, s4, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s5, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, s6, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s7, v5, vcc_lo
	global_load_u16 v6, v[6:7], off
	global_load_u16 v4, v[4:5], off
	v_cmp_lt_u32_e32 vcc_lo, 0x13ff, v1
	s_or_b32 s8, vcc_lo, s8
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 16, v6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 16, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, v5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_1
; %bb.2:
	s_or_b32 exec_lo, exec_lo, s8
	v_lshlrev_b32_e32 v1, 2, v0
	s_mov_b32 s4, exec_lo
	ds_store_b32 v1, v3
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 0x80, v0
	s_cbranch_execz .LBB0_4
; %bb.3:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:2
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB0_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 64, v0
	s_cbranch_execz .LBB0_6
; %bb.5:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:1
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB0_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_8
; %bb.7:
	ds_load_2addr_b32 v[2:3], v1 offset1:32
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB0_10
; %bb.9:
	ds_load_2addr_b32 v[2:3], v1 offset1:16
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB0_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 8, v0
	s_cbranch_execz .LBB0_12
; %bb.11:
	ds_load_2addr_b32 v[2:3], v1 offset1:8
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB0_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 4, v0
	s_cbranch_execz .LBB0_14
; %bb.13:
	ds_load_2addr_b32 v[2:3], v1 offset1:4
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB0_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 2, v0
	s_cbranch_execz .LBB0_16
; %bb.15:
	ds_load_2addr_b32 v[2:3], v1 offset1:2
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB0_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB0_18
; %bb.17:
	ds_load_2addr_b32 v[2:3], v1 offset1:1
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v3, v2
	ds_store_b32 v1, v0
.LBB0_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB0_24
; %bb.19:
	v_mov_b32_e32 v0, 0
	ds_load_b32 v0, v0
	s_wait_dscnt 0x0
	v_and_b32_e32 v1, 0x7f800000, v0
	v_readfirstlane_b32 s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v1
	s_cbranch_vccnz .LBB0_21
; %bb.20:
	s_wait_alu depctr_sa_sdst(0)
	s_bfe_u32 s5, s4, 0x10010
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s5, s4, s5
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s5, 0x7fff
	s_cbranch_execz .LBB0_22
	s_branch .LBB0_23
.LBB0_21:
                                        ; implicit-def: $sgpr5
.LBB0_22:
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s4, 0xffff
	s_or_b32 s6, s4, 0x10000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s5, s4, s6
.LBB0_23:
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s5
	s_lshl_b64 s[2:3], s[2:3], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	global_store_d16_hi_b16 v0, v1, s[0:1]
.LBB0_24:
	s_endpgm
.Lfunc_end0:
	.size	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_, .Lfunc_end0-_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
		.amdhsa_group_segment_fixed_size 1024
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
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 11
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_,"axG",@progbits,_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.num_vgpr, 8
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.numbered_sgpr, 11
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 952
; TotalNumSgprs: 13
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1024 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 13
; NumVGPRsForWavesPerEU: 8
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_,"axG",@progbits,_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_,comdat
	.globl	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_ ; -- Begin function _ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.p2align	8
	.type	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_,@function
_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_: ; @_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b32 s13, s[0:1], 0x34
	s_load_b64 s[0:1], s[0:1], 0x20
	s_sub_co_i32 s2, ttmp9, 48
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, v0
	s_min_u32 s2, s2, ttmp9
	s_cmp_gt_u32 ttmp9, 47
	s_mov_b32 s3, 0
	s_cselect_b32 s12, -1, 0
	v_mov_b32_e32 v3, v2
	s_and_b32 s16, s12, exec_lo
	s_mul_u64 s[14:15], s[2:3], 0x2800
	s_wait_kmcnt 0x0
	s_cselect_b32 s7, s9, s7
	s_cselect_b32 s6, s8, s6
	s_and_b32 s8, s13, 0xffff
	s_add_nc_u64 s[6:7], s[6:7], s[14:15]
	s_mov_b32 s9, s3
.LBB1_1:                                ; =>This Inner Loop Header: Depth=1
	v_lshlrev_b64_e32 v[4:5], 1, v[1:2]
	v_add_nc_u32_e32 v1, s8, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, vcc_lo, s4, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s5, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, s6, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s7, v5, vcc_lo
	global_load_u16 v6, v[6:7], off
	global_load_u16 v4, v[4:5], off
	v_cmp_lt_u32_e32 vcc_lo, 0x13ff, v1
	s_or_b32 s9, vcc_lo, s9
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v5, 16, v6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 16, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, v5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s9
	s_cbranch_execnz .LBB1_1
; %bb.2:
	s_or_b32 exec_lo, exec_lo, s9
	v_lshlrev_b32_e32 v1, 2, v0
	s_mov_b32 s4, exec_lo
	ds_store_b32 v1, v3
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 0x80, v0
	s_cbranch_execz .LBB1_4
; %bb.3:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:2
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB1_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 64, v0
	s_cbranch_execz .LBB1_6
; %bb.5:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:1
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB1_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB1_8
; %bb.7:
	ds_load_2addr_b32 v[2:3], v1 offset1:32
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB1_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB1_10
; %bb.9:
	ds_load_2addr_b32 v[2:3], v1 offset1:16
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB1_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 8, v0
	s_cbranch_execz .LBB1_12
; %bb.11:
	ds_load_2addr_b32 v[2:3], v1 offset1:8
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB1_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 4, v0
	s_cbranch_execz .LBB1_14
; %bb.13:
	ds_load_2addr_b32 v[2:3], v1 offset1:4
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB1_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 2, v0
	s_cbranch_execz .LBB1_16
; %bb.15:
	ds_load_2addr_b32 v[2:3], v1 offset1:2
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v3, v2
	ds_store_b32 v1, v2
.LBB1_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB1_18
; %bb.17:
	ds_load_2addr_b32 v[2:3], v1 offset1:1
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v3, v2
	ds_store_b32 v1, v0
.LBB1_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB1_24
; %bb.19:
	v_mov_b32_e32 v0, 0
	ds_load_b32 v0, v0
	s_wait_dscnt 0x0
	v_and_b32_e32 v1, 0x7f800000, v0
	v_readfirstlane_b32 s5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v1
	s_cbranch_vccnz .LBB1_21
; %bb.20:
	s_bfe_u32 s4, s5, 0x10010
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s4, 0x7fff
	s_cbranch_execz .LBB1_22
	s_branch .LBB1_23
.LBB1_21:
                                        ; implicit-def: $sgpr4
.LBB1_22:
	s_and_b32 s4, s5, 0xffff
	s_or_b32 s6, s5, 0x10000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s4, s5, s6
.LBB1_23:
	s_and_b32 s5, s12, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s4
	s_cselect_b32 s1, s1, s11
	s_cselect_b32 s0, s0, s10
	s_lshl_b64 s[2:3], s[2:3], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	global_store_d16_hi_b16 v0, v1, s[0:1]
.LBB1_24:
	s_endpgm
.Lfunc_end1:
	.size	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_, .Lfunc_end1-_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
		.amdhsa_group_segment_fixed_size 1024
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 17
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_,"axG",@progbits,_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.num_vgpr, 8
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.numbered_sgpr, 17
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 976
; TotalNumSgprs: 19
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1024 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 19
; NumVGPRsForWavesPerEU: 8
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_,"axG",@progbits,_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_,comdat
	.globl	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_ ; -- Begin function _ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.p2align	8
	.type	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_,@function
_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_: ; @_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 48, v0
	s_cbranch_execz .LBB2_4
; %bb.1:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v2, 1, v0
	v_lshlrev_b32_e32 v1, 2, v0
	s_wait_kmcnt 0x0
	global_load_u16 v3, v2, s[4:5]
	global_load_b32 v4, v1, s[10:11]
	global_load_d16_b16 v0, v2, s[6:7]
	global_load_b32 v2, v1, s[8:9]
	s_load_b128 s[4:7], s[0:1], 0x20
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v3, 16, v3
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v4, v3
	v_cmpx_nlt_f32_e32 0x41a00000, v3
	s_cbranch_execz .LBB2_3
; %bb.2:
	v_mul_f32_e32 v4, 0x3fb8aa3b, v3
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v3
	s_mov_b32 s1, 0x3e9b6dac
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rndne_f32_e32 v5, v4
	v_fma_f32 v6, 0x3fb8aa3b, v3, -v4
	v_sub_f32_e32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v6, v3, 0x32a5705f, v6
	v_cvt_i32_f32_e32 v5, v5
	v_add_f32_e32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, 0x7f800000, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, 1.0, v3
	v_frexp_mant_f32_e32 v5, v4
	v_frexp_exp_i32_f32_e32 v6, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_f32_e32 vcc_lo, 0x3f2aaaab, v5
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v5, null, 0, v6, vcc_lo
	v_add_f32_e32 v6, -1.0, v4
	v_cmp_neq_f32_e32 vcc_lo, 0x7f800000, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v7, 0, v5
	v_cvt_f32_i32_e32 v5, v5
	v_sub_f32_e32 v8, v6, v4
	v_sub_f32_e32 v6, v3, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v4, v4, v7
	v_add_f32_e32 v8, 1.0, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, 1.0, v4
	v_add_f32_e32 v6, v6, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v6, v6, v7
	v_dual_add_f32 v7, -1.0, v4 :: v_dual_add_f32 v8, -1.0, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v10, 1.0, v7
	v_sub_f32_e32 v8, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v4, v4, v10
	v_add_f32_e32 v8, v6, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v6, v4
	v_add_f32_e32 v10, v9, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v11, v7, v4
	v_rcp_f32_e32 v6, v10
	v_sub_f32_e32 v9, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_sub_f32 v7, v7, v11 :: v_dual_add_f32 v8, v8, v9
	v_mul_f32_e32 v12, v11, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v4, v4, v7 :: v_dual_mul_f32 v13, v10, v12
	v_fma_f32 v9, v12, v10, -v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v12, v8
	v_add_f32_e32 v14, v13, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v15, v11, v14
	v_sub_f32_e32 v7, v14, v13
	v_sub_f32_e32 v11, v11, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v7, v7, v9
	v_sub_f32_e32 v11, v11, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v11
	v_add_f32_e32 v4, v7, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v15, v4
	v_dual_mul_f32 v9, v6, v7 :: v_dual_sub_f32 v14, v15, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v11, v10, v9 :: v_dual_add_f32 v4, v4, v14
	v_fma_f32 v10, v9, v10, -v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v10, v9, v8
	v_add_f32_e32 v8, v11, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v13, v7, v8
	v_sub_f32_e32 v11, v8, v11
	v_sub_f32_e32 v7, v7, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v7, v7, v8
	v_sub_f32_e32 v8, v11, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v7
	v_add_f32_e32 v7, v12, v9
	v_add_f32_e32 v4, v8, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v8, v7, v12
	v_add_f32_e32 v4, v13, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v8, v9, v8
	v_mul_f32_e32 v4, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v8, v4
	v_add_f32_e32 v6, v7, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v8, v6, v6
	v_dual_fmaak_f32 v9, s1, v8, 0x3ecc95a3 :: v_dual_mul_f32 v10, v6, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmaak_f32 v8, v8, v9, 0x3f2aaada :: v_dual_add_f32 v9, v6, v6
	v_sub_f32_e32 v6, v6, v7
	v_mul_f32_e32 v8, v10, v8
	v_mul_f32_e32 v10, 0x3f317218, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v4, v4, v6 :: v_dual_add_f32 v7, v9, v8
	v_add_f32_e32 v4, v4, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v6, v7, v9
	v_fma_f32 v9, 0x3f317218, v5, -v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v6, v8, v6 :: v_dual_fmamk_f32 v5, v5, 0xb102e308, v9
	v_add_f32_e32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v6, v10, v5
	v_add_f32_e32 v8, v7, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v10, v6, v10 :: v_dual_add_f32 v9, v6, v8
	v_sub_f32_e32 v7, v8, v7
	v_sub_f32_e32 v5, v5, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v11, v9, v6 :: v_dual_sub_f32 v4, v4, v7
	v_dual_sub_f32 v12, v9, v11 :: v_dual_sub_f32 v7, v8, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v8, v5, v4
	v_sub_f32_e32 v6, v6, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v6, v7, v6 :: v_dual_sub_f32 v7, v8, v5
	v_add_f32_e32 v6, v8, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v8, v8, v7
	v_sub_f32_e32 v4, v4, v7
	v_add_f32_e32 v10, v9, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v5, v5, v8
	v_sub_f32_e32 v7, v10, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v4, v4, v5 :: v_dual_sub_f32 v5, v6, v7
	v_add_f32_e32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v10, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, 0x33800000, |v3|
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v3, v4, v3, vcc_lo
.LBB2_3:
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v0, 16, v0
	s_wait_loadcnt 0x0
	v_cmp_ngt_f32_e64 s0, 0xc2ce8ed0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v4, 0xbfb8aa3b, v0
	v_rndne_f32_e32 v5, v4
	v_fma_f32 v6, 0xbfb8aa3b, v0, -v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v4, v4, v5
	v_fmamk_f32 v6, v0, 0xb2a5705f, v6
	v_cvt_i32_f32_e32 v5, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ldexp_f32 v4, v4, v5
	v_mul_f32_e32 v5, 0x3fb8aa3b, v2
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v0
	v_rndne_f32_e32 v7, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v0
	v_fma_f32 v8, 0x3fb8aa3b, v2, -v5
	v_sub_f32_e32 v5, v5, v7
	v_cvt_i32_f32_e32 v7, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0x7f800000, v4, vcc_lo
	v_fmamk_f32 v8, v2, 0x32a5705f, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v0, 1.0, v0
	v_add_f32_e32 v5, v5, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v4, null, v0, v0, 1.0
	v_div_scale_f32 v8, vcc_lo, 1.0, v0, 1.0
	v_exp_f32_e32 v5, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v6, v4
	v_ldexp_f32 v5, v5, v7
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v9, -v4, v6, 1.0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v5, 0, v5, s0
	v_cmp_nlt_f32_e64 s0, 0x42b17218, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v6, v9, v6
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, 0x7f800000, v5, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v9, v8, v6
	v_mul_f32_e64 v2, v3, -v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v10, -v4, v9, v8
	v_fmac_f32_e32 v9, v10, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v4, v9, v8
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v4, v4, v6, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v0, v4, v0, 1.0
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_store_b32 v1, v2, s[4:5]
	global_store_b32 v1, v0, s[6:7]
.LBB2_4:
	s_endpgm
.Lfunc_end2:
	.size	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_, .Lfunc_end2-_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
		.amdhsa_group_segment_fixed_size 0
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 16
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_,"axG",@progbits,_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.num_vgpr, 16
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.numbered_sgpr, 12
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1292
; TotalNumSgprs: 14
; NumVgprs: 16
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 16
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_,"axG",@progbits,_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_,comdat
	.globl	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_ ; -- Begin function _ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.p2align	8
	.type	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_,@function
_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_: ; @_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s22, s[0:1], 0x54
	s_load_b512 s[4:19], s[0:1], 0x0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v1, v0
	s_mov_b32 s2, ttmp9
	s_mov_b32 s3, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_u64 s[20:21], s[2:3], 0x1400
	s_wait_kmcnt 0x0
	s_and_b32 s23, s22, 0xffff
	s_mov_b32 s22, s3
.LBB3_1:                                ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v5, s24, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s21, 0, s24
	v_lshlrev_b64_e32 v[7:8], 1, v[1:2]
	v_add_nc_u32_e32 v1, s23, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 1, v[5:6]
	v_add_co_u32 v7, vcc_lo, s4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v8, null, s5, v8, vcc_lo
	v_add_co_u32 v9, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s7, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_u16 v7, v[7:8], off
	global_load_u16 v8, v[9:10], off
	global_load_u16 v5, v[5:6], off
	v_cmp_lt_u32_e32 vcc_lo, 0x13ff, v1
	s_or_b32 s22, vcc_lo, s22
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v6, 16, v7
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 16, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v4, v6, v5 :: v_dual_lshlrev_b32 v7, 16, v8
	v_fmac_f32_e32 v3, v6, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB3_1
; %bb.2:
	s_or_b32 exec_lo, exec_lo, s22
	v_lshlrev_b32_e32 v1, 2, v0
	s_mov_b32 s4, exec_lo
	ds_store_2addr_stride64_b32 v1, v3, v4 offset1:4
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 0x80, v0
	s_cbranch_execz .LBB3_4
; %bb.3:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:2
	ds_load_2addr_stride64_b32 v[4:5], v1 offset0:4 offset1:6
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB3_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 64, v0
	s_cbranch_execz .LBB3_6
; %bb.5:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:1
	ds_load_2addr_stride64_b32 v[4:5], v1 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB3_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB3_8
; %bb.7:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:32
	ds_load_2addr_b32 v[4:5], v4 offset1:32
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB3_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB3_10
; %bb.9:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:16
	ds_load_2addr_b32 v[4:5], v4 offset1:16
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB3_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 8, v0
	s_cbranch_execz .LBB3_12
; %bb.11:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:8
	ds_load_2addr_b32 v[4:5], v4 offset1:8
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB3_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 4, v0
	s_cbranch_execz .LBB3_14
; %bb.13:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:4
	ds_load_2addr_b32 v[4:5], v4 offset1:4
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB3_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_u32_e32 2, v0
	s_cbranch_execz .LBB3_16
; %bb.15:
	v_add_nc_u32_e32 v4, 0x400, v1
	ds_load_2addr_b32 v[2:3], v1 offset1:2
	ds_load_2addr_b32 v[4:5], v4 offset1:2
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v3, v2 :: v_dual_add_f32 v3, v5, v4
	ds_store_2addr_stride64_b32 v1, v2, v3 offset1:4
.LBB3_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB3_18
; %bb.17:
	v_add_nc_u32_e64 v0, 4, 0
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:4
	ds_load_2addr_stride64_b32 v[4:5], v0 offset1:4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v4, v2
	v_add_f32_e32 v2, v5, v3
	ds_store_2addr_stride64_b32 v1, v0, v2 offset1:4
.LBB3_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB3_30
; %bb.19:
	v_mov_b32_e32 v0, 0
	ds_load_b32 v0, v0
	s_wait_dscnt 0x0
	v_and_b32_e32 v1, 0x7f800000, v0
	v_readfirstlane_b32 s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v1
	s_cbranch_vccnz .LBB3_21
; %bb.20:
	s_wait_alu depctr_sa_sdst(0)
	s_bfe_u32 s5, s4, 0x10010
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s5, s4, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s5, 0x7fff
	s_cbranch_execz .LBB3_22
	s_branch .LBB3_23
.LBB3_21:
                                        ; implicit-def: $sgpr8
.LBB3_22:
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s4, 0xffff
	s_or_b32 s6, s4, 0x10000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s8, s4, s6
.LBB3_23:
	v_mov_b32_e32 v0, 0
	ds_load_b32 v0, v0 offset:1024
	s_wait_dscnt 0x0
	v_and_b32_e32 v1, 0x7f800000, v0
	v_readfirstlane_b32 s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v1
	s_cbranch_vccnz .LBB3_25
; %bb.24:
	s_bfe_u32 s5, s4, 0x10010
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s5, s4, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s5, 0x7fff
	s_cbranch_execz .LBB3_26
	s_branch .LBB3_27
.LBB3_25:
                                        ; implicit-def: $sgpr6
.LBB3_26:
	s_and_b32 s5, s4, 0xffff
	s_or_b32 s6, s4, 0x10000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s6, s4, s6
.LBB3_27:
	s_lshl_b64 s[4:5], s[2:3], 2
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s8
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[12:13], s[4:5]
	s_add_nc_u64 s[10:11], s[10:11], s[4:5]
	s_load_b32 s9, s[12:13], 0x0
	s_load_b32 s7, s[10:11], 0x0
	s_and_b32 s12, s8, 0xffff0000
	s_lshl_b64 s[10:11], s[2:3], 1
	v_mov_b32_e32 v2, s6
	s_wait_kmcnt 0x0
	s_add_f32 s3, s9, s12
	s_add_nc_u64 s[8:9], s[14:15], s[10:11]
	s_add_nc_u64 s[10:11], s[16:17], s[10:11]
	s_clause 0x1
	global_store_d16_hi_b16 v0, v1, s[8:9]
	global_store_d16_hi_b16 v0, v2, s[10:11]
	s_cmp_gt_f32 s3, 0x41a00000
	s_cbranch_scc1 .LBB3_29
; %bb.28:
	s_mul_f32 s2, s3, 0x3fb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3)
	s_xor_b32 s8, s2, 0x80000000
	s_rndne_f32 s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s8, s3, 0x3fb8aa3b, s8
	s_cmp_nlt_f32 s3, 0xc2ce8ed0
	s_sub_f32 s2, s2, s9
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s8, s3, 0x32a5705f, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s2, s2, s8
	s_cvt_i32_f32 s8, s9
	v_s_exp_f32 s2, s2
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_1)
	v_ldexp_f32 v1, s2, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s2, v1
	s_cselect_b32 s2, s2, 0
	s_cmp_ngt_f32 s3, 0x42b17218
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s2, s2, 0x7f800000
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s2, 1.0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_frexp_mant_f32_e32 v1, s3
	v_frexp_exp_i32_f32_e32 v2, s3
	v_readfirstlane_b32 s8, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_readfirstlane_b32 s9, v2
	s_cmp_lt_f32 s8, 0x3f2aaaab
	s_add_f32 s8, s3, -1.0
	s_sub_co_ci_u32 s9, s9, 0
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s11, s8, s3
	s_sub_co_i32 s10, 0, s9
	s_cvt_f32_i32 s9, s9
	v_ldexp_f32 v1, s3, s10
	s_sub_f32 s3, s2, s8
	s_add_f32 s8, s11, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_readfirstlane_b32 s11, v1
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s3, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_ldexp_f32 v1, s3, s10
	s_add_f32 s8, s11, 1.0
	s_add_f32 s12, s11, -1.0
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s8, -1.0
	v_readfirstlane_b32 s10, v1
	s_add_f32 s13, s12, 1.0
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s3, s11, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_sub_f32 s11, s11, s13
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_add_f32 s10, s10, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s13, s8, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_2)
	s_add_f32 s14, s12, s10
	v_s_rcp_f32 s11, s13
	s_sub_f32 s8, s8, s13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_sub_f32 s12, s12, s14
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s3, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	s_add_f32 s10, s10, s12
	s_mul_f32 s15, s14, s11
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s16, s13, s15
	s_xor_b32 s17, s16, 0x80000000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_fmac_f32 s17, s15, s13
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s17, s15, s3
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s8, s16, s17
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s20, s14, s8
	s_sub_f32 s12, s8, s16
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_f32 s14, s14, s20
	s_sub_f32 s12, s12, s17
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s8, s14, s8
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s8, s10, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s8, s12, s8
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s10, s20, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_mul_f32 s12, s11, s10
	s_sub_f32 s17, s20, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s14, s13, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_2)
	s_add_f32 s8, s8, s17
	s_xor_b32 s16, s14, 0x80000000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_fmac_f32 s16, s12, s13
	s_fmac_f32 s16, s12, s3
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s3, s14, s16
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s13, s10, s3
	s_sub_f32 s14, s3, s14
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s10, s10, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s3, s10, s3
	s_sub_f32 s10, s14, s16
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_f32 s3, s8, s3
	s_add_f32 s8, s15, s12
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_sub_f32 s10, s8, s15
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s13, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_2)
	s_sub_f32 s10, s12, s10
	s_mov_b32 s12, 0x3e9b6dac
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s3, s11, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s10, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s11, s10, s10
	s_sub_f32 s8, s10, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1)
	s_fmaak_f32 s12, s11, s12, 0x3ecc95a3
	s_mul_f32 s13, s10, s11
	s_sub_f32 s3, s3, s8
	s_wait_alu depctr_sa_sdst(0)
	s_fmaak_f32 s11, s11, s12, 0x3f2aaada
	s_add_f32 s12, s10, s10
	s_add_f32 s3, s3, s3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s11, s13, s11
	s_mul_f32 s13, s9, 0x3f317218
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_f32 s10, s12, s11
	s_xor_b32 s14, s13, 0x80000000
	s_cmp_neq_f32 s2, 0x7f800000
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s8, s10, s12
	s_fmamk_f32 s12, s9, 0x3f317218, s14
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_f32 s8, s11, s8
	s_fmamk_f32 s9, s9, 0xb102e308, s12
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_f32 s3, s3, s8
	s_add_f32 s8, s13, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_f32 s11, s10, s3
	s_sub_f32 s13, s8, s13
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_f32 s12, s8, s11
	s_sub_f32 s10, s11, s10
	s_sub_f32 s9, s9, s13
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s14, s12, s8
	s_sub_f32 s3, s3, s10
	s_delay_alu instid0(SALU_CYCLE_2)
	s_sub_f32 s15, s12, s14
	s_sub_f32 s10, s11, s14
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s11, s9, s3
	s_sub_f32 s8, s8, s15
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_f32 s8, s10, s8
	s_sub_f32 s10, s11, s9
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s8, s11, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_f32 s11, s11, s10
	s_sub_f32 s3, s3, s10
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s13, s12, s8
	s_sub_f32 s9, s9, s11
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_f32 s10, s13, s12
	s_add_f32 s3, s3, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_sub_f32 s8, s8, s10
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s3, s3, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_add_f32 s3, s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s3, s3, 0x7f800000
	s_and_b32 s8, s2, 0x7fffffff
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_f32 s8, 0x33800000
	s_cselect_b32 s3, s2, s3
.LBB3_29:
	s_mul_f32 s2, s7, 0x3fb8aa3b
	s_load_b64 s[0:1], s[0:1], 0x40
	s_delay_alu instid0(SALU_CYCLE_2)
	s_xor_b32 s8, s2, 0x80000000
	s_rndne_f32 s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s8, s7, 0x3fb8aa3b, s8
	s_cmp_nlt_f32 s7, 0xc2ce8ed0
	s_sub_f32 s2, s2, s9
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s8, s7, 0x32a5705f, s8
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s7, 0x42b17218
	s_wait_alu depctr_sa_sdst(0)
	s_add_f32 s2, s2, s8
	s_cvt_i32_f32 s8, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_s_exp_f32 s2, s2
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	v_ldexp_f32 v1, s2, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_and_b32 s6, s6, 0xffff0000
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s2, s6, 0xbfb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_xor_b32 s7, s2, 0x80000000
	s_rndne_f32 s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s7, s6, 0xbfb8aa3b, s7
	s_cmp_ngt_f32 s6, 0x42ce8ed0
	s_sub_f32 s2, s2, s8
	s_wait_alu depctr_sa_sdst(0)
	s_fmamk_f32 s7, s6, 0xb2a5705f, s7
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_f32 s2, s2, s7
	s_cvt_i32_f32 s7, s8
	s_wait_alu depctr_sa_sdst(0)
	v_s_exp_f32 s2, s2
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_ldexp_f32 v2, s2, s7
	s_cselect_b32 s2, -1, 0
	s_cmp_nlt_f32 s6, 0xc2b17218
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v2, 0, v2, s2
	s_cselect_b32 s2, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v2, 0x7f800000, v2, s2
	v_dual_add_f32 v2, 1.0, v2 :: v_dual_cndmask_b32 v1, 0x7f800000, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v3, null, v2, v2, 1.0
	v_xor_b32_e32 v1, 0x80000000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v4, v3
	v_xor_b32_e32 v3, 0x80000000, v3
	v_mul_f32_e32 v1, s3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v5, v3, v4, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v5, v4
	v_div_scale_f32 v5, s2, 1.0, v2, 1.0
	s_mov_b32 vcc_lo, s2
	s_add_nc_u64 s[2:3], s[18:19], s[4:5]
	v_mul_f32_e32 v6, v5, v4
	v_fma_f32 v7, v3, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v7, v4
	v_fmac_f32_e32 v5, v3, v6
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v5, v4, v6
	v_div_fixup_f32 v2, v3, v2, 1.0
	s_clause 0x1
	global_store_b32 v0, v1, s[2:3]
	global_store_b32 v0, v2, s[0:1]
.LBB3_30:
	s_endpgm
.Lfunc_end3:
	.size	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_, .Lfunc_end3-_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 328
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
		.amdhsa_next_free_sgpr 25
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_,"axG",@progbits,_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.num_vgpr, 11
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.numbered_sgpr, 25
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2812
; TotalNumSgprs: 27
; NumVgprs: 11
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2048 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 27
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
	.section	.text._ZN12_GLOBAL__N_115eviction_kernelEPKjPjm,"axG",@progbits,_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm,comdat
	.globl	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm ; -- Begin function _ZN12_GLOBAL__N_115eviction_kernelEPKjPjm
	.p2align	8
	.type	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm,@function
_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm: ; @_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b32 s8, s[0:1], 0x24
	s_load_b64 s[2:3], s[0:1], 0x10
	s_load_b128 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_mov_b32 s12, exec_lo
	s_wait_kmcnt 0x0
	s_and_b32 s8, s8, 0xffff
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_mad_co_u64_u32 v[2:3], null, s8, ttmp9, v[0:1]
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u64_e64 s[2:3], v[2:3]
	s_cbranch_execz .LBB4_4
; %bb.1:
	s_add_nc_u64 s[0:1], s[0:1], 24
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]
	s_load_b32 s0, s[0:1], 0x0
	s_mov_b32 s9, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s1, s9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, s4, v0
	v_add_co_ci_u32_e64 v5, null, s5, v1, vcc_lo
	v_mov_b32_e32 v1, 0
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[10:11], s[0:1], s[8:9]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[4:5], s[10:11], 2
.LBB4_2:                                ; =>This Inner Loop Header: Depth=1
	global_load_b32 v0, v[4:5], off
	v_add_co_u32 v2, vcc_lo, v2, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s11, v3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, s0, v4, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s5, v5, s0
	v_cmp_le_u64_e32 vcc_lo, s[2:3], v[2:3]
	s_or_b32 s9, vcc_lo, s9
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v1, v0, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s9
	s_cbranch_execnz .LBB4_2
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s9
.LBB4_4:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s12
	s_mov_b32 s0, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ne_u32_e32 0, v1
	s_cbranch_execz .LBB4_9
; %bb.5:
	s_mov_b32 s1, exec_lo
.LBB4_6:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_ctz_i32_b32 s2, s1
	s_wait_alu depctr_sa_sdst(0)
	v_readlane_b32 s3, v1, s2
	s_lshl_b32 s2, 1, s2
	s_xor_b32 s0, s0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 s1, s1, s2
	s_cbranch_scc1 .LBB4_6
; %bb.7:
	v_mbcnt_lo_u32_b32 v0, exec_lo, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB4_9
; %bb.8:
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0
	global_atomic_xor_b32 v0, v1, s[6:7] scope:SCOPE_DEV
.LBB4_9:
	s_endpgm
.Lfunc_end4:
	.size	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm, .Lfunc_end4-_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_115eviction_kernelEPKjPjm
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
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 13
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_115eviction_kernelEPKjPjm,"axG",@progbits,_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm,comdat
                                        ; -- End function
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.num_vgpr, 6
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.num_agpr, 0
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.numbered_sgpr, 13
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.num_named_barrier, 0
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.private_seg_size, 0
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.uses_vcc, 1
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.uses_flat_scratch, 0
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.has_dyn_sized_stack, 0
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.has_recursion, 0
	.set .L_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 360
; TotalNumSgprs: 15
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 15
; NumVGPRsForWavesPerEU: 6
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
	.type	__hip_cuid_57a29556a1d631fd,@object ; @__hip_cuid_57a29556a1d631fd
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_57a29556a1d631fd
__hip_cuid_57a29556a1d631fd:
	.byte	0                               ; 0x0
	.size	__hip_cuid_57a29556a1d631fd, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_57a29556a1d631fd
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
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
    .private_segment_fixed_size: 0
    .sgpr_count:     13
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
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
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
    .private_segment_fixed_size: 0
    .sgpr_count:     19
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     16
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
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         76
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         80
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         84
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         86
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         88
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         90
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         92
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         94
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         136
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 2048
    .kernarg_segment_align: 8
    .kernarg_segment_size: 328
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
    .private_segment_fixed_size: 0
    .sgpr_count:     27
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_.kd
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
    .name:           _ZN12_GLOBAL__N_115eviction_kernelEPKjPjm
    .private_segment_fixed_size: 0
    .sgpr_count:     15
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_115eviction_kernelEPKjPjm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
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
	.file	"bf16_gdn_control_t1_qual.hip"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI0_0:
	.byte	45                              # 0x2d
	.byte	45                              # 0x2d
	.byte	115                             # 0x73
	.byte	116                             # 0x74
	.byte	97                              # 0x61
	.byte	116                             # 0x74
	.byte	105                             # 0x69
	.byte	99                              # 0x63
	.byte	45                              # 0x2d
	.byte	114                             # 0x72
	.byte	101                             # 0x65
	.byte	99                              # 0x63
	.byte	101                             # 0x65
	.byte	105                             # 0x69
	.byte	112                             # 0x70
	.byte	116                             # 0x74
.LCPI0_1:
	.byte	101                             # 0x65
	.byte	111                             # 0x6f
	.byte	110                             # 0x6e
	.byte	32                              # 0x20
	.byte	65                              # 0x41
	.byte	73                              # 0x49
	.byte	32                              # 0x20
	.byte	80                              # 0x50
	.byte	82                              # 0x52
	.byte	79                              # 0x4f
	.byte	32                              # 0x20
	.byte	82                              # 0x52
	.byte	57                              # 0x39
	.byte	55                              # 0x37
	.byte	48                              # 0x30
	.byte	48                              # 0x30
.LCPI0_2:
	.byte	65                              # 0x41
	.byte	77                              # 0x4d
	.byte	68                              # 0x44
	.byte	32                              # 0x20
	.byte	82                              # 0x52
	.byte	97                              # 0x61
	.byte	100                             # 0x64
	.byte	101                             # 0x65
	.byte	111                             # 0x6f
	.byte	110                             # 0x6e
	.byte	32                              # 0x20
	.byte	65                              # 0x41
	.byte	73                              # 0x49
	.byte	32                              # 0x20
	.byte	80                              # 0x50
	.byte	82                              # 0x52
.LCPI0_5:
	.long	0xc0200000                      # float -2.5
	.long	0xc01f0000                      # float -2.484375
	.long	0xc01e0000                      # float -2.46875
	.long	0xc01d0000                      # float -2.453125
.LCPI0_6:
	.long	0xbf400000                      # float -0.75
	.long	0xbf3ccccd                      # float -0.737500011
	.long	0xbf39999a                      # float -0.725000024
	.long	0xbf366666                      # float -0.712499976
.LCPI0_7:
	.long	0xc01c0000                      # float -2.4375
	.long	0xc01b0000                      # float -2.421875
	.long	0xc01a0000                      # float -2.40625
	.long	0xc0190000                      # float -2.390625
.LCPI0_8:
	.long	0xbf333333                      # float -0.699999988
	.long	0xbf300000                      # float -0.6875
	.long	0xbf2ccccd                      # float -0.675000011
	.long	0xbf29999a                      # float -0.662500024
.LCPI0_9:
	.long	0xc0180000                      # float -2.375
	.long	0xc0170000                      # float -2.359375
	.long	0xc0160000                      # float -2.34375
	.long	0xc0150000                      # float -2.328125
.LCPI0_10:
	.long	0xbf266666                      # float -0.649999976
	.long	0xbf233333                      # float -0.637499988
	.long	0xbf200000                      # float -0.625
	.long	0xbf1ccccd                      # float -0.612500011
.LCPI0_11:
	.long	0xc0140000                      # float -2.3125
	.long	0xc0130000                      # float -2.296875
	.long	0xc0120000                      # float -2.28125
	.long	0xc0110000                      # float -2.265625
.LCPI0_12:
	.long	0xbf19999a                      # float -0.600000024
	.long	0xbf166666                      # float -0.587499976
	.long	0xbf133333                      # float -0.574999988
	.long	0xbf100000                      # float -0.5625
.LCPI0_13:
	.long	0xc0100000                      # float -2.25
	.long	0xc00f0000                      # float -2.234375
	.long	0xc00e0000                      # float -2.21875
	.long	0xc00d0000                      # float -2.203125
.LCPI0_14:
	.long	0xbf0ccccd                      # float -0.550000012
	.long	0xbf09999a                      # float -0.537500024
	.long	0xbf066666                      # float -0.524999976
	.long	0xbf033333                      # float -0.512499988
.LCPI0_15:
	.long	0xc00c0000                      # float -2.1875
	.long	0xc00b0000                      # float -2.171875
	.long	0xc00a0000                      # float -2.15625
	.long	0xc0090000                      # float -2.140625
.LCPI0_16:
	.long	0xbf000000                      # float -0.5
	.long	0xbef9999a                      # float -0.487500012
	.long	0xbef33333                      # float -0.474999994
	.long	0xbeeccccd                      # float -0.462500006
.LCPI0_17:
	.long	0xc0080000                      # float -2.125
	.long	0xc0070000                      # float -2.109375
	.long	0xc0060000                      # float -2.09375
	.long	0xc0050000                      # float -2.078125
.LCPI0_18:
	.long	0xbee66666                      # float -0.449999988
	.long	0xbee00000                      # float -0.4375
	.long	0xbed9999a                      # float -0.425000012
	.long	0xbed33333                      # float -0.412499994
.LCPI0_19:
	.long	0xc0040000                      # float -2.0625
	.long	0xc0030000                      # float -2.046875
	.long	0xc0020000                      # float -2.03125
	.long	0xc0010000                      # float -2.015625
.LCPI0_20:
	.long	0xbecccccd                      # float -0.400000006
	.long	0xbec66666                      # float -0.387499988
	.long	0xbec00000                      # float -0.375
	.long	0xbeb9999a                      # float -0.362500012
.LCPI0_21:
	.long	0xc0000000                      # float -2
	.long	0xbffe0000                      # float -1.984375
	.long	0xbffc0000                      # float -1.96875
	.long	0xbffa0000                      # float -1.953125
.LCPI0_22:
	.long	0xbeb33333                      # float -0.349999994
	.long	0xbeaccccd                      # float -0.337500006
	.long	0xbea66666                      # float -0.324999988
	.long	0xbea00000                      # float -0.3125
.LCPI0_23:
	.long	0xbff80000                      # float -1.9375
	.long	0xbff60000                      # float -1.921875
	.long	0xbff40000                      # float -1.90625
	.long	0xbff20000                      # float -1.890625
.LCPI0_24:
	.long	0xbe99999a                      # float -0.300000012
	.long	0xbe933333                      # float -0.287499994
	.long	0xbe8ccccd                      # float -0.275000006
	.long	0xbe866666                      # float -0.262499988
.LCPI0_25:
	.long	0xbff00000                      # float -1.875
	.long	0xbfee0000                      # float -1.859375
	.long	0xbfec0000                      # float -1.84375
	.long	0xbfea0000                      # float -1.828125
.LCPI0_26:
	.long	0xbe800000                      # float -0.25
	.long	0xbe733334                      # float -0.237500012
	.long	0xbe666668                      # float -0.225000024
	.long	0xbe599998                      # float -0.212499976
.LCPI0_27:
	.long	0xbfe80000                      # float -1.8125
	.long	0xbfe60000                      # float -1.796875
	.long	0xbfe40000                      # float -1.78125
	.long	0xbfe20000                      # float -1.765625
.LCPI0_28:
	.long	0xbe4ccccc                      # float -0.199999988
	.long	0xbe400000                      # float -0.1875
	.long	0xbe333334                      # float -0.175000012
	.long	0xbe266668                      # float -0.162500024
.LCPI0_30:
	.quad	0x8000000000000000              # double -0
	.quad	0x8000000000000000              # double -0
.LCPI0_32:
	.quad	0x7fffffffffffffff              # double NaN
	.quad	0x7fffffffffffffff              # double NaN
.LCPI0_35:
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_3:
	.long	0x3a800000                      # float 9.765625E-4
.LCPI0_4:
	.long	0x38800000                      # float 6.10351563E-5
.LCPI0_29:
	.long	0x41a00000                      # float 20
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI0_31:
	.quad	0x3ff0000000000000              # double 1
.LCPI0_33:
	.quad	0x3fb0000000000000              # double 0.0625
.LCPI0_34:
	.quad	0x3f33a92a30553261              # double 2.9999999999999997E-4
	.text
	.globl	main
	.prefalign	4, .Lfunc_end0, nop
	.type	main,@function
main:                                   # @main
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
	subq	$3112, %rsp                     # imm = 0xC28
	.cfi_def_cfa_offset 3168
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %r12
	cmpl	$7, %edi
	jne	.LBB0_45
# %bb.1:
	movq	8(%r12), %rbx
	leaq	1656(%rsp), %r13
	movq	%r13, 1640(%rsp)
	testq	%rbx, %rbx
	je	.LBB0_532
# %bb.2:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	movq	%r13, %r15
	cmpq	$16, %rax
	jb	.LBB0_7
# %bb.3:
	testq	%r14, %r14
	js	.LBB0_537
# %bb.4:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB0_501
# %bb.5:
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.6:
	movq	%rax, %r15
	movq	%rax, 1640(%rsp)
	movq	%r14, 1656(%rsp)
.LBB0_7:
	testq	%r14, %r14
	je	.LBB0_11
# %bb.8:
	cmpq	$1, %r14
	jne	.LBB0_10
# %bb.9:
	movzbl	(%rbx), %eax
	movb	%al, (%r15)
	jmp	.LBB0_11
.LBB0_10:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB0_11:
	movq	%r14, 1648(%rsp)
	movb	$0, (%r15,%r14)
	movq	1640(%rsp), %rdi
	cmpq	$8, 1648(%rsp)
	movb	$1, %r14b
	movq	%r12, 328(%rsp)                 # 8-byte Spill
	jne	.LBB0_42
# %bb.12:
	movabsq	$8391737126163524909, %rax      # imm = 0x74757074756F2D2D
	cmpq	%rax, (%rdi)
	jne	.LBB0_42
# %bb.13:
	movq	24(%r12), %rbx
	leaq	384(%rsp), %rbp
	movq	%rbp, 368(%rsp)
	testq	%rbx, %rbx
	je	.LBB0_575
# %bb.14:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	movq	%rbp, %r15
	cmpq	$16, %rax
	jb	.LBB0_19
# %bb.15:
	testq	%r14, %r14
	js	.LBB0_577
# %bb.16:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB0_517
# %bb.17:
.Ltmp2:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp3:                                 # EH_LABEL
# %bb.18:
	movq	%rax, %r15
	movq	%rax, 368(%rsp)
	movq	%r14, 384(%rsp)
.LBB0_19:
	testq	%r14, %r14
	je	.LBB0_23
# %bb.20:
	cmpq	$1, %r14
	jne	.LBB0_22
# %bb.21:
	movzbl	(%rbx), %eax
	movb	%al, (%r15)
	jmp	.LBB0_23
.LBB0_22:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB0_23:
	movq	%r14, 376(%rsp)
	movb	$0, (%r15,%r14)
	movq	368(%rsp), %rbx
	cmpq	$10, 376(%rsp)
	movb	$1, %r14b
	jne	.LBB0_39
# %bb.24:
	movabsq	$7092436534709792045, %rax      # imm = 0x626D657373612D2D
	xorq	(%rbx), %rax
	movzwl	8(%rbx), %ecx
	xorq	$31084, %rcx                    # imm = 0x796C
	orq	%rax, %rcx
	jne	.LBB0_39
# %bb.25:
	movq	328(%rsp), %rax                 # 8-byte Reload
	movq	40(%rax), %r14
	leaq	80(%rsp), %r12
	movq	%r12, 64(%rsp)
	testq	%r14, %r14
	je	.LBB0_581
# %bb.26:
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	strlen@PLT
	movq	%rax, %r15
	cmpq	$16, %rax
	jb	.LBB0_31
# %bb.27:
	testq	%r15, %r15
	js	.LBB0_583
# %bb.28:
	movq	%r15, %rdi
	incq	%rdi
	js	.LBB0_521
# %bb.29:
.Ltmp4:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp5:                                 # EH_LABEL
# %bb.30:
	movq	%rax, %r12
	movq	%rax, 64(%rsp)
	movq	%r15, 80(%rsp)
.LBB0_31:
	testq	%r15, %r15
	je	.LBB0_35
# %bb.32:
	cmpq	$1, %r15
	jne	.LBB0_34
# %bb.33:
	movzbl	(%r14), %eax
	movb	%al, (%r12)
	jmp	.LBB0_35
.LBB0_34:
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%r14, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB0_35:
	movq	%r15, 72(%rsp)
	movb	$0, (%r12,%r15)
	movq	64(%rsp), %rdi
	cmpq	$16, 72(%rsp)
	movb	$1, %r14b
	jne	.LBB0_37
# %bb.36:
	movdqu	(%rdi), %xmm0
	pcmpeqb	.LCPI0_0(%rip), %xmm0
	pmovmskb	%xmm0, %eax
	cmpl	$65535, %eax                    # imm = 0xFFFF
	setne	%r14b
.LBB0_37:
	leaq	80(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_39
# %bb.38:
	movq	80(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	368(%rsp), %rbx
.LBB0_39:
	cmpq	%rbp, %rbx
	je	.LBB0_41
# %bb.40:
	movq	384(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
.LBB0_41:
	movq	1640(%rsp), %rdi
	movq	328(%rsp), %r12                 # 8-byte Reload
.LBB0_42:
	cmpq	%r13, %rdi
	je	.LBB0_44
# %bb.43:
	movq	1656(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_44:
	testb	%r14b, %r14b
	je	.LBB0_50
.LBB0_45:
.Ltmp587:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cerr@GOTPCREL(%rip), %rbx
	leaq	.L.str.3(%rip), %rsi
	movl	$7, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp588:                               # EH_LABEL
# %bb.46:
	movq	(%r12), %r14
	testq	%r14, %r14
	je	.LBB0_48
# %bb.47:
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	strlen@PLT
.Ltmp589:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cerr@GOTPCREL(%rip), %rdi
	movq	%r14, %rsi
	movq	%rax, %rdx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp590:                               # EH_LABEL
	jmp	.LBB0_49
.LBB0_48:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	leaq	(%rbx,%rax), %rdi
	movl	32(%rbx,%rax), %esi
	orl	$1, %esi
.Ltmp591:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
.Ltmp592:                               # EH_LABEL
.LBB0_49:
	movl	$2, %ebx
.Ltmp593:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cerr@GOTPCREL(%rip), %rdi
	leaq	.L.str.4(%rip), %rsi
	movl	$66, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp594:                               # EH_LABEL
	jmp	.LBB0_498
.LBB0_50:
	movl	$0, 324(%rsp)
.Ltmp6:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	324(%rsp), %rdi
	callq	hipGetDeviceCount@PLT
.Ltmp7:                                 # EH_LABEL
# %bb.51:
.Ltmp8:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rsi
	movl	%eax, %edi
	movl	$611, %edx                      # imm = 0x263
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp9:                                 # EH_LABEL
# %bb.52:
	cmpl	$0, 324(%rsp)
	jle	.LBB0_539
# %bb.53:
.Ltmp10:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%edi, %edi
	callq	hipSetDevice@PLT
.Ltmp11:                                # EH_LABEL
# %bb.54:
.Ltmp12:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.8(%rip), %rsi
	movl	%eax, %edi
	movl	$613, %edx                      # imm = 0x265
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp13:                                # EH_LABEL
# %bb.55:
	.cfi_escape 0x2e, 0x00
	leaq	1640(%rsp), %rbx
	movl	$1472, %edx                     # imm = 0x5C0
	movq	%rbx, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
.Ltmp15:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	xorl	%esi, %esi
	callq	hipGetDevicePropertiesR0600@PLT
.Ltmp16:                                # EH_LABEL
# %bb.56:
.Ltmp17:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.9(%rip), %rsi
	movl	%eax, %edi
	movl	$615, %edx                      # imm = 0x267
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp18:                                # EH_LABEL
# %bb.57:
	leaq	384(%rsp), %r13
	movq	%r13, 368(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	1640(%rsp), %rdi
	callq	strlen@PLT
	movq	%rax, %rbx
	movq	%r13, %r14
	cmpq	$16, %rax
	jb	.LBB0_62
# %bb.58:
	testq	%rbx, %rbx
	js	.LBB0_565
# %bb.59:
	movq	%rbx, %rdi
	incq	%rdi
	js	.LBB0_507
# %bb.60:
.Ltmp20:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp21:                                # EH_LABEL
# %bb.61:
	movq	%rax, %r14
	movq	%rax, 368(%rsp)
	movq	%rbx, 384(%rsp)
.LBB0_62:
	testq	%rbx, %rbx
	je	.LBB0_66
# %bb.63:
	cmpq	$1, %rbx
	jne	.LBB0_65
# %bb.64:
	movzbl	1640(%rsp), %eax
	movb	%al, (%r14)
	jmp	.LBB0_66
.LBB0_65:
	.cfi_escape 0x2e, 0x00
	leaq	1640(%rsp), %rsi
	movq	%r14, %rdi
	movq	%rbx, %rdx
	callq	memcpy@PLT
.LBB0_66:
	movq	%rbx, 376(%rsp)
	movb	$0, (%r14,%rbx)
	movq	368(%rsp), %rbx
	cmpq	$23, 376(%rsp)
	movb	$1, %r14b
	jne	.LBB0_81
# %bb.67:
	movdqu	(%rbx), %xmm0
	movdqu	7(%rbx), %xmm1
	pcmpeqb	.LCPI0_1(%rip), %xmm1
	pcmpeqb	.LCPI0_2(%rip), %xmm0
	pand	%xmm1, %xmm0
	pmovmskb	%xmm0, %eax
	cmpl	$65535, %eax                    # imm = 0xFFFF
	jne	.LBB0_81
# %bb.68:
	leaq	2800(%rsp), %r15
	leaq	80(%rsp), %rbp
	movq	%rbp, 64(%rsp)
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	movq	%rbp, %r12
	cmpq	$16, %rax
	jb	.LBB0_73
# %bb.69:
	testq	%r14, %r14
	js	.LBB0_579
# %bb.70:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB0_519
# %bb.71:
.Ltmp22:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp23:                                # EH_LABEL
# %bb.72:
	movq	%rax, %r12
	movq	%rax, 64(%rsp)
	movq	%r14, 80(%rsp)
.LBB0_73:
	testq	%r14, %r14
	je	.LBB0_77
# %bb.74:
	cmpq	$1, %r14
	jne	.LBB0_76
# %bb.75:
	movzbl	2800(%rsp), %eax
	movb	%al, (%r12)
	jmp	.LBB0_77
.LBB0_76:
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB0_77:
	movq	%r14, 72(%rsp)
	movb	$0, (%r12,%r14)
	movq	64(%rsp), %rdi
	cmpq	$7, 72(%rsp)
	movb	$1, %r14b
	jb	.LBB0_79
# %bb.78:
	movl	$829974119, %eax                # imm = 0x31786667
	xorl	(%rdi), %eax
	movl	$825242161, %ecx                # imm = 0x31303231
	xorl	3(%rdi), %ecx
	orl	%eax, %ecx
	setne	%r14b
.LBB0_79:
	cmpq	%rbp, %rdi
	je	.LBB0_81
# %bb.80:
	movq	80(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	368(%rsp), %rbx
.LBB0_81:
	cmpq	%r13, %rbx
	je	.LBB0_83
# %bb.82:
	movq	384(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
.LBB0_83:
	testb	%r14b, %r14b
	jne	.LBB0_542
# %bb.84:
	pxor	%xmm0, %xmm0
	movdqa	%xmm0, 1472(%rsp)
	movdqa	%xmm0, 1456(%rsp)
.Ltmp24:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1456(%rsp), %rdi
	movl	$32, %esi
	xorl	%edx, %edx
	callq	hipDeviceGetPCIBusId@PLT
.Ltmp25:                                # EH_LABEL
# %bb.85:
.Ltmp26:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.13(%rip), %rsi
	movl	%eax, %edi
	movl	$621, %edx                      # imm = 0x26D
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp27:                                # EH_LABEL
# %bb.86:
	leaq	248(%rsp), %r14
	movq	%r14, 232(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	1456(%rsp), %rdi
	callq	strlen@PLT
	movq	%rax, %rbx
	cmpq	$16, %rax
	jb	.LBB0_91
# %bb.87:
	testq	%rbx, %rbx
	js	.LBB0_567
# %bb.88:
	movq	%rbx, %rdi
	incq	%rdi
	js	.LBB0_509
# %bb.89:
.Ltmp29:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp30:                                # EH_LABEL
# %bb.90:
	movq	%rax, %r14
	movq	%rax, 232(%rsp)
	movq	%rbx, 248(%rsp)
.LBB0_91:
	testq	%rbx, %rbx
	je	.LBB0_95
# %bb.92:
	cmpq	$1, %rbx
	jne	.LBB0_94
# %bb.93:
	movzbl	1456(%rsp), %eax
	movb	%al, (%r14)
	jmp	.LBB0_95
.LBB0_94:
	.cfi_escape 0x2e, 0x00
	leaq	1456(%rsp), %rsi
	movq	%r14, %rdi
	movq	%rbx, %rdx
	callq	memcpy@PLT
.LBB0_95:
	movq	%rbx, 240(%rsp)
	movb	$0, (%r14,%rbx)
	movq	240(%rsp), %r14
	cmpq	$7, %r14
	jne	.LBB0_112
# %bb.96:
	movq	232(%rsp), %rcx
.Ltmp31:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.14(%rip), %rsi
	leaq	368(%rsp), %rdi
	leaq	64(%rsp), %r9
	movl	$5, %edx
	movl	$7, %r8d
	callq	_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE
.Ltmp32:                                # EH_LABEL
# %bb.97:
	movq	232(%rsp), %rdi
	movq	368(%rsp), %rsi
	leaq	384(%rsp), %rbx
	leaq	248(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_101
# %bb.98:
	cmpq	%rbx, %rsi
	je	.LBB0_104
# %bb.99:
	movq	248(%rsp), %rax
	movq	%rsi, 232(%rsp)
	movdqu	376(%rsp), %xmm0
	movdqu	%xmm0, 240(%rsp)
	testq	%rdi, %rdi
	je	.LBB0_103
# %bb.100:
	movq	%rdi, 368(%rsp)
	movq	%rax, 384(%rsp)
	jmp	.LBB0_109
.LBB0_101:
	cmpq	%rbx, %rsi
	je	.LBB0_104
# %bb.102:
	movq	%rsi, 232(%rsp)
	movdqu	376(%rsp), %xmm0
	movdqu	%xmm0, 240(%rsp)
.LBB0_103:
	movq	%rbx, 368(%rsp)
	movq	%rbx, %rdi
	jmp	.LBB0_109
.LBB0_104:
	movq	376(%rsp), %rdx
	testq	%rdx, %rdx
	je	.LBB0_108
# %bb.105:
	cmpq	$1, %rdx
	jne	.LBB0_107
# %bb.106:
	movzbl	(%rsi), %eax
	movb	%al, (%rdi)
	jmp	.LBB0_108
.LBB0_107:
	.cfi_escape 0x2e, 0x00
	callq	memcpy@PLT
.LBB0_108:
	movq	376(%rsp), %rax
	movq	%rax, 240(%rsp)
	movq	232(%rsp), %rcx
	movb	$0, (%rcx,%rax)
	movq	368(%rsp), %rdi
.LBB0_109:
	movq	$0, 376(%rsp)
	movb	$0, (%rdi)
	movq	368(%rsp), %rdi
	cmpq	%rbx, %rdi
	je	.LBB0_111
# %bb.110:
	movq	384(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_111:
	movq	240(%rsp), %r14
.LBB0_112:
	movq	232(%rsp), %rbx
	testq	%r14, %r14
	je	.LBB0_116
# %bb.113:
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_114:                              # =>This Inner Loop Header: Depth=1
	movzbl	(%rbx,%r15), %edi
	.cfi_escape 0x2e, 0x00
	callq	tolower@PLT
	movb	%al, (%rbx,%r15)
	incq	%r15
	cmpq	%r15, %r14
	jne	.LBB0_114
# %bb.115:
	movq	232(%rsp), %rbx
	movq	240(%rsp), %r8
	jmp	.LBB0_117
.LBB0_116:
	xorl	%r8d, %r8d
.LBB0_117:
.Ltmp34:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.15(%rip), %rsi
	leaq	368(%rsp), %r13
	leaq	64(%rsp), %r9
	movl	$21, %edx
	movq	%r13, %rdi
	movq	%rbx, %rcx
	callq	_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE
.Ltmp35:                                # EH_LABEL
# %bb.118:
	movabsq	$9223372036854775773, %rax      # imm = 0x7FFFFFFFFFFFFFDD
	movq	376(%rsp), %rsi
	cmpq	%rax, %rsi
	jg	.LBB0_545
# %bb.119:
	leaq	34(%rsi), %r14
	movq	368(%rsp), %rax
	leaq	384(%rsp), %rbx
	movl	$15, %ecx
	cmpq	%rbx, %rax
	je	.LBB0_121
# %bb.120:
	movq	384(%rsp), %rcx
.LBB0_121:
	cmpq	%rcx, %r14
	jbe	.LBB0_124
# %bb.122:
.Ltmp37:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.16(%rip), %rcx
	leaq	368(%rsp), %rdi
	movl	$34, %r8d
	xorl	%edx, %edx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp38:                                # EH_LABEL
# %bb.123:
	movq	368(%rsp), %rax
	jmp	.LBB0_125
.LBB0_124:
	movups	.L.str.16+16(%rip), %xmm0
	movups	%xmm0, 16(%rax,%rsi)
	movdqu	.L.str.16(%rip), %xmm0
	movdqu	%xmm0, (%rax,%rsi)
	movw	$27749, 32(%rax,%rsi)           # imm = 0x6C65
.LBB0_125:
	movq	%r14, 376(%rsp)
	movb	$0, (%rax,%r14)
	leaq	1136(%rsp), %r15
	movq	%r15, 1120(%rsp)
	movq	368(%rsp), %rsi
	cmpq	%rbx, %rsi
	je	.LBB0_127
# %bb.126:
	movq	%rsi, 1120(%rsp)
	movq	376(%rsp), %r14
	movq	384(%rsp), %rax
	movq	%rax, 1136(%rsp)
	jmp	.LBB0_128
.LBB0_127:
	movq	376(%rsp), %r14
	leaq	1(%r14), %rdx
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%rbx, %rsi
	callq	memcpy@PLT
	movq	%r15, %rsi
.LBB0_128:
	movq	%r14, 1128(%rsp)
.Ltmp39:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1272(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc
.Ltmp40:                                # EH_LABEL
# %bb.129:
	cmpq	$4, 1280(%rsp)
	jne	.LBB0_534
# %bb.130:
	movq	1272(%rsp), %rax
	cmpl	$1869903201, (%rax)             # imm = 0x6F747561
	jne	.LBB0_534
# %bb.131:
	movl	$0, 320(%rsp)
.Ltmp48:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	320(%rsp), %rdi
	callq	hipRuntimeGetVersion@PLT
.Ltmp49:                                # EH_LABEL
# %bb.132:
.Ltmp50:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.19(%rip), %rsi
	movl	%eax, %edi
	movl	$632, %edx                      # imm = 0x278
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp51:                                # EH_LABEL
# %bb.133:
.Ltmp53:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$10240, %edi                    # imm = 0x2800
	callq	_Znwm@PLT
.Ltmp54:                                # EH_LABEL
# %bb.134:
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$10240, %edx                    # imm = 0x2800
	movq	%rax, 280(%rsp)                 # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movl	$1, %eax
	movss	.LCPI0_3(%rip), %xmm0           # xmm0 = [9.765625E-4,0.0E+0,0.0E+0,0.0E+0]
	jmp	.LBB0_138
	.p2align	4
.LBB0_135:                              #   in Loop: Header=BB0_138 Depth=1
	btl	$16, %ecx
	adcl	$32767, %ecx                    # imm = 0x7FFF
.LBB0_136:                              #   in Loop: Header=BB0_138 Depth=1
	movd	%ecx, %xmm1
.LBB0_137:                              #   in Loop: Header=BB0_138 Depth=1
	movd	%xmm1, %ecx
	shrl	$16, %ecx
	movq	280(%rsp), %rdx                 # 8-byte Reload
	movw	%cx, (%rdx,%rbx)
	addq	$2, %rbx
	cmpq	$10240, %rbx                    # imm = 0x2800
	je	.LBB0_141
.LBB0_138:                              # =>This Inner Loop Header: Depth=1
	imull	$1664525, %eax, %eax            # imm = 0x19660D
	addl	$1013904223, %eax               # imm = 0x3C6EF35F
	movl	%eax, %ecx
	shrl	$9, %ecx
	imulq	$8386561, %rcx, %rdx            # imm = 0x7FF801
	shrq	$35, %rdx
	movl	%edx, %esi
	shll	$12, %esi
	orl	%edx, %esi
	negl	%esi
	addl	%esi, %ecx
	addl	$-2048, %ecx                    # imm = 0xF800
	xorps	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	%xmm0, %xmm1
	movd	%xmm1, %ecx
	movl	%ecx, %edx
	notl	%edx
	testl	$2139095040, %edx               # imm = 0x7F800000
	jne	.LBB0_135
# %bb.139:                              #   in Loop: Header=BB0_138 Depth=1
	testw	%cx, %cx
	je	.LBB0_137
# %bb.140:                              #   in Loop: Header=BB0_138 Depth=1
	orl	$65536, %ecx                    # imm = 0x10000
	jmp	.LBB0_136
.LBB0_141:
.Ltmp56:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edi                   # imm = 0x78000
	callq	_Znwm@PLT
.Ltmp57:                                # EH_LABEL
# %bb.142:
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$491520, %edx                   # imm = 0x78000
	movq	%rax, 160(%rsp)                 # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movl	$3, %eax
	jmp	.LBB0_146
	.p2align	4
.LBB0_143:                              #   in Loop: Header=BB0_146 Depth=1
	btl	$16, %ecx
	adcl	$32767, %ecx                    # imm = 0x7FFF
.LBB0_144:                              #   in Loop: Header=BB0_146 Depth=1
	movd	%ecx, %xmm0
.LBB0_145:                              #   in Loop: Header=BB0_146 Depth=1
	movd	%xmm0, %ecx
	shrl	$16, %ecx
	movq	160(%rsp), %rdx                 # 8-byte Reload
	movw	%cx, (%rdx,%rbx)
	addq	$2, %rbx
	cmpq	$491520, %rbx                   # imm = 0x78000
	je	.LBB0_149
.LBB0_146:                              # =>This Inner Loop Header: Depth=1
	imull	$1664525, %eax, %eax            # imm = 0x19660D
	addl	$1013904223, %eax               # imm = 0x3C6EF35F
	movl	%eax, %ecx
	shrl	$9, %ecx
	imulq	$8386561, %rcx, %rdx            # imm = 0x7FF801
	shrq	$35, %rdx
	movl	%edx, %esi
	shll	$12, %esi
	orl	%edx, %esi
	negl	%esi
	addl	%esi, %ecx
	addl	$-2048, %ecx                    # imm = 0xF800
	xorps	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	.LCPI0_4(%rip), %xmm0
	movd	%xmm0, %ecx
	movl	%ecx, %edx
	notl	%edx
	testl	$2139095040, %edx               # imm = 0x7F800000
	jne	.LBB0_143
# %bb.147:                              #   in Loop: Header=BB0_146 Depth=1
	testw	%cx, %cx
	je	.LBB0_145
# %bb.148:                              #   in Loop: Header=BB0_146 Depth=1
	orl	$65536, %ecx                    # imm = 0x10000
	jmp	.LBB0_144
.LBB0_149:
.Ltmp59:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edi                   # imm = 0x78000
	callq	_Znwm@PLT
.Ltmp60:                                # EH_LABEL
# %bb.150:
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$491520, %edx                   # imm = 0x78000
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movl	$7, %eax
	jmp	.LBB0_154
	.p2align	4
.LBB0_151:                              #   in Loop: Header=BB0_154 Depth=1
	btl	$16, %ecx
	adcl	$32767, %ecx                    # imm = 0x7FFF
.LBB0_152:                              #   in Loop: Header=BB0_154 Depth=1
	movd	%ecx, %xmm0
.LBB0_153:                              #   in Loop: Header=BB0_154 Depth=1
	movd	%xmm0, %ecx
	shrl	$16, %ecx
	movq	152(%rsp), %rdx                 # 8-byte Reload
	movw	%cx, (%rdx,%rbx)
	addq	$2, %rbx
	cmpq	$491520, %rbx                   # imm = 0x78000
	je	.LBB0_157
.LBB0_154:                              # =>This Inner Loop Header: Depth=1
	imull	$1664525, %eax, %eax            # imm = 0x19660D
	addl	$1013904223, %eax               # imm = 0x3C6EF35F
	movl	%eax, %ecx
	shrl	$9, %ecx
	imulq	$8386561, %rcx, %rdx            # imm = 0x7FF801
	shrq	$35, %rdx
	movl	%edx, %esi
	shll	$12, %esi
	orl	%edx, %esi
	negl	%esi
	addl	%esi, %ecx
	addl	$-2048, %ecx                    # imm = 0xF800
	xorps	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	.LCPI0_4(%rip), %xmm0
	movd	%xmm0, %ecx
	movl	%ecx, %edx
	notl	%edx
	testl	$2139095040, %edx               # imm = 0x7F800000
	jne	.LBB0_151
# %bb.155:                              #   in Loop: Header=BB0_154 Depth=1
	testw	%cx, %cx
	je	.LBB0_153
# %bb.156:                              #   in Loop: Header=BB0_154 Depth=1
	orl	$65536, %ecx                    # imm = 0x10000
	jmp	.LBB0_152
.LBB0_157:
.Ltmp62:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$192, %edi
	callq	_Znwm@PLT
.Ltmp63:                                # EH_LABEL
# %bb.158:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 176(%rax)
	movups	%xmm0, 160(%rax)
	movups	%xmm0, 144(%rax)
	movups	%xmm0, 128(%rax)
	movups	%xmm0, 112(%rax)
	movups	%xmm0, 96(%rax)
	movups	%xmm0, 80(%rax)
	movups	%xmm0, 64(%rax)
	movups	%xmm0, 48(%rax)
	movups	%xmm0, 32(%rax)
	movups	%xmm0, 16(%rax)
	movups	%xmm0, (%rax)
.Ltmp65:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$192, %edi
	movq	%rax, %rbx
	movq	%rax, 272(%rsp)                 # 8-byte Spill
	callq	_Znwm@PLT
.Ltmp66:                                # EH_LABEL
# %bb.159:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 176(%rax)
	movups	%xmm0, 160(%rax)
	movups	%xmm0, 144(%rax)
	movups	%xmm0, 128(%rax)
	movups	%xmm0, 112(%rax)
	movups	%xmm0, 96(%rax)
	movups	%xmm0, 80(%rax)
	movups	%xmm0, 64(%rax)
	movups	%xmm0, 48(%rax)
	movups	%xmm0, 32(%rax)
	movups	%xmm0, 16(%rax)
	movups	%xmm0, (%rax)
	movaps	.LCPI0_5(%rip), %xmm0           # xmm0 = [-2.5E+0,-2.484375E+0,-2.46875E+0,-2.453125E+0]
	movups	%xmm0, (%rbx)
	movaps	.LCPI0_6(%rip), %xmm0           # xmm0 = [-7.5E-1,-7.37500011E-1,-7.25000024E-1,-7.12499976E-1]
	movups	%xmm0, (%rax)
	movaps	.LCPI0_7(%rip), %xmm0           # xmm0 = [-2.4375E+0,-2.421875E+0,-2.40625E+0,-2.390625E+0]
	movups	%xmm0, 16(%rbx)
	movaps	.LCPI0_8(%rip), %xmm0           # xmm0 = [-6.99999988E-1,-6.875E-1,-6.75000011E-1,-6.62500024E-1]
	movups	%xmm0, 16(%rax)
	movaps	.LCPI0_9(%rip), %xmm0           # xmm0 = [-2.375E+0,-2.359375E+0,-2.34375E+0,-2.328125E+0]
	movups	%xmm0, 32(%rbx)
	movaps	.LCPI0_10(%rip), %xmm0          # xmm0 = [-6.49999976E-1,-6.37499988E-1,-6.25E-1,-6.12500011E-1]
	movups	%xmm0, 32(%rax)
	movaps	.LCPI0_11(%rip), %xmm0          # xmm0 = [-2.3125E+0,-2.296875E+0,-2.28125E+0,-2.265625E+0]
	movups	%xmm0, 48(%rbx)
	movaps	.LCPI0_12(%rip), %xmm0          # xmm0 = [-6.00000024E-1,-5.87499976E-1,-5.74999988E-1,-5.625E-1]
	movups	%xmm0, 48(%rax)
	movaps	.LCPI0_13(%rip), %xmm0          # xmm0 = [-2.25E+0,-2.234375E+0,-2.21875E+0,-2.203125E+0]
	movups	%xmm0, 64(%rbx)
	movaps	.LCPI0_14(%rip), %xmm0          # xmm0 = [-5.50000012E-1,-5.37500024E-1,-5.24999976E-1,-5.12499988E-1]
	movups	%xmm0, 64(%rax)
	movaps	.LCPI0_15(%rip), %xmm0          # xmm0 = [-2.1875E+0,-2.171875E+0,-2.15625E+0,-2.140625E+0]
	movups	%xmm0, 80(%rbx)
	movaps	.LCPI0_16(%rip), %xmm0          # xmm0 = [-5.0E-1,-4.87500012E-1,-4.74999994E-1,-4.62500006E-1]
	movups	%xmm0, 80(%rax)
	movaps	.LCPI0_17(%rip), %xmm0          # xmm0 = [-2.125E+0,-2.109375E+0,-2.09375E+0,-2.078125E+0]
	movups	%xmm0, 96(%rbx)
	movaps	.LCPI0_18(%rip), %xmm0          # xmm0 = [-4.49999988E-1,-4.375E-1,-4.25000012E-1,-4.12499994E-1]
	movups	%xmm0, 96(%rax)
	movaps	.LCPI0_19(%rip), %xmm0          # xmm0 = [-2.0625E+0,-2.046875E+0,-2.03125E+0,-2.015625E+0]
	movups	%xmm0, 112(%rbx)
	movaps	.LCPI0_20(%rip), %xmm0          # xmm0 = [-4.00000006E-1,-3.87499988E-1,-3.75E-1,-3.62500012E-1]
	movups	%xmm0, 112(%rax)
	movaps	.LCPI0_21(%rip), %xmm0          # xmm0 = [-2.0E+0,-1.984375E+0,-1.96875E+0,-1.953125E+0]
	movups	%xmm0, 128(%rbx)
	movaps	.LCPI0_22(%rip), %xmm0          # xmm0 = [-3.49999994E-1,-3.37500006E-1,-3.24999988E-1,-3.125E-1]
	movups	%xmm0, 128(%rax)
	movaps	.LCPI0_23(%rip), %xmm0          # xmm0 = [-1.9375E+0,-1.921875E+0,-1.90625E+0,-1.890625E+0]
	movups	%xmm0, 144(%rbx)
	movaps	.LCPI0_24(%rip), %xmm0          # xmm0 = [-3.00000012E-1,-2.87499994E-1,-2.75000006E-1,-2.62499988E-1]
	movups	%xmm0, 144(%rax)
	movaps	.LCPI0_25(%rip), %xmm0          # xmm0 = [-1.875E+0,-1.859375E+0,-1.84375E+0,-1.828125E+0]
	movups	%xmm0, 160(%rbx)
	movaps	.LCPI0_26(%rip), %xmm0          # xmm0 = [-2.5E-1,-2.37500012E-1,-2.25000024E-1,-2.12499976E-1]
	movups	%xmm0, 160(%rax)
	movaps	.LCPI0_27(%rip), %xmm0          # xmm0 = [-1.8125E+0,-1.796875E+0,-1.78125E+0,-1.765625E+0]
	movups	%xmm0, 176(%rbx)
	movapd	.LCPI0_28(%rip), %xmm0          # xmm0 = [-1.99999988E-1,-1.875E-1,-1.75000012E-1,-1.62500024E-1]
	movq	%rax, 264(%rsp)                 # 8-byte Spill
	movupd	%xmm0, 176(%rax)
	movq	$0, 880(%rsp)
	movq	$5120, 888(%rsp)                # imm = 0x1400
.Ltmp68:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	880(%rsp), %rdi
	movl	$10240, %esi                    # imm = 0x2800
	callq	hipMalloc@PLT
.Ltmp69:                                # EH_LABEL
# %bb.160:
.Ltmp70:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp71:                                # EH_LABEL
# %bb.161:
	movq	$0, 864(%rsp)
	movq	$48, 872(%rsp)
.Ltmp73:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	864(%rsp), %rdi
	movl	$192, %esi
	callq	hipMalloc@PLT
.Ltmp74:                                # EH_LABEL
# %bb.162:
.Ltmp75:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp76:                                # EH_LABEL
# %bb.163:
	movq	$0, 848(%rsp)
	movq	$48, 856(%rsp)
.Ltmp78:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	848(%rsp), %rdi
	movl	$192, %esi
	callq	hipMalloc@PLT
.Ltmp79:                                # EH_LABEL
# %bb.164:
.Ltmp80:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp81:                                # EH_LABEL
# %bb.165:
	movq	880(%rsp), %rdi
.Ltmp83:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$10240, %edx                    # imm = 0x2800
	movq	280(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp84:                                # EH_LABEL
# %bb.166:
.Ltmp85:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp86:                                # EH_LABEL
# %bb.167:
	movq	864(%rsp), %rdi
.Ltmp87:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$192, %edx
	movq	272(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp88:                                # EH_LABEL
# %bb.168:
.Ltmp89:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp90:                                # EH_LABEL
# %bb.169:
	movq	848(%rsp), %rdi
.Ltmp91:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$192, %edx
	movq	264(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp92:                                # EH_LABEL
# %bb.170:
.Ltmp93:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp94:                                # EH_LABEL
# %bb.171:
	movq	$0, 1056(%rsp)
	movq	$245760, 1064(%rsp)             # imm = 0x3C000
.Ltmp96:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1056(%rsp), %rbx
	movl	$491520, %esi                   # imm = 0x78000
	movq	%rbx, %rdi
	callq	hipMalloc@PLT
.Ltmp97:                                # EH_LABEL
# %bb.172:
.Ltmp98:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp99:                                # EH_LABEL
# %bb.173:
	leaq	1072(%rsp), %r14
	movq	$0, 1072(%rsp)
	movq	$245760, 1080(%rsp)             # imm = 0x3C000
.Ltmp101:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	%r14, %rdi
	callq	hipMalloc@PLT
.Ltmp102:                               # EH_LABEL
# %bb.174:
.Ltmp103:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp104:                               # EH_LABEL
# %bb.175:
	leaq	1088(%rsp), %r14
	movq	$0, 1088(%rsp)
	movq	$245760, 1096(%rsp)             # imm = 0x3C000
.Ltmp105:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	%r14, %rdi
	callq	hipMalloc@PLT
.Ltmp106:                               # EH_LABEL
# %bb.176:
.Ltmp107:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp108:                               # EH_LABEL
# %bb.177:
	movq	$0, 1008(%rsp)
	movq	$245760, 1016(%rsp)             # imm = 0x3C000
.Ltmp110:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1008(%rsp), %rbx
	movl	$491520, %esi                   # imm = 0x78000
	movq	%rbx, %rdi
	callq	hipMalloc@PLT
.Ltmp111:                               # EH_LABEL
# %bb.178:
.Ltmp112:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp113:                               # EH_LABEL
# %bb.179:
	leaq	1024(%rsp), %r14
	movq	$0, 1024(%rsp)
	movq	$245760, 1032(%rsp)             # imm = 0x3C000
.Ltmp115:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	%r14, %rdi
	callq	hipMalloc@PLT
.Ltmp116:                               # EH_LABEL
# %bb.180:
.Ltmp117:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp118:                               # EH_LABEL
# %bb.181:
	leaq	1040(%rsp), %r14
	movq	$0, 1040(%rsp)
	movq	$245760, 1048(%rsp)             # imm = 0x3C000
.Ltmp119:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	%r14, %rdi
	callq	hipMalloc@PLT
.Ltmp120:                               # EH_LABEL
# %bb.182:
.Ltmp121:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp122:                               # EH_LABEL
# %bb.183:
	movq	1056(%rsp), %rdi
.Ltmp124:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edx                   # imm = 0x78000
	movq	160(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp125:                               # EH_LABEL
# %bb.184:
.Ltmp126:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp127:                               # EH_LABEL
# %bb.185:
	movq	1008(%rsp), %rdi
.Ltmp128:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edx                   # imm = 0x78000
	movq	152(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp129:                               # EH_LABEL
# %bb.186:
.Ltmp130:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp131:                               # EH_LABEL
# %bb.187:
	movq	1072(%rsp), %rdi
.Ltmp132:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edx                   # imm = 0x78000
	movq	160(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp133:                               # EH_LABEL
# %bb.188:
.Ltmp134:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp135:                               # EH_LABEL
# %bb.189:
	movq	1024(%rsp), %rdi
.Ltmp136:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edx                   # imm = 0x78000
	movq	152(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp137:                               # EH_LABEL
# %bb.190:
.Ltmp138:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp139:                               # EH_LABEL
# %bb.191:
	movq	1088(%rsp), %rdi
.Ltmp140:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edx                   # imm = 0x78000
	movq	160(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp141:                               # EH_LABEL
# %bb.192:
.Ltmp142:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp143:                               # EH_LABEL
# %bb.193:
	movq	1040(%rsp), %rdi
.Ltmp144:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$491520, %edx                   # imm = 0x78000
	movq	152(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp145:                               # EH_LABEL
# %bb.194:
.Ltmp146:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp147:                               # EH_LABEL
# %bb.195:
	movq	$0, 368(%rsp)
	movq	$50, 376(%rsp)
	xorl	%ebx, %ebx
.Ltmp149:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$100, %esi
	movq	%r13, %rdi
	callq	hipMalloc@PLT
.Ltmp150:                               # EH_LABEL
# %bb.196:
	xorl	%ebx, %ebx
.Ltmp151:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp152:                               # EH_LABEL
# %bb.197:
	movq	$48, 384(%rsp)
	leaq	392(%rsp), %rbp
	movq	$0, 392(%rsp)
	movq	$50, 400(%rsp)
	leaq	368(%rsp), %r13
	xorl	%ebx, %ebx
.Ltmp153:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$100, %esi
	movq	%rbp, %rdi
	callq	hipMalloc@PLT
.Ltmp154:                               # EH_LABEL
# %bb.198:
	xorl	%ebx, %ebx
.Ltmp155:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp156:                               # EH_LABEL
# %bb.199:
	movq	$48, 408(%rsp)
	leaq	416(%rsp), %rdi
	movq	$0, 416(%rsp)
	movq	$50, 424(%rsp)
	leaq	368(%rsp), %r13
	xorl	%ebx, %ebx
.Ltmp157:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$200, %esi
	movq	%rbp, %r12
	callq	hipMalloc@PLT
.Ltmp158:                               # EH_LABEL
# %bb.200:
	xorl	%ebx, %ebx
.Ltmp159:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movq	%rbp, %r12
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp160:                               # EH_LABEL
# %bb.201:
	movq	$48, 432(%rsp)
	leaq	440(%rsp), %rdi
	movq	$0, 440(%rsp)
	movq	$50, 448(%rsp)
	leaq	368(%rsp), %r13
	xorl	%ebx, %ebx
.Ltmp161:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$200, %esi
	leaq	416(%rsp), %r14
	movq	%rbp, %r12
	callq	hipMalloc@PLT
.Ltmp162:                               # EH_LABEL
# %bb.202:
	xorl	%ebx, %ebx
.Ltmp163:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	leaq	416(%rsp), %r14
	movq	%rbp, %r12
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp164:                               # EH_LABEL
# %bb.203:
	movq	$48, 456(%rsp)
	leaq	464(%rsp), %r15
	movq	$0, 464(%rsp)
	movq	$50, 472(%rsp)
	movl	$96, %ebx
.Ltmp165:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$100, %esi
	movq	%r15, %r13
	movq	%r15, %rdi
	callq	hipMalloc@PLT
.Ltmp166:                               # EH_LABEL
# %bb.204:
.Ltmp167:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movq	%r15, %r13
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp168:                               # EH_LABEL
# %bb.205:
	movq	$48, 480(%rsp)
	leaq	488(%rsp), %rdi
	movq	$0, 488(%rsp)
	movq	$50, 496(%rsp)
	movl	$96, %ebx
.Ltmp169:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$100, %esi
	movq	%r15, %r13
	callq	hipMalloc@PLT
.Ltmp170:                               # EH_LABEL
# %bb.206:
.Ltmp171:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movq	%r15, %r13
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp172:                               # EH_LABEL
# %bb.207:
	movq	$48, 504(%rsp)
	leaq	512(%rsp), %rdi
	movq	$0, 512(%rsp)
	movq	$50, 520(%rsp)
	movl	$96, %ebx
.Ltmp173:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$200, %esi
	leaq	488(%rsp), %r12
	movq	%r15, %r13
	callq	hipMalloc@PLT
.Ltmp174:                               # EH_LABEL
# %bb.208:
.Ltmp175:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	leaq	488(%rsp), %r12
	movq	%r15, %r13
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp176:                               # EH_LABEL
# %bb.209:
	movq	$48, 528(%rsp)
	leaq	536(%rsp), %rdi
	movq	$0, 536(%rsp)
	movq	$50, 544(%rsp)
	movl	$96, %ebx
.Ltmp177:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$200, %esi
	leaq	512(%rsp), %r14
	leaq	488(%rsp), %r12
	movq	%r15, %r13
	callq	hipMalloc@PLT
.Ltmp178:                               # EH_LABEL
# %bb.210:
.Ltmp179:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	leaq	512(%rsp), %r14
	leaq	488(%rsp), %r12
	movq	%r15, %r13
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp180:                               # EH_LABEL
# %bb.211:
	movq	$48, 552(%rsp)
	leaq	560(%rsp), %r13
	movq	$0, 560(%rsp)
	movq	$50, 568(%rsp)
	movl	$192, %ebx
.Ltmp181:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$100, %esi
	movq	%r13, %rdi
	callq	hipMalloc@PLT
.Ltmp182:                               # EH_LABEL
# %bb.212:
.Ltmp183:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp184:                               # EH_LABEL
# %bb.213:
	movq	$48, 576(%rsp)
	leaq	584(%rsp), %rdi
	movq	$0, 584(%rsp)
	movq	$50, 592(%rsp)
	movl	$192, %ebx
.Ltmp186:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$100, %esi
	movq	%rdi, %r12
	callq	hipMalloc@PLT
.Ltmp187:                               # EH_LABEL
# %bb.214:
.Ltmp188:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp189:                               # EH_LABEL
# %bb.215:
	movq	$48, 600(%rsp)
	leaq	608(%rsp), %rdi
	movq	$0, 608(%rsp)
	movq	$50, 616(%rsp)
	movl	$192, %ebx
.Ltmp191:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$200, %esi
	movq	%rdi, %r14
	callq	hipMalloc@PLT
.Ltmp192:                               # EH_LABEL
# %bb.216:
.Ltmp193:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp194:                               # EH_LABEL
# %bb.217:
	movq	$48, 624(%rsp)
	leaq	632(%rsp), %rdi
	movq	$0, 632(%rsp)
	movq	$50, 640(%rsp)
	movl	$192, %ebx
.Ltmp196:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$200, %esi
	callq	hipMalloc@PLT
.Ltmp197:                               # EH_LABEL
# %bb.218:
.Ltmp198:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp199:                               # EH_LABEL
# %bb.219:
	movq	$48, 648(%rsp)
	movq	368(%rsp), %rdi
	movq	384(%rsp), %rsi
.Ltmp201:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Ltmp202:                               # EH_LABEL
# %bb.220:
	movq	392(%rsp), %rdi
	movq	408(%rsp), %rsi
.Ltmp203:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Ltmp204:                               # EH_LABEL
# %bb.221:
	movq	416(%rsp), %rdi
	movq	432(%rsp), %rsi
.Ltmp205:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp206:                               # EH_LABEL
# %bb.222:
	movq	440(%rsp), %rdi
	movq	456(%rsp), %rsi
.Ltmp207:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp208:                               # EH_LABEL
# %bb.223:
	movq	464(%rsp), %rdi
	movq	480(%rsp), %rsi
.Ltmp209:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Ltmp210:                               # EH_LABEL
# %bb.224:
	movq	488(%rsp), %rdi
	movq	504(%rsp), %rsi
.Ltmp211:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Ltmp212:                               # EH_LABEL
# %bb.225:
	movq	512(%rsp), %rdi
	movq	528(%rsp), %rsi
.Ltmp213:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp214:                               # EH_LABEL
# %bb.226:
	movq	536(%rsp), %rdi
	movq	552(%rsp), %rsi
.Ltmp215:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp216:                               # EH_LABEL
# %bb.227:
	movq	560(%rsp), %rdi
	movq	576(%rsp), %rsi
.Ltmp217:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Ltmp218:                               # EH_LABEL
# %bb.228:
	movq	584(%rsp), %rdi
	movq	600(%rsp), %rsi
.Ltmp219:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Ltmp220:                               # EH_LABEL
# %bb.229:
	movq	608(%rsp), %rdi
	movq	624(%rsp), %rsi
.Ltmp221:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp222:                               # EH_LABEL
# %bb.230:
	movq	632(%rsp), %rdi
	movq	648(%rsp), %rsi
.Ltmp223:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp224:                               # EH_LABEL
# %bb.231:
	movq	880(%rsp), %rax
	movq	1056(%rsp), %rsi
	movq	1008(%rsp), %rdi
	movq	864(%rsp), %rcx
	movq	848(%rsp), %rdx
	movq	%rax, 1176(%rsp)
	movq	%rsi, 1184(%rsp)
	movq	%rdi, 1192(%rsp)
	movq	%rcx, 1200(%rsp)
	movq	%rdx, 1208(%rsp)
	movq	368(%rsp), %rsi
	movq	392(%rsp), %rdi
	addq	$2, %rsi
	movq	%rsi, 1216(%rsp)
	addq	$2, %rdi
	movq	%rdi, 1224(%rsp)
	movq	416(%rsp), %rsi
	addq	$4, %rsi
	movq	%rsi, 1232(%rsp)
	movq	440(%rsp), %rsi
	addq	$4, %rsi
	movq	%rsi, 1240(%rsp)
	movq	1072(%rsp), %rsi
	movq	1024(%rsp), %rdi
	movq	%rax, 1568(%rsp)
	movq	%rsi, 1576(%rsp)
	movq	%rdi, 1584(%rsp)
	movq	%rcx, 1592(%rsp)
	movq	%rdx, 1600(%rsp)
	movq	464(%rsp), %rsi
	movq	488(%rsp), %rdi
	addq	$2, %rsi
	movq	%rsi, 1608(%rsp)
	addq	$2, %rdi
	movq	%rdi, 1616(%rsp)
	movq	512(%rsp), %rsi
	addq	$4, %rsi
	movq	%rsi, 1624(%rsp)
	movq	536(%rsp), %rsi
	addq	$4, %rsi
	movq	%rsi, 1632(%rsp)
	movq	1088(%rsp), %rsi
	movq	1040(%rsp), %rdi
	movq	%rax, 1496(%rsp)
	movq	%rsi, 1504(%rsp)
	movq	%rdi, 1512(%rsp)
	movq	%rcx, 1520(%rsp)
	movq	%rdx, 1528(%rsp)
	movq	560(%rsp), %rax
	addq	$2, %rax
	movq	%rax, 1536(%rsp)
	movq	584(%rsp), %rax
	addq	$2, %rax
	movq	%rax, 1544(%rsp)
	movq	608(%rsp), %rax
	addq	$4, %rax
	movq	%rax, 1552(%rsp)
	movq	632(%rsp), %rax
	addq	$4, %rax
	movq	%rax, 1560(%rsp)
.Ltmp226:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1176(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
.Ltmp227:                               # EH_LABEL
# %bb.232:
.Ltmp228:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.20(%rip), %rsi
	movl	%eax, %edi
	movl	$658, %edx                      # imm = 0x292
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp229:                               # EH_LABEL
# %bb.233:
.Ltmp230:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1568(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp231:                               # EH_LABEL
# %bb.234:
.Ltmp232:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.21(%rip), %rsi
	movl	%eax, %edi
	movl	$659, %edx                      # imm = 0x293
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp233:                               # EH_LABEL
# %bb.235:
.Ltmp234:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1496(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp235:                               # EH_LABEL
# %bb.236:
.Ltmp236:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.22(%rip), %rsi
	movl	%eax, %edi
	movl	$660, %edx                      # imm = 0x294
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp237:                               # EH_LABEL
# %bb.237:
.Ltmp238:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp239:                               # EH_LABEL
# %bb.238:
.Ltmp240:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.23(%rip), %rsi
	movl	%eax, %edi
	movl	$661, %edx                      # imm = 0x295
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp241:                               # EH_LABEL
# %bb.239:
	xorpd	%xmm0, %xmm0
	movapd	%xmm0, 112(%rsp)
	movapd	%xmm0, 96(%rsp)
	movapd	%xmm0, 80(%rsp)
	movapd	%xmm0, 64(%rsp)
	movq	$0, 128(%rsp)
	movapd	%xmm0, 704(%rsp)
	movapd	%xmm0, 688(%rsp)
	movapd	%xmm0, 672(%rsp)
	movapd	%xmm0, 656(%rsp)
	movq	$0, 720(%rsp)
	movapd	%xmm0, 976(%rsp)
	movapd	%xmm0, 960(%rsp)
	movapd	%xmm0, 944(%rsp)
	movapd	%xmm0, 928(%rsp)
	movq	$0, 992(%rsp)
	movapd	%xmm0, 800(%rsp)
	movapd	%xmm0, 784(%rsp)
	movapd	%xmm0, 768(%rsp)
	movapd	%xmm0, 752(%rsp)
	movq	$0, 816(%rsp)
.Ltmp243:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	368(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
.Ltmp244:                               # EH_LABEL
# %bb.240:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	32(%rsp), %rax
	movapd	%xmm0, 64(%rsp)
	movq	%rax, 1392(%rsp)                # 8-byte Spill
	movq	%rax, 80(%rsp)
.Ltmp245:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	movq	%rbp, %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
.Ltmp246:                               # EH_LABEL
# %bb.241:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 360(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %rbx
	movapd	%xmm0, 656(%rsp)
	movq	%rbx, 672(%rsp)
.Ltmp247:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	416(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.Ltmp248:                               # EH_LABEL
# %bb.242:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 344(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %rbp
	movapd	%xmm0, 928(%rsp)
	movq	%rbp, 944(%rsp)
.Ltmp249:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	440(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.Ltmp250:                               # EH_LABEL
# %bb.243:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 288(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %rax
	movapd	%xmm0, 752(%rsp)
	movq	%rax, 1384(%rsp)                # 8-byte Spill
	movq	%rax, 768(%rsp)
.Ltmp251:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	movq	%r15, %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
.Ltmp252:                               # EH_LABEL
# %bb.244:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 352(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %r15
	movupd	%xmm0, 88(%rsp)
	movq	%r15, 104(%rsp)
.Ltmp253:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	488(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
.Ltmp254:                               # EH_LABEL
# %bb.245:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 1168(%rsp)                # 8-byte Spill
	movq	32(%rsp), %rax
	movupd	%xmm0, 680(%rsp)
	movq	%rax, 1376(%rsp)                # 8-byte Spill
	movq	%rax, 696(%rsp)
.Ltmp255:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	512(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.Ltmp256:                               # EH_LABEL
# %bb.246:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 304(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %rax
	movupd	%xmm0, 952(%rsp)
	movq	%rax, 1368(%rsp)                # 8-byte Spill
	movq	%rax, 968(%rsp)
.Ltmp257:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	536(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.Ltmp258:                               # EH_LABEL
# %bb.247:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %rax
	movupd	%xmm0, 776(%rsp)
	movq	%rax, 1360(%rsp)                # 8-byte Spill
	movq	%rax, 792(%rsp)
.Ltmp259:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	movq	%r13, %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
.Ltmp260:                               # EH_LABEL
# %bb.248:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 1160(%rsp)                # 8-byte Spill
	movq	32(%rsp), %rax
	movapd	%xmm0, 112(%rsp)
	movq	%rax, 1344(%rsp)                # 8-byte Spill
	movq	%rax, 128(%rsp)
.Ltmp262:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	movq	%r12, %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
.Ltmp263:                               # EH_LABEL
# %bb.249:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 1152(%rsp)                # 8-byte Spill
	movq	32(%rsp), %rax
	movapd	%xmm0, 704(%rsp)
	movq	%rax, 1320(%rsp)                # 8-byte Spill
	movq	%rax, 720(%rsp)
.Ltmp265:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	movq	%r14, %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.Ltmp266:                               # EH_LABEL
# %bb.250:
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 208(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %r14
	movapd	%xmm0, 976(%rsp)
	movq	%r14, 992(%rsp)
.Ltmp268:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	632(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.Ltmp269:                               # EH_LABEL
# %bb.251:
	movq	%r14, 1312(%rsp)                # 8-byte Spill
	movq	%rbp, 1328(%rsp)                # 8-byte Spill
	movq	%r15, 1352(%rsp)                # 8-byte Spill
	movq	%rbx, 1336(%rsp)                # 8-byte Spill
	movapd	16(%rsp), %xmm0
	movq	16(%rsp), %rax
	movq	%rax, 168(%rsp)                 # 8-byte Spill
	movq	32(%rsp), %rax
	movapd	%xmm0, 800(%rsp)
	movq	%rax, 1304(%rsp)                # 8-byte Spill
	movq	%rax, 816(%rsp)
	xorpd	%xmm0, %xmm0
	movsd	%xmm0, 336(%rsp)                # 8-byte Spill
	movb	$1, %r12b
	xorl	%r13d, %r13d
	movq	160(%rsp), %r15                 # 8-byte Reload
	movq	152(%rsp), %rbp                 # 8-byte Reload
	movapd	%xmm0, 1248(%rsp)               # 16-byte Spill
	movq	280(%rsp), %rbx                 # 8-byte Reload
	jmp	.LBB0_255
	.p2align	4
.LBB0_252:                              #   in Loop: Header=BB0_255 Depth=1
	movl	%r8d, %r9d
.LBB0_253:                              #   in Loop: Header=BB0_255 Depth=1
	movq	208(%rsp), %r8                  # 8-byte Reload
	movd	(%r8,%r13,4), %xmm5             # xmm5 = mem[0],zero,zero,zero
	shll	$16, %r9d
	movd	%r9d, %xmm6
	cvtss2sd	%xmm6, %xmm7
	movq	168(%rsp), %r8                  # 8-byte Reload
	movd	(%r8,%r13,4), %xmm6             # xmm6 = mem[0],zero,zero,zero
	subsd	%xmm8, %xmm7
	andpd	.LCPI0_32(%rip), %xmm7
	xorl	%r12d, %r12d
.LBB0_254:                              #   in Loop: Header=BB0_255 Depth=1
	movsd	.LCPI0_31(%rip), %xmm8          # xmm8 = [1.0E+0,0.0E+0]
	addsd	%xmm8, %xmm0
	divsd	%xmm0, %xmm8
	mulsd	%xmm9, %xmm12
	shll	$16, %esi
	movd	%esi, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movsd	216(%rsp), %xmm11               # 8-byte Reload
                                        # xmm11 = mem[0],zero
	subsd	%xmm11, %xmm0
	andpd	%xmm10, %xmm0
	maxsd	336(%rsp), %xmm0                # 8-byte Folded Reload
	maxsd	%xmm0, %xmm1
	movd	%edx, %xmm0
	cvtss2sd	%xmm0, %xmm0
	addsd	%xmm12, %xmm0
	andpd	%xmm10, %xmm0
	maxsd	1248(%rsp), %xmm0               # 16-byte Folded Reload
	movd	%eax, %xmm9
	cvtss2sd	%xmm9, %xmm9
	subsd	%xmm8, %xmm9
	andpd	%xmm10, %xmm9
	maxsd	%xmm0, %xmm9
	shll	$16, %ecx
	movd	%ecx, %xmm0
	cvtss2sd	%xmm0, %xmm0
	subsd	%xmm11, %xmm0
	andpd	%xmm10, %xmm0
	maxsd	%xmm1, %xmm0
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm4, %xmm1
	maxsd	%xmm0, %xmm2
	addsd	%xmm12, %xmm1
	andpd	%xmm10, %xmm1
	maxsd	%xmm9, %xmm1
	xorps	%xmm0, %xmm0
	cvtss2sd	%xmm3, %xmm0
	subsd	%xmm8, %xmm0
	andpd	%xmm10, %xmm0
	shll	$16, %edi
	movd	%edi, %xmm3
	cvtss2sd	%xmm3, %xmm3
	maxsd	%xmm1, %xmm0
	subsd	%xmm11, %xmm3
	andpd	%xmm10, %xmm3
	maxsd	%xmm2, %xmm3
	maxsd	%xmm3, %xmm7
	movsd	%xmm7, 336(%rsp)                # 8-byte Spill
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm5, %xmm1
	addsd	%xmm12, %xmm1
	andpd	%xmm10, %xmm1
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm6, %xmm2
	maxsd	%xmm0, %xmm1
	subsd	%xmm8, %xmm2
	andpd	%xmm10, %xmm2
	maxsd	%xmm1, %xmm2
	movapd	%xmm2, 1248(%rsp)               # 16-byte Spill
	incq	%r13
	addq	$10240, %rbp                    # imm = 0x2800
	addq	$10240, %r15                    # imm = 0x2800
	cmpq	$48, %r13
	je	.LBB0_279
.LBB0_255:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_256 Depth 2
	xorl	%r14d, %r14d
	xorps	%xmm2, %xmm2
	xorpd	%xmm0, %xmm0
	.p2align	4
.LBB0_256:                              #   Parent Loop BB0_255 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movsd	%xmm0, 144(%rsp)                # 8-byte Spill
	movzwl	(%rbx,%r14,2), %eax
	shll	$16, %eax
	movd	%eax, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movsd	%xmm0, 56(%rsp)                 # 8-byte Spill
	movzwl	(%r15,%r14,2), %eax
	shll	$16, %eax
	movd	%eax, %xmm1
	cvtss2sd	%xmm1, %xmm1
	.cfi_escape 0x2e, 0x00
	callq	fma@PLT
	movsd	%xmm0, 216(%rsp)                # 8-byte Spill
	movzwl	(%rbp,%r14,2), %eax
	shll	$16, %eax
	movd	%eax, %xmm0
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	.cfi_escape 0x2e, 0x00
	movsd	56(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	144(%rsp), %xmm2                # 8-byte Reload
                                        # xmm2 = mem[0],zero
	callq	fma@PLT
	movsd	216(%rsp), %xmm2                # 8-byte Reload
                                        # xmm2 = mem[0],zero
	incq	%r14
	cmpq	$5120, %r14                     # imm = 0x1400
	jne	.LBB0_256
# %bb.257:                              #   in Loop: Header=BB0_255 Depth=1
	xorps	%xmm1, %xmm1
	cvtsd2ss	%xmm2, %xmm1
	movd	%xmm1, %eax
	movl	%eax, %ecx
	notl	%ecx
	testl	$2139095040, %ecx               # imm = 0x7F800000
	jne	.LBB0_260
# %bb.258:                              #   in Loop: Header=BB0_255 Depth=1
	testw	%ax, %ax
	movq	264(%rsp), %rdx                 # 8-byte Reload
	je	.LBB0_261
# %bb.259:                              #   in Loop: Header=BB0_255 Depth=1
	orl	$65536, %eax                    # imm = 0x10000
	movd	%eax, %xmm1
	jmp	.LBB0_261
	.p2align	4
.LBB0_260:                              #   in Loop: Header=BB0_255 Depth=1
	btl	$16, %eax
	adcl	$32767, %eax                    # imm = 0x7FFF
	movd	%eax, %xmm1
	movq	264(%rsp), %rdx                 # 8-byte Reload
.LBB0_261:                              #   in Loop: Header=BB0_255 Depth=1
	cvtsd2ss	%xmm0, %xmm4
	movd	%xmm4, %eax
	movl	%eax, %ecx
	notl	%ecx
	testl	$2139095040, %ecx               # imm = 0x7F800000
	jne	.LBB0_264
# %bb.262:                              #   in Loop: Header=BB0_255 Depth=1
	testw	%ax, %ax
	je	.LBB0_266
# %bb.263:                              #   in Loop: Header=BB0_255 Depth=1
	orl	$65536, %eax                    # imm = 0x10000
	jmp	.LBB0_265
	.p2align	4
.LBB0_264:                              #   in Loop: Header=BB0_255 Depth=1
	btl	$16, %eax
	adcl	$32767, %eax                    # imm = 0x7FFF
.LBB0_265:                              #   in Loop: Header=BB0_255 Depth=1
	movd	%eax, %xmm4
.LBB0_266:                              #   in Loop: Header=BB0_255 Depth=1
	movsd	%xmm0, 144(%rsp)                # 8-byte Spill
	movd	%xmm1, %eax
	andl	$-65536, %eax                   # imm = 0xFFFF0000
	movd	%eax, %xmm1
	addss	(%rdx,%r13,4), %xmm1
	xorps	%xmm0, %xmm0
	cvtss2sd	%xmm1, %xmm0
	ucomiss	.LCPI0_29(%rip), %xmm1
	ja	.LBB0_268
# %bb.267:                              #   in Loop: Header=BB0_255 Depth=1
	.cfi_escape 0x2e, 0x00
	movss	%xmm4, 136(%rsp)                # 4-byte Spill
	callq	exp@PLT
	.cfi_escape 0x2e, 0x00
	callq	log1p@PLT
	movss	136(%rsp), %xmm4                # 4-byte Reload
                                        # xmm4 = mem[0],zero,zero,zero
.LBB0_268:                              #   in Loop: Header=BB0_255 Depth=1
	movsd	%xmm0, 56(%rsp)                 # 8-byte Spill
	movq	272(%rsp), %rax                 # 8-byte Reload
	movd	%xmm4, %r14d
	andl	$-65536, %r14d                  # imm = 0xFFFF0000
	movss	(%rax,%r13,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	.cfi_escape 0x2e, 0x00
	callq	exp@PLT
	movsd	%xmm0, 136(%rsp)                # 8-byte Spill
	movd	%r14d, %xmm0
	cvtss2sd	%xmm0, %xmm0
	xorps	.LCPI0_30(%rip), %xmm0
	.cfi_escape 0x2e, 0x00
	callq	exp@PLT
	movq	8(%rsp), %rax                   # 8-byte Reload
	movzwl	(%rax,%r13,2), %esi
	movq	360(%rsp), %rax                 # 8-byte Reload
	movzwl	(%rax,%r13,2), %r8d
	movq	344(%rsp), %rax                 # 8-byte Reload
	movl	(%rax,%r13,4), %edx
	movq	288(%rsp), %rax                 # 8-byte Reload
	movl	(%rax,%r13,4), %eax
	movl	%r8d, %ecx
	shll	$16, %ecx
	movd	%ecx, %xmm1
	cvtss2sd	%xmm1, %xmm1
	movsd	144(%rsp), %xmm8                # 8-byte Reload
                                        # xmm8 = mem[0],zero
	subsd	%xmm8, %xmm1
	movapd	.LCPI0_32(%rip), %xmm10         # xmm10 = [NaN,NaN]
	andpd	%xmm10, %xmm1
	movq	352(%rsp), %rcx                 # 8-byte Reload
	movzwl	(%rcx,%r13,2), %ecx
	movq	1168(%rsp), %rdi                # 8-byte Reload
	movzwl	(%rdi,%r13,2), %edi
	cmpw	%si, %cx
	jne	.LBB0_273
# %bb.269:                              #   in Loop: Header=BB0_255 Depth=1
	cmpw	%r8w, %di
	jne	.LBB0_273
# %bb.270:                              #   in Loop: Header=BB0_255 Depth=1
	movq	304(%rsp), %rdi                 # 8-byte Reload
	movl	(%rdi,%r13,4), %r9d
	movl	%r8d, %edi
	cmpl	%edx, %r9d
	jne	.LBB0_273
# %bb.271:                              #   in Loop: Header=BB0_255 Depth=1
	movq	296(%rsp), %rdi                 # 8-byte Reload
	movl	(%rdi,%r13,4), %r10d
	movl	%r8d, %edi
	cmpl	%eax, %r10d
	jne	.LBB0_273
# %bb.272:                              #   in Loop: Header=BB0_255 Depth=1
	movd	%r9d, %xmm4
	movd	%r10d, %xmm3
	movapd	%xmm1, %xmm2
	jmp	.LBB0_274
	.p2align	4
.LBB0_273:                              #   in Loop: Header=BB0_255 Depth=1
	movq	304(%rsp), %r9                  # 8-byte Reload
	movd	(%r9,%r13,4), %xmm4             # xmm4 = mem[0],zero,zero,zero
	shll	$16, %edi
	movd	%edi, %xmm2
	cvtss2sd	%xmm2, %xmm2
	movq	296(%rsp), %rdi                 # 8-byte Reload
	movd	(%rdi,%r13,4), %xmm3            # xmm3 = mem[0],zero,zero,zero
	subsd	%xmm8, %xmm2
	andpd	.LCPI0_32(%rip), %xmm2
	xorl	%r12d, %r12d
.LBB0_274:                              #   in Loop: Header=BB0_255 Depth=1
	movsd	56(%rsp), %xmm12                # 8-byte Reload
                                        # xmm12 = mem[0],zero
	movsd	136(%rsp), %xmm9                # 8-byte Reload
                                        # xmm9 = mem[0],zero
	movq	1160(%rsp), %rdi                # 8-byte Reload
	movzwl	(%rdi,%r13,2), %edi
	movq	1152(%rsp), %r9                 # 8-byte Reload
	movzwl	(%r9,%r13,2), %r9d
	cmpw	%si, %di
	jne	.LBB0_253
# %bb.275:                              #   in Loop: Header=BB0_255 Depth=1
	cmpw	%r8w, %r9w
	jne	.LBB0_253
# %bb.276:                              #   in Loop: Header=BB0_255 Depth=1
	movq	208(%rsp), %r9                  # 8-byte Reload
	movl	(%r9,%r13,4), %r10d
	cmpl	%edx, %r10d
	jne	.LBB0_252
# %bb.277:                              #   in Loop: Header=BB0_255 Depth=1
	movq	168(%rsp), %r9                  # 8-byte Reload
	movl	(%r9,%r13,4), %r11d
	movl	%r8d, %r9d
	cmpl	%eax, %r11d
	jne	.LBB0_253
# %bb.278:                              #   in Loop: Header=BB0_255 Depth=1
	movd	%r10d, %xmm5
	movd	%r11d, %xmm6
	movapd	%xmm1, %xmm7
	jmp	.LBB0_254
.LBB0_279:
	testb	$1, %r12b
	movq	1352(%rsp), %r14                # 8-byte Reload
	je	.LBB0_547
# %bb.280:
	movsd	336(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	ucomisd	.LCPI0_33(%rip), %xmm0
	movq	1336(%rsp), %rbx                # 8-byte Reload
	movq	1328(%rsp), %r15                # 8-byte Reload
	ja	.LBB0_550
# %bb.281:
	movapd	1248(%rsp), %xmm0               # 16-byte Reload
	ucomisd	.LCPI0_34(%rip), %xmm0
	ja	.LBB0_553
# %bb.282:
	movq	168(%rsp), %rdi                 # 8-byte Reload
	movq	1304(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	296(%rsp), %rdi                 # 8-byte Reload
	movq	1360(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	288(%rsp), %rdi                 # 8-byte Reload
	movq	1384(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	208(%rsp), %rdi                 # 8-byte Reload
	movq	1312(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	304(%rsp), %rdi                 # 8-byte Reload
	movq	1368(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	344(%rsp), %rdi                 # 8-byte Reload
	subq	%rdi, %r15
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
	movq	1152(%rsp), %rdi                # 8-byte Reload
	movq	1320(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	1168(%rsp), %rdi                # 8-byte Reload
	movq	1376(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	360(%rsp), %rdi                 # 8-byte Reload
	subq	%rdi, %rbx
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
	movq	1160(%rsp), %rdi                # 8-byte Reload
	movq	1344(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movq	352(%rsp), %rdi                 # 8-byte Reload
	subq	%rdi, %r14
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
	movq	8(%rsp), %rdi                   # 8-byte Reload
	movq	1392(%rsp), %rsi                # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	movupd	1176(%rsp), %xmm0
	movdqu	1192(%rsp), %xmm1
	movupd	1208(%rsp), %xmm2
	movupd	1224(%rsp), %xmm3
	movapd	%xmm0, 656(%rsp)
	movq	1240(%rsp), %rax
	movq	%rax, 720(%rsp)
	movapd	%xmm3, 704(%rsp)
	movapd	%xmm2, 688(%rsp)
	movdqa	%xmm1, 672(%rsp)
	movq	$0, 656(%rsp)
.Ltmp277:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
.Ltmp278:                               # EH_LABEL
# %bb.283:
	cmpl	$1, %eax
	jne	.LBB0_523
# %bb.284:
.Ltmp279:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp280:                               # EH_LABEL
# %bb.285:
	cmpl	$1, %eax
	jne	.LBB0_523
# %bb.286:
.Ltmp281:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp282:                               # EH_LABEL
# %bb.287:
	cmpl	$1, %eax
	jne	.LBB0_523
# %bb.288:
	movupd	1176(%rsp), %xmm0
	movdqu	1192(%rsp), %xmm1
	movupd	1208(%rsp), %xmm2
	movupd	1224(%rsp), %xmm3
	movapd	%xmm2, 688(%rsp)
	movapd	%xmm3, 704(%rsp)
	movq	1240(%rsp), %rax
	movq	%rax, 720(%rsp)
	movdqa	%xmm1, 672(%rsp)
	movapd	%xmm0, 656(%rsp)
	movq	696(%rsp), %rax
	movq	%rax, 704(%rsp)
.Ltmp289:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
.Ltmp290:                               # EH_LABEL
# %bb.289:
	cmpl	$1, %eax
	jne	.LBB0_526
# %bb.290:
.Ltmp291:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp292:                               # EH_LABEL
# %bb.291:
	cmpl	$1, %eax
	jne	.LBB0_526
# %bb.292:
.Ltmp293:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp294:                               # EH_LABEL
# %bb.293:
	cmpl	$1, %eax
	jne	.LBB0_526
# %bb.294:
	movupd	1176(%rsp), %xmm0
	movdqu	1192(%rsp), %xmm1
	movupd	1208(%rsp), %xmm2
	movupd	1224(%rsp), %xmm3
	movapd	%xmm0, 656(%rsp)
	movapd	%xmm2, 688(%rsp)
	movq	1240(%rsp), %rax
	movq	%rax, 720(%rsp)
	movapd	%xmm3, 704(%rsp)
	movdqa	%xmm1, 672(%rsp)
	movq	656(%rsp), %rax
	movq	%rax, 696(%rsp)
.Ltmp301:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
.Ltmp302:                               # EH_LABEL
# %bb.295:
	cmpl	$1, %eax
	jne	.LBB0_529
# %bb.296:
.Ltmp303:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp304:                               # EH_LABEL
# %bb.297:
	cmpl	$1, %eax
	jne	.LBB0_529
# %bb.298:
.Ltmp305:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp306:                               # EH_LABEL
# %bb.299:
	cmpl	$1, %eax
	jne	.LBB0_529
# %bb.300:
	movq	$0, 736(%rsp)
	movq	$25165824, 744(%rsp)            # imm = 0x1800000
.Ltmp314:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	736(%rsp), %rdi
	movl	$100663296, %esi                # imm = 0x6000000
	callq	hipMalloc@PLT
.Ltmp315:                               # EH_LABEL
# %bb.301:
.Ltmp316:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp317:                               # EH_LABEL
# %bb.302:
	movq	$0, 832(%rsp)
	movq	$1, 840(%rsp)
.Ltmp319:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	832(%rsp), %rdi
	movl	$4, %esi
	callq	hipMalloc@PLT
.Ltmp320:                               # EH_LABEL
# %bb.303:
.Ltmp321:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.139(%rip), %rsi
	movl	%eax, %edi
	movl	$49, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp322:                               # EH_LABEL
# %bb.304:
	movq	736(%rsp), %rdi
	movq	744(%rsp), %rdx
	shlq	$2, %rdx
.Ltmp324:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$165, %esi
	callq	hipMemset@PLT
.Ltmp325:                               # EH_LABEL
# %bb.305:
.Ltmp326:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.27(%rip), %rsi
	movl	%eax, %edi
	movl	$682, %edx                      # imm = 0x2AA
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp327:                               # EH_LABEL
# %bb.306:
	movq	832(%rsp), %rdi
	movq	840(%rsp), %rdx
	shlq	$2, %rdx
.Ltmp328:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipMemset@PLT
.Ltmp329:                               # EH_LABEL
# %bb.307:
.Ltmp330:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.28(%rip), %rsi
	movl	%eax, %edi
	movl	$683, %edx                      # imm = 0x2AB
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp331:                               # EH_LABEL
# %bb.308:
	xorpd	%xmm0, %xmm0
	movapd	%xmm0, 976(%rsp)
	movapd	%xmm0, 960(%rsp)
	movapd	%xmm0, 944(%rsp)
	movapd	%xmm0, 928(%rsp)
	movq	$0, 992(%rsp)
	movapd	%xmm0, 176(%rsp)
	movq	$0, 192(%rsp)
	xorl	%esi, %esi
	xorl	%r12d, %r12d
	movq	$0, 8(%rsp)                     # 8-byte Folded Spill
	xorl	%edx, %edx
	jmp	.LBB0_310
	.p2align	4
.LBB0_309:                              #   in Loop: Header=BB0_310 Depth=1
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%rax, 176(%rsp)
	incq	%rdx
	cmpq	$12, %rdx
	je	.LBB0_387
.LBB0_310:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_312 Depth 2
                                        #       Child Loop BB0_330 Depth 3
                                        #       Child Loop BB0_334 Depth 3
                                        #       Child Loop BB0_336 Depth 3
	movq	%rdx, %rax
	subq	$6, %rax
	cmovbq	%rdx, %rax
	movq	%rax, 288(%rsp)                 # 8-byte Spill
	leaq	(%rax,%rax,2), %rax
	leaq	.L__const.main.orders(%rip), %rcx
	leaq	(%rcx,%rax,4), %rax
	movq	%rax, 344(%rsp)                 # 8-byte Spill
	xorl	%ecx, %ecx
	movq	%rdx, 168(%rsp)                 # 8-byte Spill
	jmp	.LBB0_312
	.p2align	4
.LBB0_311:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rdx, (%r12)
	movq	288(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 8(%r12)
	movq	%rcx, 16(%r12)
	movl	216(%rsp), %eax                 # 4-byte Reload
	movl	%eax, 24(%r12)
	movq	144(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 32(%r12)
	movss	56(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 40(%r12)
	addq	$48, %r12
	movq	%r12, 184(%rsp)
	incq	%rcx
	cmpq	$3, %rcx
	je	.LBB0_309
.LBB0_312:                              #   Parent Loop BB0_310 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_330 Depth 3
                                        #       Child Loop BB0_334 Depth 3
                                        #       Child Loop BB0_336 Depth 3
	movq	%r12, 304(%rsp)                 # 8-byte Spill
	movq	%rsi, 208(%rsp)                 # 8-byte Spill
	movq	344(%rsp), %rax                 # 8-byte Reload
	movq	%rcx, 136(%rsp)                 # 8-byte Spill
	movl	(%rax,%rcx,4), %eax
	movl	%eax, 216(%rsp)                 # 4-byte Spill
	movslq	%eax, %rbx
	leaq	(%rdx,%rbx), %rax
	movq	%rax, 144(%rsp)                 # 8-byte Spill
	movabsq	$-6148914691236517205, %rcx     # imm = 0xAAAAAAAAAAAAAAAB
	mulq	%rcx
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
.Ltmp333:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294968320, %rdi               # imm = 0x100000400
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp334:                               # EH_LABEL
# %bb.313:                              #   in Loop: Header=BB0_312 Depth=2
	testl	%eax, %eax
	jne	.LBB0_316
# %bb.314:                              #   in Loop: Header=BB0_312 Depth=2
	movq	832(%rsp), %rax
	movq	736(%rsp), %rcx
	movq	744(%rsp), %rdx
	movq	%rcx, 1104(%rsp)
	movq	%rax, 920(%rsp)
	movq	%rdx, 912(%rsp)
	leaq	1104(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	920(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	912(%rsp), %rax
	movq	%rax, 80(%rsp)
.Ltmp335:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	752(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	904(%rsp), %rdx
	leaq	896(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp336:                               # EH_LABEL
# %bb.315:                              #   in Loop: Header=BB0_312 Depth=2
	movq	752(%rsp), %rsi
	movl	760(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
.Ltmp337:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm(%rip), %rdi
	leaq	64(%rsp), %r9
	pushq	896(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	912(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp338:                               # EH_LABEL
.LBB0_316:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp339:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp340:                               # EH_LABEL
# %bb.317:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp341:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.29(%rip), %rsi
	movl	$695, %edx                      # imm = 0x2B7
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp342:                               # EH_LABEL
# %bb.318:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp343:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp344:                               # EH_LABEL
# %bb.319:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp345:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.23(%rip), %rsi
	movl	$696, %edx                      # imm = 0x2B8
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp346:                               # EH_LABEL
# %bb.320:                              #   in Loop: Header=BB0_312 Depth=2
	leaq	(%rbx,%rbx,2), %rcx
	movq	%rcx, %r12
	shlq	$5, %r12
	movq	384(%rsp,%r12), %r15
	leaq	2(%r15), %r14
	movq	%r14, %rax
	shrq	$62, %rax
	jne	.LBB0_499
# %bb.321:                              #   in Loop: Header=BB0_312 Depth=2
	movq	368(%rsp,%r12), %rbp
	testq	%r14, %r14
	movq	%rcx, 296(%rsp)                 # 8-byte Spill
	je	.LBB0_328
# %bb.322:                              #   in Loop: Header=BB0_312 Depth=2
	leaq	(%r14,%r14), %rdi
.Ltmp347:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp348:                               # EH_LABEL
# %bb.323:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rax, %r13
	movw	$0, (%rax)
	leaq	2(%rax), %rbx
	incq	%r15
	je	.LBB0_325
# %bb.324:                              #   in Loop: Header=BB0_312 Depth=2
	leaq	(%r15,%r15), %rdx
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	(%rbx,%r15,2), %rbx
.LBB0_325:                              #   in Loop: Header=BB0_312 Depth=2
	movapd	.LCPI0_35(%rip), %xmm0          # xmm0 = [32602,32602,32602,32602,32602,32602,32602,32602]
	movabsq	$9176787217381228378, %rdi      # imm = 0x7F5A7F5A7F5A7F5A
	movq	%rbx, %rdx
	subq	%r13, %rdx
	addq	$-2, %rdx
	movq	%r13, %rax
	cmpq	$6, %rdx
	jb	.LBB0_336
# %bb.326:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rdx, %rcx
	shrq	%rcx
	incq	%rcx
	cmpq	$30, %rdx
	jae	.LBB0_329
# %bb.327:                              #   in Loop: Header=BB0_312 Depth=2
	xorl	%edx, %edx
	jmp	.LBB0_333
	.p2align	4
.LBB0_328:                              #   in Loop: Header=BB0_312 Depth=2
	xorl	%ebx, %ebx
	xorl	%r13d, %r13d
	xorl	%r14d, %r14d
	jmp	.LBB0_338
.LBB0_329:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rcx, %rdx
	andq	$-16, %rdx
	leaq	(,%rdx,2), %rax
	addq	%r13, %rax
	xorl	%esi, %esi
	.p2align	4
.LBB0_330:                              #   Parent Loop BB0_310 Depth=1
                                        #     Parent Loop BB0_312 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movupd	%xmm0, (%r13,%rsi,2)
	movupd	%xmm0, 16(%r13,%rsi,2)
	addq	$16, %rsi
	cmpq	%rsi, %rdx
	jne	.LBB0_330
# %bb.331:                              #   in Loop: Header=BB0_312 Depth=2
	cmpq	%rdx, %rcx
	je	.LBB0_337
# %bb.332:                              #   in Loop: Header=BB0_312 Depth=2
	testb	$12, %cl
	je	.LBB0_336
.LBB0_333:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rcx, %rsi
	andq	$-4, %rsi
	leaq	(,%rsi,2), %rax
	addq	%r13, %rax
	.p2align	4
.LBB0_334:                              #   Parent Loop BB0_310 Depth=1
                                        #     Parent Loop BB0_312 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rdi, (%r13,%rdx,2)
	addq	$4, %rdx
	cmpq	%rdx, %rsi
	jne	.LBB0_334
# %bb.335:                              #   in Loop: Header=BB0_312 Depth=2
	cmpq	%rsi, %rcx
	je	.LBB0_337
	.p2align	4
.LBB0_336:                              #   Parent Loop BB0_310 Depth=1
                                        #     Parent Loop BB0_312 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movw	$32602, (%rax)                  # imm = 0x7F5A
	addq	$2, %rax
	cmpq	%rbx, %rax
	jne	.LBB0_336
.LBB0_337:                              #   in Loop: Header=BB0_312 Depth=2
	leaq	(,%r14,2), %r14
	addq	%r13, %r14
.LBB0_338:                              #   in Loop: Header=BB0_312 Depth=2
	subq	%r13, %rbx
.Ltmp349:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r13, %rsi
	movq	%rbx, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp350:                               # EH_LABEL
# %bb.339:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp351:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.140(%rip), %rsi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp352:                               # EH_LABEL
# %bb.340:                              #   in Loop: Header=BB0_312 Depth=2
	testq	%r13, %r13
	je	.LBB0_342
# %bb.341:                              #   in Loop: Header=BB0_312 Depth=2
	subq	%r13, %r14
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
.LBB0_342:                              #   in Loop: Header=BB0_312 Depth=2
	addq	%rsp, %r12
	addq	$368, %r12                      # imm = 0x170
	movq	24(%r12), %rdi
	movq	40(%r12), %rsi
.Ltmp354:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Ltmp355:                               # EH_LABEL
# %bb.343:                              #   in Loop: Header=BB0_312 Depth=2
	movq	48(%r12), %rdi
	movq	64(%r12), %rsi
.Ltmp356:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp357:                               # EH_LABEL
# %bb.344:                              #   in Loop: Header=BB0_312 Depth=2
	movq	72(%r12), %rdi
	movq	88(%r12), %rsi
.Ltmp358:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Ltmp359:                               # EH_LABEL
# %bb.345:                              #   in Loop: Header=BB0_312 Depth=2
	movq	56(%rsp), %rax                  # 8-byte Reload
	shrq	%rax
	leaq	(%rax,%rax,2), %rax
	movq	144(%rsp), %rcx                 # 8-byte Reload
	subq	%rax, %rcx
	movq	%rcx, 144(%rsp)                 # 8-byte Spill
	movq	%rcx, %rax
	shlq	$4, %rax
	movq	880(%rsp), %rcx
	movq	1056(%rsp,%rax), %rdx
	movq	1008(%rsp,%rax), %rax
	movq	864(%rsp), %r13
	movq	848(%rsp), %rbp
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movq	%rcx, 752(%rsp)
	movq	%rdx, 360(%rsp)                 # 8-byte Spill
	movq	%rdx, 760(%rsp)
	movq	%rax, 352(%rsp)                 # 8-byte Spill
	movq	%rax, 768(%rsp)
	movq	%r13, 776(%rsp)
	movq	%rbp, 784(%rsp)
	movq	(%r12), %r15
	addq	$2, %r15
	movq	%r15, 792(%rsp)
	movq	24(%r12), %rbx
	addq	$2, %rbx
	movq	%rbx, 800(%rsp)
	movq	48(%r12), %r14
	addq	$4, %r14
	movq	%r14, 808(%rsp)
	movq	72(%r12), %r12
	addq	$4, %r12
	movq	%r12, 816(%rsp)
	movq	$0, 312(%rsp)
.Ltmp361:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	312(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp362:                               # EH_LABEL
# %bb.346:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp363:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.146(%rip), %rsi
	movl	$68, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp364:                               # EH_LABEL
# %bb.347:                              #   in Loop: Header=BB0_312 Depth=2
	movq	$0, 224(%rsp)
.Ltmp366:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	224(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp367:                               # EH_LABEL
# %bb.348:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp368:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.146(%rip), %rsi
	movl	$68, %edx
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp369:                               # EH_LABEL
# %bb.349:                              #   in Loop: Header=BB0_312 Depth=2
	movq	312(%rsp), %rdi
.Ltmp371:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp372:                               # EH_LABEL
# %bb.350:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp373:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.141(%rip), %rsi
	movl	$475, %edx                      # imm = 0x1DB
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp374:                               # EH_LABEL
# %bb.351:                              #   in Loop: Header=BB0_312 Depth=2
	movl	216(%rsp), %eax                 # 4-byte Reload
	cmpl	$1, %eax
	je	.LBB0_354
# %bb.352:                              #   in Loop: Header=BB0_312 Depth=2
	testl	%eax, %eax
	jne	.LBB0_355
# %bb.353:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp377:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	752(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
.Ltmp378:                               # EH_LABEL
	jmp	.LBB0_361
	.p2align	4
.LBB0_354:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp375:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	752(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
.Ltmp376:                               # EH_LABEL
	jmp	.LBB0_361
	.p2align	4
.LBB0_355:                              #   in Loop: Header=BB0_312 Depth=2
	.cfi_escape 0x2e, 0x00
	leaq	752(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE
	movl	%eax, %ecx
	movl	$1, %eax
	testb	%cl, %cl
	je	.LBB0_361
# %bb.356:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp379:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967344, %rdi               # imm = 0x100000030
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp380:                               # EH_LABEL
# %bb.357:                              #   in Loop: Header=BB0_312 Depth=2
	testl	%eax, %eax
	jne	.LBB0_360
# %bb.358:                              #   in Loop: Header=BB0_312 Depth=2
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 920(%rsp)
	movq	360(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 912(%rsp)
	movq	352(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 904(%rsp)
	movq	%r13, 896(%rsp)
	movq	%rbp, 1448(%rsp)
	movq	%r15, 1440(%rsp)
	movq	%rbx, 1432(%rsp)
	movq	%r14, 1424(%rsp)
	movq	%r12, 1416(%rsp)
	leaq	920(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	912(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	904(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	896(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	1448(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	1440(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	1432(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	1424(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	1416(%rsp), %rax
	movq	%rax, 128(%rsp)
.Ltmp381:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	1104(%rsp), %rsi
	leaq	1408(%rsp), %rdx
	leaq	1400(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp382:                               # EH_LABEL
# %bb.359:                              #   in Loop: Header=BB0_312 Depth=2
	movq	16(%rsp), %rsi
	movl	24(%rsp), %edx
	movq	1104(%rsp), %rcx
	movl	1112(%rsp), %r8d
.Ltmp383:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_(%rip), %rdi
	leaq	64(%rsp), %r9
	pushq	1400(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	1416(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp384:                               # EH_LABEL
.LBB0_360:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp385:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp386:                               # EH_LABEL
	.p2align	4
.LBB0_361:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp387:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.142(%rip), %rsi
	movl	$476, %edx                      # imm = 0x1DC
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp388:                               # EH_LABEL
# %bb.362:                              #   in Loop: Header=BB0_312 Depth=2
	movq	224(%rsp), %rdi
.Ltmp389:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp390:                               # EH_LABEL
# %bb.363:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp391:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.143(%rip), %rsi
	movl	$477, %edx                      # imm = 0x1DD
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp392:                               # EH_LABEL
# %bb.364:                              #   in Loop: Header=BB0_312 Depth=2
	movq	224(%rsp), %rdi
.Ltmp393:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp394:                               # EH_LABEL
# %bb.365:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp395:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.144(%rip), %rsi
	movl	$478, %edx                      # imm = 0x1DE
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp396:                               # EH_LABEL
# %bb.366:                              #   in Loop: Header=BB0_312 Depth=2
	movl	$0, 64(%rsp)
	movq	312(%rsp), %rsi
	movq	224(%rsp), %rdx
.Ltmp398:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp399:                               # EH_LABEL
# %bb.367:                              #   in Loop: Header=BB0_312 Depth=2
.Ltmp400:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	leaq	.L.str.145(%rip), %rsi
	movl	$480, %edx                      # imm = 0x1E0
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp401:                               # EH_LABEL
# %bb.368:                              #   in Loop: Header=BB0_312 Depth=2
	movss	64(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 56(%rsp)                 # 4-byte Spill
	movq	224(%rsp), %rdi
.Ltmp409:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp410:                               # EH_LABEL
# %bb.369:                              #   in Loop: Header=BB0_312 Depth=2
	movq	312(%rsp), %rdi
.Ltmp412:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp413:                               # EH_LABEL
# %bb.370:                              #   in Loop: Header=BB0_312 Depth=2
	movq	296(%rsp), %rax                 # 8-byte Reload
	leaq	(%rsp,%rax,8), %r15
	addq	$928, %r15                      # imm = 0x3A0
	movq	936(%rsp,%rax,8), %rbx
	cmpq	944(%rsp,%rax,8), %rbx
	je	.LBB0_372
# %bb.371:                              #   in Loop: Header=BB0_312 Depth=2
	movss	56(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rbx)
	addq	$4, %rbx
	movq	%rbx, 8(%r15)
	jmp	.LBB0_379
	.p2align	4
.LBB0_372:                              #   in Loop: Header=BB0_312 Depth=2
	movq	(%r15), %r14
	subq	%r14, %rbx
	movabsq	$9223372036854775773, %rax      # imm = 0x7FFFFFFFFFFFFFDD
	addq	$31, %rax
	cmpq	%rax, %rbx
	je	.LBB0_503
# %bb.373:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rbx, %rax
	sarq	$2, %rax
	cmpq	$1, %rax
	movq	%rax, %rcx
	adcq	$0, %rcx
	leaq	(%rcx,%rax), %r12
	movabsq	$2305843009213693951, %rdx      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rdx, %r12
	cmovaeq	%rdx, %r12
	addq	%rax, %rcx
	cmovbq	%rdx, %r12
	leaq	(,%r12,4), %rdi
.Ltmp415:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp416:                               # EH_LABEL
# %bb.374:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rax, %r13
	movss	56(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%rbx)
	testq	%rbx, %rbx
	jle	.LBB0_376
# %bb.375:                              #   in Loop: Header=BB0_312 Depth=2
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r14, %rsi
	movq	%rbx, %rdx
	callq	memmove@PLT
.LBB0_376:                              #   in Loop: Header=BB0_312 Depth=2
	testq	%r14, %r14
	je	.LBB0_378
# %bb.377:                              #   in Loop: Header=BB0_312 Depth=2
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
.LBB0_378:                              #   in Loop: Header=BB0_312 Depth=2
	addq	%r13, %rbx
	addq	$4, %rbx
	movq	%r13, (%r15)
	movq	%rbx, 8(%r15)
	leaq	(,%r12,4), %rax
	addq	%r13, %rax
	movq	%rax, 16(%r15)
.LBB0_379:                              #   in Loop: Header=BB0_312 Depth=2
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	304(%rsp), %r12                 # 8-byte Reload
	cmpq	%rsi, %r12
	movq	168(%rsp), %rdx                 # 8-byte Reload
	movq	136(%rsp), %rcx                 # 8-byte Reload
	jne	.LBB0_311
# %bb.380:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rsi, %r12
	movq	8(%rsp), %rcx                   # 8-byte Reload
	subq	%rcx, %r12
	movabsq	$9223372036854775773, %rax      # imm = 0x7FFFFFFFFFFFFFDD
	addq	$3, %rax
	cmpq	%rax, %r12
	je	.LBB0_505
# %bb.381:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%r12, %rax
	sarq	$4, %rax
	movabsq	$-6148914691236517205, %rdx     # imm = 0xAAAAAAAAAAAAAAAB
	imulq	%rdx, %rax
	cmpq	%rcx, %rsi
	movq	%rax, %rcx
	movl	$1, %edx
	cmoveq	%rdx, %rcx
	leaq	(%rcx,%rax), %r14
	movabsq	$192153584101141162, %rdx       # imm = 0x2AAAAAAAAAAAAAA
	cmpq	%rdx, %r14
	cmovaeq	%rdx, %r14
	addq	%rax, %rcx
	cmovbq	%rdx, %r14
	movq	%r14, %rax
	shlq	$4, %rax
	leaq	(%rax,%rax,2), %rdi
.Ltmp418:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp419:                               # EH_LABEL
	movl	216(%rsp), %ecx                 # 4-byte Reload
# %bb.382:                              #   in Loop: Header=BB0_312 Depth=2
	movq	%rax, %rbx
	movq	168(%rsp), %rax                 # 8-byte Reload
	movq	%rax, (%rbx,%r12)
	movq	288(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 8(%rbx,%r12)
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 16(%rbx,%r12)
	movl	%ecx, 24(%rbx,%r12)
	movq	144(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 32(%rbx,%r12)
	movss	56(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 40(%rbx,%r12)
	testq	%r12, %r12
	movq	8(%rsp), %r15                   # 8-byte Reload
	jle	.LBB0_384
# %bb.383:                              #   in Loop: Header=BB0_312 Depth=2
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	movq	%r15, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
.LBB0_384:                              #   in Loop: Header=BB0_312 Depth=2
	testq	%r15, %r15
	je	.LBB0_386
# %bb.385:                              #   in Loop: Header=BB0_312 Depth=2
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB0_386:                              #   in Loop: Header=BB0_312 Depth=2
	addq	%rbx, %r12
	addq	$48, %r12
	movq	%r12, 184(%rsp)
	leaq	(%r14,%r14,2), %rsi
	shlq	$4, %rsi
	addq	%rbx, %rsi
	movq	%rsi, 192(%rsp)
	movq	%rbx, 8(%rsp)                   # 8-byte Spill
	movq	168(%rsp), %rdx                 # 8-byte Reload
	movq	136(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	cmpq	$3, %rcx
	jne	.LBB0_312
	jmp	.LBB0_309
.LBB0_387:
	movq	1120(%rsp), %rsi
.Ltmp421:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc
.Ltmp422:                               # EH_LABEL
# %bb.388:
	movq	64(%rsp), %rdi
	cmpq	$4, 72(%rsp)
	movq	328(%rsp), %rbp                 # 8-byte Reload
	jne	.LBB0_390
# %bb.389:
	cmpl	$1869903201, (%rdi)             # imm = 0x6F747561
	sete	%bl
	leaq	80(%rsp), %r12
	cmpq	%r12, %rdi
	jne	.LBB0_391
	jmp	.LBB0_392
.LBB0_390:
	xorl	%ebx, %ebx
	leaq	80(%rsp), %r12
	cmpq	%r12, %rdi
	je	.LBB0_392
.LBB0_391:
	movq	80(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_392:
	testb	%bl, %bl
	je	.LBB0_556
# %bb.393:
	movq	16(%rbp), %rbx
	movq	%r12, 64(%rsp)
	testq	%rbx, %rbx
	je	.LBB0_559
# %bb.394:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	movq	%r12, %r15
	cmpq	$16, %rax
	jb	.LBB0_399
# %bb.395:
	testq	%r14, %r14
	js	.LBB0_569
# %bb.396:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB0_511
# %bb.397:
.Ltmp430:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp431:                               # EH_LABEL
# %bb.398:
	movq	%rax, %r15
	movq	%rax, 64(%rsp)
	movq	%r14, 80(%rsp)
.LBB0_399:
	testq	%r14, %r14
	je	.LBB0_403
# %bb.400:
	cmpq	$1, %r14
	jne	.LBB0_402
# %bb.401:
	movzbl	(%rbx), %eax
	movb	%al, (%r15)
	jmp	.LBB0_403
.LBB0_402:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB0_403:
	movq	%r14, 72(%rsp)
	movb	$0, (%r15,%r14)
	movq	32(%rbp), %rbx
	leaq	768(%rsp), %r13
	movq	%r13, 752(%rsp)
	testq	%rbx, %rbx
	je	.LBB0_561
# %bb.404:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	movq	%r13, %r15
	cmpq	$16, %rax
	jb	.LBB0_409
# %bb.405:
	testq	%r14, %r14
	js	.LBB0_571
# %bb.406:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB0_513
# %bb.407:
.Ltmp432:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp433:                               # EH_LABEL
# %bb.408:
	movq	%rax, %r15
	movq	%rax, 752(%rsp)
	movq	%r14, 768(%rsp)
.LBB0_409:
	testq	%r14, %r14
	je	.LBB0_413
# %bb.410:
	cmpq	$1, %r14
	jne	.LBB0_412
# %bb.411:
	movzbl	(%rbx), %eax
	movb	%al, (%r15)
	jmp	.LBB0_413
.LBB0_412:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB0_413:
	movq	%r14, 760(%rsp)
	movb	$0, (%r15,%r14)
	movq	48(%rbp), %rbx
	leaq	32(%rsp), %rbp
	movq	%rbp, 16(%rsp)
	testq	%rbx, %rbx
	je	.LBB0_563
# %bb.414:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	movq	%rbp, %r15
	cmpq	$16, %rax
	jb	.LBB0_419
# %bb.415:
	testq	%r14, %r14
	js	.LBB0_573
# %bb.416:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB0_515
# %bb.417:
.Ltmp434:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp435:                               # EH_LABEL
# %bb.418:
	movq	%rax, %r15
	movq	%rax, 16(%rsp)
	movq	%r14, 32(%rsp)
.LBB0_419:
	testq	%r14, %r14
	je	.LBB0_423
# %bb.420:
	cmpq	$1, %r14
	jne	.LBB0_422
# %bb.421:
	movzbl	(%rbx), %eax
	movb	%al, (%r15)
	jmp	.LBB0_423
.LBB0_422:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB0_423:
	movq	%r14, 24(%rsp)
	movb	$0, (%r15,%r14)
	movl	320(%rsp), %eax
.Ltmp436:                               # EH_LABEL
	.cfi_escape 0x2e, 0x20
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	184(%rsp), %r10
	leaq	936(%rsp), %r11
	leaq	72(%rsp), %rdi
	leaq	760(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	1280(%rsp), %rcx
	leaq	1128(%rsp), %r8
	leaq	1648(%rsp), %r9
	movapd	1256(%rsp), %xmm0               # 16-byte Reload
	movq	344(%rsp), %xmm1                # 8-byte Folded Reload
                                        # xmm1 = mem[0],zero
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	_ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
.Ltmp437:                               # EH_LABEL
# %bb.424:
	movq	16(%rsp), %rdi
	cmpq	%rbp, %rdi
	je	.LBB0_426
# %bb.425:
	movq	32(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_426:
	movq	752(%rsp), %rdi
	cmpq	%r13, %rdi
	movq	328(%rsp), %r14                 # 8-byte Reload
	je	.LBB0_428
# %bb.427:
	movq	768(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_428:
	movq	64(%rsp), %rdi
	cmpq	%r12, %rdi
	je	.LBB0_430
# %bb.429:
	movq	80(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_430:
.Ltmp439:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rbx
	leaq	.L.str.31(%rip), %rsi
	movl	$11, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp440:                               # EH_LABEL
# %bb.431:
	movq	16(%r14), %r14
	testq	%r14, %r14
	je	.LBB0_433
# %bb.432:
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	strlen@PLT
.Ltmp441:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movq	%r14, %rsi
	movq	%rax, %rdx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp442:                               # EH_LABEL
	jmp	.LBB0_434
.LBB0_433:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	leaq	(%rbx,%rax), %rdi
	movl	32(%rbx,%rax), %esi
	orl	$1, %esi
.Ltmp443:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
.Ltmp444:                               # EH_LABEL
.LBB0_434:
	movb	$10, 64(%rsp)
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	cmpq	$0, 16(%rbx,%rax)
	je	.LBB0_436
# %bb.435:
.Ltmp445:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	64(%rsp), %rsi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp446:                               # EH_LABEL
	jmp	.LBB0_437
.LBB0_436:
.Ltmp447:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movl	$10, %esi
	callq	_ZNSo3putEc@PLT
.Ltmp448:                               # EH_LABEL
.LBB0_437:
	movq	8(%rsp), %rdi                   # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB0_439
# %bb.438:
	movq	192(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_439:
	movq	976(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_441
# %bb.440:
	movq	992(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_441:
	movq	952(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_443
# %bb.442:
	movq	968(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_443:
	movq	928(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_445
# %bb.444:
	movq	944(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_445:
	movq	832(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_447
# %bb.446:
.Ltmp450:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp451:                               # EH_LABEL
.LBB0_447:
	movq	736(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_449
# %bb.448:
.Ltmp453:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp454:                               # EH_LABEL
.LBB0_449:
	movq	632(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_451
# %bb.450:
.Ltmp456:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp457:                               # EH_LABEL
.LBB0_451:
	movq	608(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_453
# %bb.452:
.Ltmp459:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp460:                               # EH_LABEL
.LBB0_453:
	movq	584(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_455
# %bb.454:
.Ltmp462:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp463:                               # EH_LABEL
.LBB0_455:
	movq	560(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_457
# %bb.456:
.Ltmp465:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp466:                               # EH_LABEL
.LBB0_457:
	movq	536(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_459
# %bb.458:
.Ltmp468:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp469:                               # EH_LABEL
.LBB0_459:
	movq	512(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_461
# %bb.460:
.Ltmp471:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp472:                               # EH_LABEL
.LBB0_461:
	movq	488(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_463
# %bb.462:
.Ltmp474:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp475:                               # EH_LABEL
.LBB0_463:
	movq	464(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_465
# %bb.464:
.Ltmp477:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp478:                               # EH_LABEL
.LBB0_465:
	movq	440(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_467
# %bb.466:
.Ltmp480:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp481:                               # EH_LABEL
.LBB0_467:
	movq	416(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_469
# %bb.468:
.Ltmp483:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp484:                               # EH_LABEL
.LBB0_469:
	movq	392(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_471
# %bb.470:
.Ltmp486:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp487:                               # EH_LABEL
.LBB0_471:
	movq	368(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_473
# %bb.472:
.Ltmp489:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp490:                               # EH_LABEL
.LBB0_473:
	movq	1040(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_475
# %bb.474:
.Ltmp492:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp493:                               # EH_LABEL
.LBB0_475:
	movq	1024(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_477
# %bb.476:
.Ltmp494:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp495:                               # EH_LABEL
.LBB0_477:
	movq	1008(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_479
# %bb.478:
.Ltmp496:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp497:                               # EH_LABEL
.LBB0_479:
	movq	1088(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_481
# %bb.480:
.Ltmp499:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp500:                               # EH_LABEL
.LBB0_481:
	movq	1072(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_483
# %bb.482:
.Ltmp501:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp502:                               # EH_LABEL
.LBB0_483:
	movq	1056(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_485
# %bb.484:
.Ltmp503:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp504:                               # EH_LABEL
.LBB0_485:
	movq	848(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_487
# %bb.486:
.Ltmp506:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp507:                               # EH_LABEL
.LBB0_487:
	movq	864(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_489
# %bb.488:
.Ltmp509:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp510:                               # EH_LABEL
.LBB0_489:
	movq	880(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_491
# %bb.490:
.Ltmp512:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp513:                               # EH_LABEL
.LBB0_491:
	.cfi_escape 0x2e, 0x00
	movl	$192, %esi
	movq	264(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$192, %esi
	movq	272(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	152(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	160(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$10240, %esi                    # imm = 0x2800
	movq	280(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	movq	1272(%rsp), %rdi
	leaq	1288(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_493
# %bb.492:
	movq	1288(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_493:
	movq	1120(%rsp), %rdi
	leaq	1136(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_495
# %bb.494:
	movq	1136(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_495:
	movq	232(%rsp), %rdi
	leaq	248(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_497
# %bb.496:
	movq	248(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_497:
	xorl	%ebx, %ebx
.LBB0_498:
	movl	%ebx, %eax
	addq	$3112, %rsp                     # imm = 0xC28
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
.LBB0_499:
	.cfi_def_cfa_offset 3168
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%rax, 176(%rsp)
.Ltmp542:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.40(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp543:                               # EH_LABEL
# %bb.500:
.LBB0_501:
.Ltmp610:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp611:                               # EH_LABEL
# %bb.502:
.LBB0_503:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%rax, 176(%rsp)
.Ltmp539:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.47(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp540:                               # EH_LABEL
# %bb.504:
.LBB0_505:
	movq	%rcx, 176(%rsp)
.Ltmp536:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.47(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp537:                               # EH_LABEL
# %bb.506:
.LBB0_507:
.Ltmp576:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp577:                               # EH_LABEL
# %bb.508:
.LBB0_509:
.Ltmp560:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp561:                               # EH_LABEL
# %bb.510:
.LBB0_511:
.Ltmp529:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp530:                               # EH_LABEL
# %bb.512:
.LBB0_513:
.Ltmp522:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp523:                               # EH_LABEL
# %bb.514:
.LBB0_515:
.Ltmp515:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp516:                               # EH_LABEL
# %bb.516:
.LBB0_517:
.Ltmp603:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp604:                               # EH_LABEL
# %bb.518:
.LBB0_519:
.Ltmp571:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp572:                               # EH_LABEL
# %bb.520:
.LBB0_521:
.Ltmp596:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp597:                               # EH_LABEL
# %bb.522:
.LBB0_523:
.Ltmp283:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.24(%rip), %rsi
	leaq	64(%rsp), %rdi
	leaq	928(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp284:                               # EH_LABEL
# %bb.524:
.Ltmp286:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp287:                               # EH_LABEL
# %bb.525:
.LBB0_526:
.Ltmp295:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.25(%rip), %rsi
	leaq	64(%rsp), %rdi
	leaq	928(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp296:                               # EH_LABEL
# %bb.527:
.Ltmp298:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp299:                               # EH_LABEL
# %bb.528:
.LBB0_529:
.Ltmp308:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.26(%rip), %rsi
	leaq	64(%rsp), %rdi
	leaq	928(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp309:                               # EH_LABEL
# %bb.530:
.Ltmp311:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp312:                               # EH_LABEL
# %bb.531:
.LBB0_532:
.Ltmp614:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.33(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.Ltmp615:                               # EH_LABEL
# %bb.533:
.LBB0_534:
.Ltmp42:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.18(%rip), %rsi
	leaq	368(%rsp), %rdi
	leaq	64(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp43:                                # EH_LABEL
# %bb.535:
.Ltmp45:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	368(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp46:                                # EH_LABEL
# %bb.536:
.LBB0_537:
.Ltmp612:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp613:                               # EH_LABEL
# %bb.538:
.LBB0_539:
.Ltmp581:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.7(%rip), %rsi
	leaq	1640(%rsp), %rdi
	leaq	368(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp582:                               # EH_LABEL
# %bb.540:
.Ltmp584:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	1640(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp585:                               # EH_LABEL
# %bb.541:
.LBB0_542:
.Ltmp565:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.12(%rip), %rsi
	leaq	368(%rsp), %rdi
	leaq	64(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp566:                               # EH_LABEL
# %bb.543:
.Ltmp568:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	368(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp569:                               # EH_LABEL
# %bb.544:
.LBB0_545:
.Ltmp557:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.39(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp558:                               # EH_LABEL
# %bb.546:
.LBB0_547:
.Ltmp271:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.41(%rip), %rsi
	leaq	16(%rsp), %rdi
	leaq	176(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp272:                               # EH_LABEL
# %bb.548:
.Ltmp274:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp275:                               # EH_LABEL
# %bb.549:
.LBB0_550:
.Ltmp551:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.42(%rip), %rsi
	leaq	16(%rsp), %rdi
	leaq	176(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp552:                               # EH_LABEL
# %bb.551:
.Ltmp554:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp555:                               # EH_LABEL
# %bb.552:
.LBB0_553:
.Ltmp545:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.43(%rip), %rsi
	leaq	16(%rsp), %rdi
	leaq	176(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp546:                               # EH_LABEL
# %bb.554:
.Ltmp548:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp549:                               # EH_LABEL
# %bb.555:
.LBB0_556:
.Ltmp424:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.30(%rip), %rsi
	leaq	64(%rsp), %rdi
	leaq	752(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp425:                               # EH_LABEL
# %bb.557:
.Ltmp427:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp428:                               # EH_LABEL
# %bb.558:
.LBB0_559:
.Ltmp533:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.33(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.Ltmp534:                               # EH_LABEL
# %bb.560:
.LBB0_561:
.Ltmp526:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.33(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.Ltmp527:                               # EH_LABEL
# %bb.562:
.LBB0_563:
.Ltmp519:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.33(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.Ltmp520:                               # EH_LABEL
# %bb.564:
.LBB0_565:
.Ltmp578:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp579:                               # EH_LABEL
# %bb.566:
.LBB0_567:
.Ltmp562:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp563:                               # EH_LABEL
# %bb.568:
.LBB0_569:
.Ltmp531:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp532:                               # EH_LABEL
# %bb.570:
.LBB0_571:
.Ltmp524:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp525:                               # EH_LABEL
# %bb.572:
.LBB0_573:
.Ltmp517:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp518:                               # EH_LABEL
# %bb.574:
.LBB0_575:
.Ltmp607:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.33(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.Ltmp608:                               # EH_LABEL
# %bb.576:
.LBB0_577:
.Ltmp605:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp606:                               # EH_LABEL
# %bb.578:
.LBB0_579:
.Ltmp573:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp574:                               # EH_LABEL
# %bb.580:
.LBB0_581:
.Ltmp600:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.33(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.Ltmp601:                               # EH_LABEL
# %bb.582:
.LBB0_583:
.Ltmp598:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp599:                               # EH_LABEL
# %bb.584:
.LBB0_585:
.Ltmp429:                               # EH_LABEL
	jmp	.LBB0_700
.LBB0_586:
.Ltmp426:                               # EH_LABEL
	jmp	.LBB0_715
.LBB0_587:
.Ltmp550:                               # EH_LABEL
	jmp	.LBB0_592
.LBB0_588:
.Ltmp547:                               # EH_LABEL
	jmp	.LBB0_660
.LBB0_589:
.Ltmp556:                               # EH_LABEL
	jmp	.LBB0_592
.LBB0_590:
.Ltmp553:                               # EH_LABEL
	jmp	.LBB0_660
.LBB0_591:
.Ltmp276:                               # EH_LABEL
.LBB0_592:
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_661
# %bb.593:
	movq	32(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB0_661
.LBB0_594:
.Ltmp273:                               # EH_LABEL
	jmp	.LBB0_660
.LBB0_595:
.Ltmp33:                                # EH_LABEL
	jmp	.LBB0_631
.LBB0_596:
.Ltmp570:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	368(%rsp), %rdi
	leaq	384(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB0_692
	jmp	.LBB0_753
.LBB0_597:
.Ltmp567:                               # EH_LABEL
	jmp	.LBB0_711
.LBB0_598:
.Ltmp586:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	1640(%rsp), %rdi
	leaq	1656(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB0_695
	jmp	.LBB0_753
.LBB0_599:
.Ltmp583:                               # EH_LABEL
	jmp	.LBB0_711
.LBB0_600:
.Ltmp514:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_601:
.Ltmp511:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_602:
.Ltmp508:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_603:
.Ltmp491:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_604:
.Ltmp488:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_605:
.Ltmp485:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_606:
.Ltmp482:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_607:
.Ltmp479:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_608:
.Ltmp476:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_609:
.Ltmp473:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_610:
.Ltmp470:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_611:
.Ltmp467:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_612:
.Ltmp464:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_613:
.Ltmp461:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_614:
.Ltmp458:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_615:
.Ltmp455:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_616:
.Ltmp452:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_617:
.Ltmp438:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	16(%rsp), %rdi
	cmpq	%rbp, %rdi
	je	.LBB0_697
# %bb.618:
	movq	32(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB0_697
.LBB0_619:
.Ltmp423:                               # EH_LABEL
	jmp	.LBB0_715
.LBB0_620:
.Ltmp67:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_741
.LBB0_621:
.Ltmp64:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_742
.LBB0_622:
.Ltmp61:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_743
.LBB0_623:
.Ltmp58:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_744
.LBB0_624:
.Ltmp55:                                # EH_LABEL
	jmp	.LBB0_652
.LBB0_625:
.Ltmp47:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %r14
	movq	368(%rsp), %rdi
	cmpq	%rbx, %rdi
	je	.LBB0_627
# %bb.626:
	movq	384(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_627:
	movq	%r14, %rbx
	jmp	.LBB0_745
.LBB0_628:
.Ltmp44:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_745
.LBB0_629:
.Ltmp41:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_747
.LBB0_630:
.Ltmp36:                                # EH_LABEL
.LBB0_631:
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_750
.LBB0_632:
.Ltmp313:                               # EH_LABEL
	jmp	.LBB0_637
.LBB0_633:
.Ltmp310:                               # EH_LABEL
	jmp	.LBB0_687
.LBB0_634:
.Ltmp300:                               # EH_LABEL
	jmp	.LBB0_637
.LBB0_635:
.Ltmp297:                               # EH_LABEL
	jmp	.LBB0_687
.LBB0_636:
.Ltmp288:                               # EH_LABEL
.LBB0_637:
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	64(%rsp), %rdi
	leaq	80(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_734
# %bb.638:
	movq	80(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB0_734
.LBB0_639:
.Ltmp285:                               # EH_LABEL
	jmp	.LBB0_687
.LBB0_640:
.Ltmp559:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	368(%rsp), %rdi
	leaq	384(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_750
# %bb.641:
	movq	384(%rsp), %rsi
	jmp	.LBB0_749
.LBB0_642:
.Ltmp505:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_643:
.Ltmp498:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_644:
.Ltmp323:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_733
.LBB0_645:
.Ltmp318:                               # EH_LABEL
	jmp	.LBB0_687
.LBB0_646:
.Ltmp114:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_736
.LBB0_647:
.Ltmp100:                               # EH_LABEL
	jmp	.LBB0_682
.LBB0_648:
.Ltmp82:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_738
.LBB0_649:
.Ltmp77:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_739
.LBB0_650:
.Ltmp72:                                # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_740
.LBB0_651:
.Ltmp52:                                # EH_LABEL
.LBB0_652:
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_745
.LBB0_653:
.Ltmp28:                                # EH_LABEL
	jmp	.LBB0_711
.LBB0_654:
.Ltmp19:                                # EH_LABEL
	jmp	.LBB0_711
.LBB0_655:
.Ltmp449:                               # EH_LABEL
	jmp	.LBB0_715
.LBB0_656:
.Ltmp270:                               # EH_LABEL
	jmp	.LBB0_660
.LBB0_657:
.Ltmp267:                               # EH_LABEL
	jmp	.LBB0_660
.LBB0_658:
.Ltmp264:                               # EH_LABEL
	jmp	.LBB0_660
.LBB0_659:
.Ltmp261:                               # EH_LABEL
.LBB0_660:
	movq	%rdx, %r15
	movq	%rax, %rbx
.LBB0_661:
	.cfi_escape 0x2e, 0x00
	leaq	752(%rsp), %rdi
	callq	_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev
	.cfi_escape 0x2e, 0x00
	leaq	928(%rsp), %rdi
	callq	_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev
	.cfi_escape 0x2e, 0x00
	leaq	656(%rsp), %rdi
	callq	_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	callq	_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev
	jmp	.LBB0_734
.LBB0_662:
.Ltmp332:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_732
.LBB0_663:
.Ltmp123:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %r12
.LBB0_664:                              # =>This Inner Loop Header: Depth=1
	movq	-16(%r14), %rdi
	addq	$-16, %r14
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev
	cmpq	%rbx, %r14
	jne	.LBB0_664
# %bb.665:
	movq	%r12, %rbx
	jmp	.LBB0_736
.LBB0_666:
.Ltmp109:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %r12
.LBB0_667:                              # =>This Inner Loop Header: Depth=1
	movq	-16(%r14), %rdi
	addq	$-16, %r14
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev
	cmpq	%rbx, %r14
	jne	.LBB0_667
# %bb.668:
	movq	%r12, %rbx
	jmp	.LBB0_737
.LBB0_669:
.Ltmp14:                                # EH_LABEL
	jmp	.LBB0_711
.LBB0_670:
.Ltmp200:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbp
	movq	(%r14), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112GuardedFloatD2Ev
	jmp	.LBB0_672
.LBB0_671:
.Ltmp195:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbp
.LBB0_672:
	movq	(%r12), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_111GuardedBf16D2Ev
	jmp	.LBB0_674
.LBB0_673:
.Ltmp190:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbp
.LBB0_674:
	movq	368(%rsp,%rbx), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_111GuardedBf16D2Ev
	jmp	.LBB0_676
.LBB0_675:
.Ltmp185:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbp
.LBB0_676:
	testq	%rbx, %rbx
	je	.LBB0_680
# %bb.677:
	leaq	368(%rsp), %rbx
.LBB0_678:                              # =>This Inner Loop Header: Depth=1
	addq	$-96, %r13
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_ZN12_GLOBAL__N_17OutputsD2Ev
	cmpq	%rbx, %r13
	jne	.LBB0_678
.LBB0_680:
	movq	%rbp, %rbx
	jmp	.LBB0_735
.LBB0_681:
.Ltmp95:                                # EH_LABEL
.LBB0_682:
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_737
.LBB0_683:
.Ltmp242:                               # EH_LABEL
	jmp	.LBB0_687
.LBB0_684:
.Ltmp307:                               # EH_LABEL
	jmp	.LBB0_687
.LBB0_685:
.Ltmp595:                               # EH_LABEL
	jmp	.LBB0_711
.LBB0_686:
.Ltmp225:                               # EH_LABEL
.LBB0_687:
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_734
.LBB0_688:
.Ltmp148:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_735
.LBB0_689:
.Ltmp602:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	368(%rsp), %rdi
	cmpq	%rbp, %rdi
	je	.LBB0_694
# %bb.690:
	movq	384(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB0_694
.LBB0_691:
.Ltmp575:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	368(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.LBB0_753
.LBB0_692:
	movq	384(%rsp), %rsi
	jmp	.LBB0_752
.LBB0_693:
.Ltmp609:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
.LBB0_694:
	movq	1640(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.LBB0_753
.LBB0_695:
	movq	1656(%rsp), %rsi
	jmp	.LBB0_752
.LBB0_696:
.Ltmp521:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
.LBB0_697:
	movq	752(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.LBB0_701
# %bb.698:
	movq	768(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB0_701
.LBB0_699:
.Ltmp528:                               # EH_LABEL
.LBB0_700:
	movq	%rdx, %r15
	movq	%rax, %rbx
.LBB0_701:
	movq	64(%rsp), %rdi
	cmpq	%r12, %rdi
	je	.LBB0_724
# %bb.702:
	movq	80(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB0_724
.LBB0_703:
.Ltmp535:                               # EH_LABEL
	jmp	.LBB0_715
.LBB0_704:
.Ltmp564:                               # EH_LABEL
	jmp	.LBB0_711
.LBB0_705:
.Ltmp580:                               # EH_LABEL
	jmp	.LBB0_711
.LBB0_706:
.Ltmp420:                               # EH_LABEL
	jmp	.LBB0_728
.LBB0_707:
.Ltmp538:                               # EH_LABEL
	jmp	.LBB0_715
.LBB0_708:
.Ltmp541:                               # EH_LABEL
	jmp	.LBB0_715
.LBB0_709:
.Ltmp417:                               # EH_LABEL
	jmp	.LBB0_728
.LBB0_710:
.Ltmp616:                               # EH_LABEL
.LBB0_711:
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_753
.LBB0_712:
.Ltmp411:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_713:
.Ltmp414:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_714:
.Ltmp544:                               # EH_LABEL
.LBB0_715:
	movq	%rdx, %r15
	movq	%rax, %rbx
	jmp	.LBB0_724
.LBB0_716:
.Ltmp402:                               # EH_LABEL
	jmp	.LBB0_722
.LBB0_717:
.Ltmp370:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%rax, 176(%rsp)
	jmp	.LBB0_723
.LBB0_718:
.Ltmp365:                               # EH_LABEL
	jmp	.LBB0_728
.LBB0_719:
.Ltmp353:                               # EH_LABEL
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%rax, 176(%rsp)
	testq	%r13, %r13
	je	.LBB0_724
# %bb.720:
	subq	%r13, %r14
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB0_724
.LBB0_721:
.Ltmp397:                               # EH_LABEL
.LBB0_722:
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%rax, 176(%rsp)
	movq	224(%rsp), %rdi
.Ltmp403:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp404:                               # EH_LABEL
.LBB0_723:
	movq	312(%rsp), %rdi
.Ltmp406:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp407:                               # EH_LABEL
.LBB0_724:
	movq	8(%rsp), %rdi                   # 8-byte Reload
	jmp	.LBB0_729
.LBB0_725:
.Ltmp405:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_726:
.Ltmp408:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB0_727:
.Ltmp360:                               # EH_LABEL
.LBB0_728:
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	8(%rsp), %rdi                   # 8-byte Reload
	movq	%rdi, 176(%rsp)
.LBB0_729:
	testq	%rdi, %rdi
	je	.LBB0_731
# %bb.730:
	movq	192(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_731:
	.cfi_escape 0x2e, 0x00
	leaq	928(%rsp), %rdi
	callq	_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev
.LBB0_732:
	movq	832(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112DeviceBufferIjED2Ev
.LBB0_733:
	movq	736(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112DeviceBufferIjED2Ev
.LBB0_734:
	.cfi_escape 0x2e, 0x00
	leaq	368(%rsp), %rdi
	callq	_ZNSt5arrayIN12_GLOBAL__N_17OutputsELm3EED2Ev
.LBB0_735:
	.cfi_escape 0x2e, 0x00
	leaq	1008(%rsp), %rdi
	callq	_ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev
.LBB0_736:
	.cfi_escape 0x2e, 0x00
	leaq	1056(%rsp), %rdi
	callq	_ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev
.LBB0_737:
	movq	848(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112DeviceBufferIfED2Ev
.LBB0_738:
	movq	864(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112DeviceBufferIfED2Ev
.LBB0_739:
	movq	880(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev
.LBB0_740:
	.cfi_escape 0x2e, 0x00
	movl	$192, %esi
	movq	264(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB0_741:
	.cfi_escape 0x2e, 0x00
	movl	$192, %esi
	movq	272(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB0_742:
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	152(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB0_743:
	.cfi_escape 0x2e, 0x00
	movl	$491520, %esi                   # imm = 0x78000
	movq	160(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB0_744:
	.cfi_escape 0x2e, 0x00
	movl	$10240, %esi                    # imm = 0x2800
	movq	280(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB0_745:
	movq	1272(%rsp), %rdi
	leaq	1288(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_747
# %bb.746:
	movq	1288(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_747:
	movq	1120(%rsp), %rdi
	leaq	1136(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_750
# %bb.748:
	movq	1136(%rsp), %rsi
.LBB0_749:
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_750:
	movq	232(%rsp), %rdi
	leaq	248(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB0_753
# %bb.751:
	movq	248(%rsp), %rsi
.LBB0_752:
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB0_753:
	cmpl	$1, %r15d
	jne	.LBB0_759
# %bb.754:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	__cxa_begin_catch@PLT
	movq	%rax, %rbx
.Ltmp617:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cerr@GOTPCREL(%rip), %rdi
	leaq	.L.str.32(%rip), %rsi
	movl	$6, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp618:                               # EH_LABEL
# %bb.755:
	movq	(%rbx), %rax
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	*16(%rax)
.Ltmp619:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cerr@GOTPCREL(%rip), %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
.Ltmp620:                               # EH_LABEL
# %bb.756:
.Ltmp621:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c@PLT
.Ltmp622:                               # EH_LABEL
# %bb.757:
	.cfi_escape 0x2e, 0x00
	callq	__cxa_end_catch@PLT
	movl	$1, %ebx
	jmp	.LBB0_498
.LBB0_758:
.Ltmp623:                               # EH_LABEL
	movq	%rax, %rbx
.Ltmp624:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	__cxa_end_catch@PLT
.Ltmp625:                               # EH_LABEL
.LBB0_759:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB0_760:
.Ltmp626:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
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
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 1 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp616-.Lfunc_begin0         #     jumps to .Ltmp616
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp2-.Ltmp1                  #   Call between .Ltmp1 and .Ltmp2
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp2-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp3-.Ltmp2                  #   Call between .Ltmp2 and .Ltmp3
	.uleb128 .Ltmp609-.Lfunc_begin0         #     jumps to .Ltmp609
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp4-.Lfunc_begin0           # >> Call Site 5 <<
	.uleb128 .Ltmp5-.Ltmp4                  #   Call between .Ltmp4 and .Ltmp5
	.uleb128 .Ltmp602-.Lfunc_begin0         #     jumps to .Ltmp602
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp5-.Lfunc_begin0           # >> Call Site 6 <<
	.uleb128 .Ltmp587-.Ltmp5                #   Call between .Ltmp5 and .Ltmp587
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp587-.Lfunc_begin0         # >> Call Site 7 <<
	.uleb128 .Ltmp594-.Ltmp587              #   Call between .Ltmp587 and .Ltmp594
	.uleb128 .Ltmp595-.Lfunc_begin0         #     jumps to .Ltmp595
	.byte	5                               #   On action: 3
	.uleb128 .Ltmp6-.Lfunc_begin0           # >> Call Site 8 <<
	.uleb128 .Ltmp13-.Ltmp6                 #   Call between .Ltmp6 and .Ltmp13
	.uleb128 .Ltmp14-.Lfunc_begin0          #     jumps to .Ltmp14
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp13-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp15-.Ltmp13                #   Call between .Ltmp13 and .Ltmp15
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp15-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp18-.Ltmp15                #   Call between .Ltmp15 and .Ltmp18
	.uleb128 .Ltmp19-.Lfunc_begin0          #     jumps to .Ltmp19
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp20-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp21-.Ltmp20                #   Call between .Ltmp20 and .Ltmp21
	.uleb128 .Ltmp580-.Lfunc_begin0         #     jumps to .Ltmp580
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp21-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp22-.Ltmp21                #   Call between .Ltmp21 and .Ltmp22
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp22-.Lfunc_begin0          # >> Call Site 13 <<
	.uleb128 .Ltmp23-.Ltmp22                #   Call between .Ltmp22 and .Ltmp23
	.uleb128 .Ltmp575-.Lfunc_begin0         #     jumps to .Ltmp575
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp23-.Lfunc_begin0          # >> Call Site 14 <<
	.uleb128 .Ltmp24-.Ltmp23                #   Call between .Ltmp23 and .Ltmp24
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp24-.Lfunc_begin0          # >> Call Site 15 <<
	.uleb128 .Ltmp27-.Ltmp24                #   Call between .Ltmp24 and .Ltmp27
	.uleb128 .Ltmp28-.Lfunc_begin0          #     jumps to .Ltmp28
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp29-.Lfunc_begin0          # >> Call Site 16 <<
	.uleb128 .Ltmp30-.Ltmp29                #   Call between .Ltmp29 and .Ltmp30
	.uleb128 .Ltmp564-.Lfunc_begin0         #     jumps to .Ltmp564
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp30-.Lfunc_begin0          # >> Call Site 17 <<
	.uleb128 .Ltmp31-.Ltmp30                #   Call between .Ltmp30 and .Ltmp31
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp31-.Lfunc_begin0          # >> Call Site 18 <<
	.uleb128 .Ltmp32-.Ltmp31                #   Call between .Ltmp31 and .Ltmp32
	.uleb128 .Ltmp33-.Lfunc_begin0          #     jumps to .Ltmp33
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp32-.Lfunc_begin0          # >> Call Site 19 <<
	.uleb128 .Ltmp34-.Ltmp32                #   Call between .Ltmp32 and .Ltmp34
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp34-.Lfunc_begin0          # >> Call Site 20 <<
	.uleb128 .Ltmp35-.Ltmp34                #   Call between .Ltmp34 and .Ltmp35
	.uleb128 .Ltmp36-.Lfunc_begin0          #     jumps to .Ltmp36
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp37-.Lfunc_begin0          # >> Call Site 21 <<
	.uleb128 .Ltmp38-.Ltmp37                #   Call between .Ltmp37 and .Ltmp38
	.uleb128 .Ltmp559-.Lfunc_begin0         #     jumps to .Ltmp559
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp38-.Lfunc_begin0          # >> Call Site 22 <<
	.uleb128 .Ltmp39-.Ltmp38                #   Call between .Ltmp38 and .Ltmp39
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp39-.Lfunc_begin0          # >> Call Site 23 <<
	.uleb128 .Ltmp40-.Ltmp39                #   Call between .Ltmp39 and .Ltmp40
	.uleb128 .Ltmp41-.Lfunc_begin0          #     jumps to .Ltmp41
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp48-.Lfunc_begin0          # >> Call Site 24 <<
	.uleb128 .Ltmp51-.Ltmp48                #   Call between .Ltmp48 and .Ltmp51
	.uleb128 .Ltmp52-.Lfunc_begin0          #     jumps to .Ltmp52
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp53-.Lfunc_begin0          # >> Call Site 25 <<
	.uleb128 .Ltmp54-.Ltmp53                #   Call between .Ltmp53 and .Ltmp54
	.uleb128 .Ltmp55-.Lfunc_begin0          #     jumps to .Ltmp55
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp54-.Lfunc_begin0          # >> Call Site 26 <<
	.uleb128 .Ltmp56-.Ltmp54                #   Call between .Ltmp54 and .Ltmp56
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp56-.Lfunc_begin0          # >> Call Site 27 <<
	.uleb128 .Ltmp57-.Ltmp56                #   Call between .Ltmp56 and .Ltmp57
	.uleb128 .Ltmp58-.Lfunc_begin0          #     jumps to .Ltmp58
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp57-.Lfunc_begin0          # >> Call Site 28 <<
	.uleb128 .Ltmp59-.Ltmp57                #   Call between .Ltmp57 and .Ltmp59
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp59-.Lfunc_begin0          # >> Call Site 29 <<
	.uleb128 .Ltmp60-.Ltmp59                #   Call between .Ltmp59 and .Ltmp60
	.uleb128 .Ltmp61-.Lfunc_begin0          #     jumps to .Ltmp61
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp60-.Lfunc_begin0          # >> Call Site 30 <<
	.uleb128 .Ltmp62-.Ltmp60                #   Call between .Ltmp60 and .Ltmp62
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp62-.Lfunc_begin0          # >> Call Site 31 <<
	.uleb128 .Ltmp63-.Ltmp62                #   Call between .Ltmp62 and .Ltmp63
	.uleb128 .Ltmp64-.Lfunc_begin0          #     jumps to .Ltmp64
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp65-.Lfunc_begin0          # >> Call Site 32 <<
	.uleb128 .Ltmp66-.Ltmp65                #   Call between .Ltmp65 and .Ltmp66
	.uleb128 .Ltmp67-.Lfunc_begin0          #     jumps to .Ltmp67
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp68-.Lfunc_begin0          # >> Call Site 33 <<
	.uleb128 .Ltmp71-.Ltmp68                #   Call between .Ltmp68 and .Ltmp71
	.uleb128 .Ltmp72-.Lfunc_begin0          #     jumps to .Ltmp72
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp73-.Lfunc_begin0          # >> Call Site 34 <<
	.uleb128 .Ltmp76-.Ltmp73                #   Call between .Ltmp73 and .Ltmp76
	.uleb128 .Ltmp77-.Lfunc_begin0          #     jumps to .Ltmp77
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp78-.Lfunc_begin0          # >> Call Site 35 <<
	.uleb128 .Ltmp81-.Ltmp78                #   Call between .Ltmp78 and .Ltmp81
	.uleb128 .Ltmp82-.Lfunc_begin0          #     jumps to .Ltmp82
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp83-.Lfunc_begin0          # >> Call Site 36 <<
	.uleb128 .Ltmp94-.Ltmp83                #   Call between .Ltmp83 and .Ltmp94
	.uleb128 .Ltmp95-.Lfunc_begin0          #     jumps to .Ltmp95
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp96-.Lfunc_begin0          # >> Call Site 37 <<
	.uleb128 .Ltmp99-.Ltmp96                #   Call between .Ltmp96 and .Ltmp99
	.uleb128 .Ltmp100-.Lfunc_begin0         #     jumps to .Ltmp100
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp101-.Lfunc_begin0         # >> Call Site 38 <<
	.uleb128 .Ltmp108-.Ltmp101              #   Call between .Ltmp101 and .Ltmp108
	.uleb128 .Ltmp109-.Lfunc_begin0         #     jumps to .Ltmp109
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp110-.Lfunc_begin0         # >> Call Site 39 <<
	.uleb128 .Ltmp113-.Ltmp110              #   Call between .Ltmp110 and .Ltmp113
	.uleb128 .Ltmp114-.Lfunc_begin0         #     jumps to .Ltmp114
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp115-.Lfunc_begin0         # >> Call Site 40 <<
	.uleb128 .Ltmp122-.Ltmp115              #   Call between .Ltmp115 and .Ltmp122
	.uleb128 .Ltmp123-.Lfunc_begin0         #     jumps to .Ltmp123
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp124-.Lfunc_begin0         # >> Call Site 41 <<
	.uleb128 .Ltmp147-.Ltmp124              #   Call between .Ltmp124 and .Ltmp147
	.uleb128 .Ltmp148-.Lfunc_begin0         #     jumps to .Ltmp148
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp149-.Lfunc_begin0         # >> Call Site 42 <<
	.uleb128 .Ltmp152-.Ltmp149              #   Call between .Ltmp149 and .Ltmp152
	.uleb128 .Ltmp185-.Lfunc_begin0         #     jumps to .Ltmp185
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp153-.Lfunc_begin0         # >> Call Site 43 <<
	.uleb128 .Ltmp156-.Ltmp153              #   Call between .Ltmp153 and .Ltmp156
	.uleb128 .Ltmp190-.Lfunc_begin0         #     jumps to .Ltmp190
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp157-.Lfunc_begin0         # >> Call Site 44 <<
	.uleb128 .Ltmp160-.Ltmp157              #   Call between .Ltmp157 and .Ltmp160
	.uleb128 .Ltmp195-.Lfunc_begin0         #     jumps to .Ltmp195
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp161-.Lfunc_begin0         # >> Call Site 45 <<
	.uleb128 .Ltmp164-.Ltmp161              #   Call between .Ltmp161 and .Ltmp164
	.uleb128 .Ltmp200-.Lfunc_begin0         #     jumps to .Ltmp200
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp165-.Lfunc_begin0         # >> Call Site 46 <<
	.uleb128 .Ltmp168-.Ltmp165              #   Call between .Ltmp165 and .Ltmp168
	.uleb128 .Ltmp185-.Lfunc_begin0         #     jumps to .Ltmp185
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp169-.Lfunc_begin0         # >> Call Site 47 <<
	.uleb128 .Ltmp172-.Ltmp169              #   Call between .Ltmp169 and .Ltmp172
	.uleb128 .Ltmp190-.Lfunc_begin0         #     jumps to .Ltmp190
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp173-.Lfunc_begin0         # >> Call Site 48 <<
	.uleb128 .Ltmp176-.Ltmp173              #   Call between .Ltmp173 and .Ltmp176
	.uleb128 .Ltmp195-.Lfunc_begin0         #     jumps to .Ltmp195
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp177-.Lfunc_begin0         # >> Call Site 49 <<
	.uleb128 .Ltmp180-.Ltmp177              #   Call between .Ltmp177 and .Ltmp180
	.uleb128 .Ltmp200-.Lfunc_begin0         #     jumps to .Ltmp200
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp181-.Lfunc_begin0         # >> Call Site 50 <<
	.uleb128 .Ltmp184-.Ltmp181              #   Call between .Ltmp181 and .Ltmp184
	.uleb128 .Ltmp185-.Lfunc_begin0         #     jumps to .Ltmp185
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp186-.Lfunc_begin0         # >> Call Site 51 <<
	.uleb128 .Ltmp189-.Ltmp186              #   Call between .Ltmp186 and .Ltmp189
	.uleb128 .Ltmp190-.Lfunc_begin0         #     jumps to .Ltmp190
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp191-.Lfunc_begin0         # >> Call Site 52 <<
	.uleb128 .Ltmp194-.Ltmp191              #   Call between .Ltmp191 and .Ltmp194
	.uleb128 .Ltmp195-.Lfunc_begin0         #     jumps to .Ltmp195
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp196-.Lfunc_begin0         # >> Call Site 53 <<
	.uleb128 .Ltmp199-.Ltmp196              #   Call between .Ltmp196 and .Ltmp199
	.uleb128 .Ltmp200-.Lfunc_begin0         #     jumps to .Ltmp200
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp201-.Lfunc_begin0         # >> Call Site 54 <<
	.uleb128 .Ltmp224-.Ltmp201              #   Call between .Ltmp201 and .Ltmp224
	.uleb128 .Ltmp225-.Lfunc_begin0         #     jumps to .Ltmp225
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp226-.Lfunc_begin0         # >> Call Site 55 <<
	.uleb128 .Ltmp241-.Ltmp226              #   Call between .Ltmp226 and .Ltmp241
	.uleb128 .Ltmp242-.Lfunc_begin0         #     jumps to .Ltmp242
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp243-.Lfunc_begin0         # >> Call Site 56 <<
	.uleb128 .Ltmp244-.Ltmp243              #   Call between .Ltmp243 and .Ltmp244
	.uleb128 .Ltmp261-.Lfunc_begin0         #     jumps to .Ltmp261
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp245-.Lfunc_begin0         # >> Call Site 57 <<
	.uleb128 .Ltmp246-.Ltmp245              #   Call between .Ltmp245 and .Ltmp246
	.uleb128 .Ltmp264-.Lfunc_begin0         #     jumps to .Ltmp264
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp247-.Lfunc_begin0         # >> Call Site 58 <<
	.uleb128 .Ltmp248-.Ltmp247              #   Call between .Ltmp247 and .Ltmp248
	.uleb128 .Ltmp267-.Lfunc_begin0         #     jumps to .Ltmp267
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp249-.Lfunc_begin0         # >> Call Site 59 <<
	.uleb128 .Ltmp250-.Ltmp249              #   Call between .Ltmp249 and .Ltmp250
	.uleb128 .Ltmp270-.Lfunc_begin0         #     jumps to .Ltmp270
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp251-.Lfunc_begin0         # >> Call Site 60 <<
	.uleb128 .Ltmp252-.Ltmp251              #   Call between .Ltmp251 and .Ltmp252
	.uleb128 .Ltmp261-.Lfunc_begin0         #     jumps to .Ltmp261
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp253-.Lfunc_begin0         # >> Call Site 61 <<
	.uleb128 .Ltmp254-.Ltmp253              #   Call between .Ltmp253 and .Ltmp254
	.uleb128 .Ltmp264-.Lfunc_begin0         #     jumps to .Ltmp264
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp255-.Lfunc_begin0         # >> Call Site 62 <<
	.uleb128 .Ltmp256-.Ltmp255              #   Call between .Ltmp255 and .Ltmp256
	.uleb128 .Ltmp267-.Lfunc_begin0         #     jumps to .Ltmp267
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp257-.Lfunc_begin0         # >> Call Site 63 <<
	.uleb128 .Ltmp258-.Ltmp257              #   Call between .Ltmp257 and .Ltmp258
	.uleb128 .Ltmp270-.Lfunc_begin0         #     jumps to .Ltmp270
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp259-.Lfunc_begin0         # >> Call Site 64 <<
	.uleb128 .Ltmp260-.Ltmp259              #   Call between .Ltmp259 and .Ltmp260
	.uleb128 .Ltmp261-.Lfunc_begin0         #     jumps to .Ltmp261
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp262-.Lfunc_begin0         # >> Call Site 65 <<
	.uleb128 .Ltmp263-.Ltmp262              #   Call between .Ltmp262 and .Ltmp263
	.uleb128 .Ltmp264-.Lfunc_begin0         #     jumps to .Ltmp264
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp265-.Lfunc_begin0         # >> Call Site 66 <<
	.uleb128 .Ltmp266-.Ltmp265              #   Call between .Ltmp265 and .Ltmp266
	.uleb128 .Ltmp267-.Lfunc_begin0         #     jumps to .Ltmp267
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp268-.Lfunc_begin0         # >> Call Site 67 <<
	.uleb128 .Ltmp269-.Ltmp268              #   Call between .Ltmp268 and .Ltmp269
	.uleb128 .Ltmp270-.Lfunc_begin0         #     jumps to .Ltmp270
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp269-.Lfunc_begin0         # >> Call Site 68 <<
	.uleb128 .Ltmp277-.Ltmp269              #   Call between .Ltmp269 and .Ltmp277
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp277-.Lfunc_begin0         # >> Call Site 69 <<
	.uleb128 .Ltmp306-.Ltmp277              #   Call between .Ltmp277 and .Ltmp306
	.uleb128 .Ltmp307-.Lfunc_begin0         #     jumps to .Ltmp307
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp314-.Lfunc_begin0         # >> Call Site 70 <<
	.uleb128 .Ltmp317-.Ltmp314              #   Call between .Ltmp314 and .Ltmp317
	.uleb128 .Ltmp318-.Lfunc_begin0         #     jumps to .Ltmp318
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp319-.Lfunc_begin0         # >> Call Site 71 <<
	.uleb128 .Ltmp322-.Ltmp319              #   Call between .Ltmp319 and .Ltmp322
	.uleb128 .Ltmp323-.Lfunc_begin0         #     jumps to .Ltmp323
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp324-.Lfunc_begin0         # >> Call Site 72 <<
	.uleb128 .Ltmp331-.Ltmp324              #   Call between .Ltmp324 and .Ltmp331
	.uleb128 .Ltmp332-.Lfunc_begin0         #     jumps to .Ltmp332
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp333-.Lfunc_begin0         # >> Call Site 73 <<
	.uleb128 .Ltmp348-.Ltmp333              #   Call between .Ltmp333 and .Ltmp348
	.uleb128 .Ltmp360-.Lfunc_begin0         #     jumps to .Ltmp360
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp348-.Lfunc_begin0         # >> Call Site 74 <<
	.uleb128 .Ltmp349-.Ltmp348              #   Call between .Ltmp348 and .Ltmp349
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp349-.Lfunc_begin0         # >> Call Site 75 <<
	.uleb128 .Ltmp352-.Ltmp349              #   Call between .Ltmp349 and .Ltmp352
	.uleb128 .Ltmp353-.Lfunc_begin0         #     jumps to .Ltmp353
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp354-.Lfunc_begin0         # >> Call Site 76 <<
	.uleb128 .Ltmp359-.Ltmp354              #   Call between .Ltmp354 and .Ltmp359
	.uleb128 .Ltmp360-.Lfunc_begin0         #     jumps to .Ltmp360
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp361-.Lfunc_begin0         # >> Call Site 77 <<
	.uleb128 .Ltmp364-.Ltmp361              #   Call between .Ltmp361 and .Ltmp364
	.uleb128 .Ltmp365-.Lfunc_begin0         #     jumps to .Ltmp365
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp366-.Lfunc_begin0         # >> Call Site 78 <<
	.uleb128 .Ltmp369-.Ltmp366              #   Call between .Ltmp366 and .Ltmp369
	.uleb128 .Ltmp370-.Lfunc_begin0         #     jumps to .Ltmp370
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp371-.Lfunc_begin0         # >> Call Site 79 <<
	.uleb128 .Ltmp396-.Ltmp371              #   Call between .Ltmp371 and .Ltmp396
	.uleb128 .Ltmp397-.Lfunc_begin0         #     jumps to .Ltmp397
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp398-.Lfunc_begin0         # >> Call Site 80 <<
	.uleb128 .Ltmp401-.Ltmp398              #   Call between .Ltmp398 and .Ltmp401
	.uleb128 .Ltmp402-.Lfunc_begin0         #     jumps to .Ltmp402
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp409-.Lfunc_begin0         # >> Call Site 81 <<
	.uleb128 .Ltmp410-.Ltmp409              #   Call between .Ltmp409 and .Ltmp410
	.uleb128 .Ltmp411-.Lfunc_begin0         #     jumps to .Ltmp411
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp412-.Lfunc_begin0         # >> Call Site 82 <<
	.uleb128 .Ltmp413-.Ltmp412              #   Call between .Ltmp412 and .Ltmp413
	.uleb128 .Ltmp414-.Lfunc_begin0         #     jumps to .Ltmp414
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp415-.Lfunc_begin0         # >> Call Site 83 <<
	.uleb128 .Ltmp416-.Ltmp415              #   Call between .Ltmp415 and .Ltmp416
	.uleb128 .Ltmp417-.Lfunc_begin0         #     jumps to .Ltmp417
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp416-.Lfunc_begin0         # >> Call Site 84 <<
	.uleb128 .Ltmp418-.Ltmp416              #   Call between .Ltmp416 and .Ltmp418
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp418-.Lfunc_begin0         # >> Call Site 85 <<
	.uleb128 .Ltmp419-.Ltmp418              #   Call between .Ltmp418 and .Ltmp419
	.uleb128 .Ltmp420-.Lfunc_begin0         #     jumps to .Ltmp420
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp419-.Lfunc_begin0         # >> Call Site 86 <<
	.uleb128 .Ltmp421-.Ltmp419              #   Call between .Ltmp419 and .Ltmp421
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp421-.Lfunc_begin0         # >> Call Site 87 <<
	.uleb128 .Ltmp422-.Ltmp421              #   Call between .Ltmp421 and .Ltmp422
	.uleb128 .Ltmp423-.Lfunc_begin0         #     jumps to .Ltmp423
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp430-.Lfunc_begin0         # >> Call Site 88 <<
	.uleb128 .Ltmp431-.Ltmp430              #   Call between .Ltmp430 and .Ltmp431
	.uleb128 .Ltmp535-.Lfunc_begin0         #     jumps to .Ltmp535
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp431-.Lfunc_begin0         # >> Call Site 89 <<
	.uleb128 .Ltmp432-.Ltmp431              #   Call between .Ltmp431 and .Ltmp432
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp432-.Lfunc_begin0         # >> Call Site 90 <<
	.uleb128 .Ltmp433-.Ltmp432              #   Call between .Ltmp432 and .Ltmp433
	.uleb128 .Ltmp528-.Lfunc_begin0         #     jumps to .Ltmp528
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp433-.Lfunc_begin0         # >> Call Site 91 <<
	.uleb128 .Ltmp434-.Ltmp433              #   Call between .Ltmp433 and .Ltmp434
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp434-.Lfunc_begin0         # >> Call Site 92 <<
	.uleb128 .Ltmp435-.Ltmp434              #   Call between .Ltmp434 and .Ltmp435
	.uleb128 .Ltmp521-.Lfunc_begin0         #     jumps to .Ltmp521
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp435-.Lfunc_begin0         # >> Call Site 93 <<
	.uleb128 .Ltmp436-.Ltmp435              #   Call between .Ltmp435 and .Ltmp436
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp436-.Lfunc_begin0         # >> Call Site 94 <<
	.uleb128 .Ltmp437-.Ltmp436              #   Call between .Ltmp436 and .Ltmp437
	.uleb128 .Ltmp438-.Lfunc_begin0         #     jumps to .Ltmp438
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp439-.Lfunc_begin0         # >> Call Site 95 <<
	.uleb128 .Ltmp448-.Ltmp439              #   Call between .Ltmp439 and .Ltmp448
	.uleb128 .Ltmp449-.Lfunc_begin0         #     jumps to .Ltmp449
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp450-.Lfunc_begin0         # >> Call Site 96 <<
	.uleb128 .Ltmp451-.Ltmp450              #   Call between .Ltmp450 and .Ltmp451
	.uleb128 .Ltmp452-.Lfunc_begin0         #     jumps to .Ltmp452
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp453-.Lfunc_begin0         # >> Call Site 97 <<
	.uleb128 .Ltmp454-.Ltmp453              #   Call between .Ltmp453 and .Ltmp454
	.uleb128 .Ltmp455-.Lfunc_begin0         #     jumps to .Ltmp455
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp456-.Lfunc_begin0         # >> Call Site 98 <<
	.uleb128 .Ltmp457-.Ltmp456              #   Call between .Ltmp456 and .Ltmp457
	.uleb128 .Ltmp458-.Lfunc_begin0         #     jumps to .Ltmp458
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp459-.Lfunc_begin0         # >> Call Site 99 <<
	.uleb128 .Ltmp460-.Ltmp459              #   Call between .Ltmp459 and .Ltmp460
	.uleb128 .Ltmp461-.Lfunc_begin0         #     jumps to .Ltmp461
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp462-.Lfunc_begin0         # >> Call Site 100 <<
	.uleb128 .Ltmp463-.Ltmp462              #   Call between .Ltmp462 and .Ltmp463
	.uleb128 .Ltmp464-.Lfunc_begin0         #     jumps to .Ltmp464
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp465-.Lfunc_begin0         # >> Call Site 101 <<
	.uleb128 .Ltmp466-.Ltmp465              #   Call between .Ltmp465 and .Ltmp466
	.uleb128 .Ltmp467-.Lfunc_begin0         #     jumps to .Ltmp467
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp468-.Lfunc_begin0         # >> Call Site 102 <<
	.uleb128 .Ltmp469-.Ltmp468              #   Call between .Ltmp468 and .Ltmp469
	.uleb128 .Ltmp470-.Lfunc_begin0         #     jumps to .Ltmp470
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp471-.Lfunc_begin0         # >> Call Site 103 <<
	.uleb128 .Ltmp472-.Ltmp471              #   Call between .Ltmp471 and .Ltmp472
	.uleb128 .Ltmp473-.Lfunc_begin0         #     jumps to .Ltmp473
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp474-.Lfunc_begin0         # >> Call Site 104 <<
	.uleb128 .Ltmp475-.Ltmp474              #   Call between .Ltmp474 and .Ltmp475
	.uleb128 .Ltmp476-.Lfunc_begin0         #     jumps to .Ltmp476
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp477-.Lfunc_begin0         # >> Call Site 105 <<
	.uleb128 .Ltmp478-.Ltmp477              #   Call between .Ltmp477 and .Ltmp478
	.uleb128 .Ltmp479-.Lfunc_begin0         #     jumps to .Ltmp479
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp480-.Lfunc_begin0         # >> Call Site 106 <<
	.uleb128 .Ltmp481-.Ltmp480              #   Call between .Ltmp480 and .Ltmp481
	.uleb128 .Ltmp482-.Lfunc_begin0         #     jumps to .Ltmp482
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp483-.Lfunc_begin0         # >> Call Site 107 <<
	.uleb128 .Ltmp484-.Ltmp483              #   Call between .Ltmp483 and .Ltmp484
	.uleb128 .Ltmp485-.Lfunc_begin0         #     jumps to .Ltmp485
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp486-.Lfunc_begin0         # >> Call Site 108 <<
	.uleb128 .Ltmp487-.Ltmp486              #   Call between .Ltmp486 and .Ltmp487
	.uleb128 .Ltmp488-.Lfunc_begin0         #     jumps to .Ltmp488
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp489-.Lfunc_begin0         # >> Call Site 109 <<
	.uleb128 .Ltmp490-.Ltmp489              #   Call between .Ltmp489 and .Ltmp490
	.uleb128 .Ltmp491-.Lfunc_begin0         #     jumps to .Ltmp491
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp492-.Lfunc_begin0         # >> Call Site 110 <<
	.uleb128 .Ltmp497-.Ltmp492              #   Call between .Ltmp492 and .Ltmp497
	.uleb128 .Ltmp498-.Lfunc_begin0         #     jumps to .Ltmp498
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp499-.Lfunc_begin0         # >> Call Site 111 <<
	.uleb128 .Ltmp504-.Ltmp499              #   Call between .Ltmp499 and .Ltmp504
	.uleb128 .Ltmp505-.Lfunc_begin0         #     jumps to .Ltmp505
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp506-.Lfunc_begin0         # >> Call Site 112 <<
	.uleb128 .Ltmp507-.Ltmp506              #   Call between .Ltmp506 and .Ltmp507
	.uleb128 .Ltmp508-.Lfunc_begin0         #     jumps to .Ltmp508
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp509-.Lfunc_begin0         # >> Call Site 113 <<
	.uleb128 .Ltmp510-.Ltmp509              #   Call between .Ltmp509 and .Ltmp510
	.uleb128 .Ltmp511-.Lfunc_begin0         #     jumps to .Ltmp511
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp512-.Lfunc_begin0         # >> Call Site 114 <<
	.uleb128 .Ltmp513-.Ltmp512              #   Call between .Ltmp512 and .Ltmp513
	.uleb128 .Ltmp514-.Lfunc_begin0         #     jumps to .Ltmp514
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp542-.Lfunc_begin0         # >> Call Site 115 <<
	.uleb128 .Ltmp543-.Ltmp542              #   Call between .Ltmp542 and .Ltmp543
	.uleb128 .Ltmp544-.Lfunc_begin0         #     jumps to .Ltmp544
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp610-.Lfunc_begin0         # >> Call Site 116 <<
	.uleb128 .Ltmp611-.Ltmp610              #   Call between .Ltmp610 and .Ltmp611
	.uleb128 .Ltmp616-.Lfunc_begin0         #     jumps to .Ltmp616
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp539-.Lfunc_begin0         # >> Call Site 117 <<
	.uleb128 .Ltmp540-.Ltmp539              #   Call between .Ltmp539 and .Ltmp540
	.uleb128 .Ltmp541-.Lfunc_begin0         #     jumps to .Ltmp541
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp536-.Lfunc_begin0         # >> Call Site 118 <<
	.uleb128 .Ltmp537-.Ltmp536              #   Call between .Ltmp536 and .Ltmp537
	.uleb128 .Ltmp538-.Lfunc_begin0         #     jumps to .Ltmp538
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp576-.Lfunc_begin0         # >> Call Site 119 <<
	.uleb128 .Ltmp577-.Ltmp576              #   Call between .Ltmp576 and .Ltmp577
	.uleb128 .Ltmp580-.Lfunc_begin0         #     jumps to .Ltmp580
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp560-.Lfunc_begin0         # >> Call Site 120 <<
	.uleb128 .Ltmp561-.Ltmp560              #   Call between .Ltmp560 and .Ltmp561
	.uleb128 .Ltmp564-.Lfunc_begin0         #     jumps to .Ltmp564
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp529-.Lfunc_begin0         # >> Call Site 121 <<
	.uleb128 .Ltmp530-.Ltmp529              #   Call between .Ltmp529 and .Ltmp530
	.uleb128 .Ltmp535-.Lfunc_begin0         #     jumps to .Ltmp535
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp522-.Lfunc_begin0         # >> Call Site 122 <<
	.uleb128 .Ltmp523-.Ltmp522              #   Call between .Ltmp522 and .Ltmp523
	.uleb128 .Ltmp528-.Lfunc_begin0         #     jumps to .Ltmp528
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp515-.Lfunc_begin0         # >> Call Site 123 <<
	.uleb128 .Ltmp516-.Ltmp515              #   Call between .Ltmp515 and .Ltmp516
	.uleb128 .Ltmp521-.Lfunc_begin0         #     jumps to .Ltmp521
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp603-.Lfunc_begin0         # >> Call Site 124 <<
	.uleb128 .Ltmp604-.Ltmp603              #   Call between .Ltmp603 and .Ltmp604
	.uleb128 .Ltmp609-.Lfunc_begin0         #     jumps to .Ltmp609
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp571-.Lfunc_begin0         # >> Call Site 125 <<
	.uleb128 .Ltmp572-.Ltmp571              #   Call between .Ltmp571 and .Ltmp572
	.uleb128 .Ltmp575-.Lfunc_begin0         #     jumps to .Ltmp575
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp596-.Lfunc_begin0         # >> Call Site 126 <<
	.uleb128 .Ltmp597-.Ltmp596              #   Call between .Ltmp596 and .Ltmp597
	.uleb128 .Ltmp602-.Lfunc_begin0         #     jumps to .Ltmp602
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp283-.Lfunc_begin0         # >> Call Site 127 <<
	.uleb128 .Ltmp284-.Ltmp283              #   Call between .Ltmp283 and .Ltmp284
	.uleb128 .Ltmp285-.Lfunc_begin0         #     jumps to .Ltmp285
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp286-.Lfunc_begin0         # >> Call Site 128 <<
	.uleb128 .Ltmp287-.Ltmp286              #   Call between .Ltmp286 and .Ltmp287
	.uleb128 .Ltmp288-.Lfunc_begin0         #     jumps to .Ltmp288
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp295-.Lfunc_begin0         # >> Call Site 129 <<
	.uleb128 .Ltmp296-.Ltmp295              #   Call between .Ltmp295 and .Ltmp296
	.uleb128 .Ltmp297-.Lfunc_begin0         #     jumps to .Ltmp297
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp298-.Lfunc_begin0         # >> Call Site 130 <<
	.uleb128 .Ltmp299-.Ltmp298              #   Call between .Ltmp298 and .Ltmp299
	.uleb128 .Ltmp300-.Lfunc_begin0         #     jumps to .Ltmp300
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp308-.Lfunc_begin0         # >> Call Site 131 <<
	.uleb128 .Ltmp309-.Ltmp308              #   Call between .Ltmp308 and .Ltmp309
	.uleb128 .Ltmp310-.Lfunc_begin0         #     jumps to .Ltmp310
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp311-.Lfunc_begin0         # >> Call Site 132 <<
	.uleb128 .Ltmp312-.Ltmp311              #   Call between .Ltmp311 and .Ltmp312
	.uleb128 .Ltmp313-.Lfunc_begin0         #     jumps to .Ltmp313
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp614-.Lfunc_begin0         # >> Call Site 133 <<
	.uleb128 .Ltmp615-.Ltmp614              #   Call between .Ltmp614 and .Ltmp615
	.uleb128 .Ltmp616-.Lfunc_begin0         #     jumps to .Ltmp616
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp42-.Lfunc_begin0          # >> Call Site 134 <<
	.uleb128 .Ltmp43-.Ltmp42                #   Call between .Ltmp42 and .Ltmp43
	.uleb128 .Ltmp44-.Lfunc_begin0          #     jumps to .Ltmp44
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp45-.Lfunc_begin0          # >> Call Site 135 <<
	.uleb128 .Ltmp46-.Ltmp45                #   Call between .Ltmp45 and .Ltmp46
	.uleb128 .Ltmp47-.Lfunc_begin0          #     jumps to .Ltmp47
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp612-.Lfunc_begin0         # >> Call Site 136 <<
	.uleb128 .Ltmp613-.Ltmp612              #   Call between .Ltmp612 and .Ltmp613
	.uleb128 .Ltmp616-.Lfunc_begin0         #     jumps to .Ltmp616
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp581-.Lfunc_begin0         # >> Call Site 137 <<
	.uleb128 .Ltmp582-.Ltmp581              #   Call between .Ltmp581 and .Ltmp582
	.uleb128 .Ltmp583-.Lfunc_begin0         #     jumps to .Ltmp583
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp584-.Lfunc_begin0         # >> Call Site 138 <<
	.uleb128 .Ltmp585-.Ltmp584              #   Call between .Ltmp584 and .Ltmp585
	.uleb128 .Ltmp586-.Lfunc_begin0         #     jumps to .Ltmp586
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp565-.Lfunc_begin0         # >> Call Site 139 <<
	.uleb128 .Ltmp566-.Ltmp565              #   Call between .Ltmp565 and .Ltmp566
	.uleb128 .Ltmp567-.Lfunc_begin0         #     jumps to .Ltmp567
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp568-.Lfunc_begin0         # >> Call Site 140 <<
	.uleb128 .Ltmp569-.Ltmp568              #   Call between .Ltmp568 and .Ltmp569
	.uleb128 .Ltmp570-.Lfunc_begin0         #     jumps to .Ltmp570
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp557-.Lfunc_begin0         # >> Call Site 141 <<
	.uleb128 .Ltmp558-.Ltmp557              #   Call between .Ltmp557 and .Ltmp558
	.uleb128 .Ltmp559-.Lfunc_begin0         #     jumps to .Ltmp559
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp271-.Lfunc_begin0         # >> Call Site 142 <<
	.uleb128 .Ltmp272-.Ltmp271              #   Call between .Ltmp271 and .Ltmp272
	.uleb128 .Ltmp273-.Lfunc_begin0         #     jumps to .Ltmp273
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp274-.Lfunc_begin0         # >> Call Site 143 <<
	.uleb128 .Ltmp275-.Ltmp274              #   Call between .Ltmp274 and .Ltmp275
	.uleb128 .Ltmp276-.Lfunc_begin0         #     jumps to .Ltmp276
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp551-.Lfunc_begin0         # >> Call Site 144 <<
	.uleb128 .Ltmp552-.Ltmp551              #   Call between .Ltmp551 and .Ltmp552
	.uleb128 .Ltmp553-.Lfunc_begin0         #     jumps to .Ltmp553
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp554-.Lfunc_begin0         # >> Call Site 145 <<
	.uleb128 .Ltmp555-.Ltmp554              #   Call between .Ltmp554 and .Ltmp555
	.uleb128 .Ltmp556-.Lfunc_begin0         #     jumps to .Ltmp556
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp545-.Lfunc_begin0         # >> Call Site 146 <<
	.uleb128 .Ltmp546-.Ltmp545              #   Call between .Ltmp545 and .Ltmp546
	.uleb128 .Ltmp547-.Lfunc_begin0         #     jumps to .Ltmp547
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp548-.Lfunc_begin0         # >> Call Site 147 <<
	.uleb128 .Ltmp549-.Ltmp548              #   Call between .Ltmp548 and .Ltmp549
	.uleb128 .Ltmp550-.Lfunc_begin0         #     jumps to .Ltmp550
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp424-.Lfunc_begin0         # >> Call Site 148 <<
	.uleb128 .Ltmp425-.Ltmp424              #   Call between .Ltmp424 and .Ltmp425
	.uleb128 .Ltmp426-.Lfunc_begin0         #     jumps to .Ltmp426
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp427-.Lfunc_begin0         # >> Call Site 149 <<
	.uleb128 .Ltmp428-.Ltmp427              #   Call between .Ltmp427 and .Ltmp428
	.uleb128 .Ltmp429-.Lfunc_begin0         #     jumps to .Ltmp429
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp533-.Lfunc_begin0         # >> Call Site 150 <<
	.uleb128 .Ltmp534-.Ltmp533              #   Call between .Ltmp533 and .Ltmp534
	.uleb128 .Ltmp535-.Lfunc_begin0         #     jumps to .Ltmp535
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp526-.Lfunc_begin0         # >> Call Site 151 <<
	.uleb128 .Ltmp527-.Ltmp526              #   Call between .Ltmp526 and .Ltmp527
	.uleb128 .Ltmp528-.Lfunc_begin0         #     jumps to .Ltmp528
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp519-.Lfunc_begin0         # >> Call Site 152 <<
	.uleb128 .Ltmp520-.Ltmp519              #   Call between .Ltmp519 and .Ltmp520
	.uleb128 .Ltmp521-.Lfunc_begin0         #     jumps to .Ltmp521
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp578-.Lfunc_begin0         # >> Call Site 153 <<
	.uleb128 .Ltmp579-.Ltmp578              #   Call between .Ltmp578 and .Ltmp579
	.uleb128 .Ltmp580-.Lfunc_begin0         #     jumps to .Ltmp580
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp562-.Lfunc_begin0         # >> Call Site 154 <<
	.uleb128 .Ltmp563-.Ltmp562              #   Call between .Ltmp562 and .Ltmp563
	.uleb128 .Ltmp564-.Lfunc_begin0         #     jumps to .Ltmp564
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp531-.Lfunc_begin0         # >> Call Site 155 <<
	.uleb128 .Ltmp532-.Ltmp531              #   Call between .Ltmp531 and .Ltmp532
	.uleb128 .Ltmp535-.Lfunc_begin0         #     jumps to .Ltmp535
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp524-.Lfunc_begin0         # >> Call Site 156 <<
	.uleb128 .Ltmp525-.Ltmp524              #   Call between .Ltmp524 and .Ltmp525
	.uleb128 .Ltmp528-.Lfunc_begin0         #     jumps to .Ltmp528
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp517-.Lfunc_begin0         # >> Call Site 157 <<
	.uleb128 .Ltmp518-.Ltmp517              #   Call between .Ltmp517 and .Ltmp518
	.uleb128 .Ltmp521-.Lfunc_begin0         #     jumps to .Ltmp521
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp607-.Lfunc_begin0         # >> Call Site 158 <<
	.uleb128 .Ltmp606-.Ltmp607              #   Call between .Ltmp607 and .Ltmp606
	.uleb128 .Ltmp609-.Lfunc_begin0         #     jumps to .Ltmp609
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp573-.Lfunc_begin0         # >> Call Site 159 <<
	.uleb128 .Ltmp574-.Ltmp573              #   Call between .Ltmp573 and .Ltmp574
	.uleb128 .Ltmp575-.Lfunc_begin0         #     jumps to .Ltmp575
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp600-.Lfunc_begin0         # >> Call Site 160 <<
	.uleb128 .Ltmp599-.Ltmp600              #   Call between .Ltmp600 and .Ltmp599
	.uleb128 .Ltmp602-.Lfunc_begin0         #     jumps to .Ltmp602
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp403-.Lfunc_begin0         # >> Call Site 161 <<
	.uleb128 .Ltmp404-.Ltmp403              #   Call between .Ltmp403 and .Ltmp404
	.uleb128 .Ltmp405-.Lfunc_begin0         #     jumps to .Ltmp405
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp406-.Lfunc_begin0         # >> Call Site 162 <<
	.uleb128 .Ltmp407-.Ltmp406              #   Call between .Ltmp406 and .Ltmp407
	.uleb128 .Ltmp408-.Lfunc_begin0         #     jumps to .Ltmp408
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp407-.Lfunc_begin0         # >> Call Site 163 <<
	.uleb128 .Ltmp617-.Ltmp407              #   Call between .Ltmp407 and .Ltmp617
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp617-.Lfunc_begin0         # >> Call Site 164 <<
	.uleb128 .Ltmp618-.Ltmp617              #   Call between .Ltmp617 and .Ltmp618
	.uleb128 .Ltmp623-.Lfunc_begin0         #     jumps to .Ltmp623
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp618-.Lfunc_begin0         # >> Call Site 165 <<
	.uleb128 .Ltmp619-.Ltmp618              #   Call between .Ltmp618 and .Ltmp619
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp619-.Lfunc_begin0         # >> Call Site 166 <<
	.uleb128 .Ltmp622-.Ltmp619              #   Call between .Ltmp619 and .Ltmp622
	.uleb128 .Ltmp623-.Lfunc_begin0         #     jumps to .Ltmp623
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp622-.Lfunc_begin0         # >> Call Site 167 <<
	.uleb128 .Ltmp624-.Ltmp622              #   Call between .Ltmp622 and .Ltmp624
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp624-.Lfunc_begin0         # >> Call Site 168 <<
	.uleb128 .Ltmp625-.Ltmp624              #   Call between .Ltmp624 and .Ltmp625
	.uleb128 .Ltmp626-.Lfunc_begin0         #     jumps to .Ltmp626
	.byte	7                               #   On action: 4
	.uleb128 .Ltmp625-.Lfunc_begin0         # >> Call Site 169 <<
	.uleb128 .Lfunc_end0-.Ltmp625           #   Call between .Ltmp625 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.byte	0                               # >> Action Record 1 <<
                                        #   Cleanup
	.byte	0                               #   No further actions
	.byte	1                               # >> Action Record 2 <<
                                        #   Catch TypeInfo 1
	.byte	125                             #   Continue to action 1
	.byte	1                               # >> Action Record 3 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.byte	2                               # >> Action Record 4 <<
                                        #   Catch TypeInfo 2
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 2
.Ltmp627:                               # TypeInfo 1
	.long	.L_ZTISt9exception.DW.stub-.Ltmp627
.Lttbase0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,comdat
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_ # -- Begin function _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.p2align	1
	.prefalign	4, .Lfunc_end1, nop
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,@function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_: # @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	16(%rdi), %r12
	movq	%r12, (%rdi)
	testq	%rsi, %rsi
	je	.LBB1_10
# %bb.1:
	movq	%rsi, %r14
	movq	%rdi, %rbx
	movq	%rsi, %rdi
	callq	strlen@PLT
	movq	%rax, %r15
	cmpq	$16, %rax
	jb	.LBB1_5
# %bb.2:
	testq	%r15, %r15
	js	.LBB1_11
# %bb.3:
	movq	%r15, %rdi
	incq	%rdi
	js	.LBB1_12
# %bb.4:
	callq	_Znwm@PLT
	movq	%rax, %r12
	movq	%rax, (%rbx)
	movq	%r15, 16(%rbx)
.LBB1_5:
	testq	%r15, %r15
	je	.LBB1_9
# %bb.6:
	cmpq	$1, %r15
	jne	.LBB1_8
# %bb.7:
	movzbl	(%r14), %eax
	movb	%al, (%r12)
	jmp	.LBB1_9
.LBB1_8:
	movq	%r12, %rdi
	movq	%r14, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB1_9:
	movq	%r15, 8(%rbx)
	movb	$0, (%r12,%r15)
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB1_12:
	.cfi_def_cfa_offset 48
	callq	_ZSt17__throw_bad_allocv@PLT
.LBB1_10:
	leaq	.L.str.33(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.LBB1_11:
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end1:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_, .Lfunc_end1-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end2, nop     # -- Begin function _ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
	.type	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i,@function
_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i: # @_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
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
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$408, %rsp                      # imm = 0x198
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	testl	%edi, %edi
	jne	.LBB2_1
# %bb.17:
	addq	$408, %rsp                      # imm = 0x198
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB2_1:
	.cfi_def_cfa_offset 448
	movl	%edx, %ebp
	movq	%rsi, %r14
	movl	%edi, %ebx
	leaq	32(%rsp), %r15
	movq	%r15, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev@PLT
.Ltmp628:                               # EH_LABEL
	leaq	.L.str.6(%rip), %rsi
	movl	$81, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp629:                               # EH_LABEL
# %bb.2:
.Ltmp630:                               # EH_LABEL
	leaq	32(%rsp), %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c@PLT
.Ltmp631:                               # EH_LABEL
# %bb.3:
.Ltmp632:                               # EH_LABEL
	movq	%rax, %rdi
	movl	%ebp, %esi
	callq	_ZNSolsEi@PLT
.Ltmp633:                               # EH_LABEL
# %bb.4:
.Ltmp634:                               # EH_LABEL
	movq	%rax, %r15
	leaq	.L.str.35(%rip), %rsi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp635:                               # EH_LABEL
# %bb.5:
.Ltmp636:                               # EH_LABEL
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
.Ltmp637:                               # EH_LABEL
# %bb.6:
.Ltmp638:                               # EH_LABEL
	movq	%rax, %r14
	leaq	.L.str.35(%rip), %rsi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp639:                               # EH_LABEL
# %bb.7:
.Ltmp640:                               # EH_LABEL
	movl	%ebx, %edi
	callq	hipGetErrorString@PLT
.Ltmp641:                               # EH_LABEL
# %bb.8:
.Ltmp642:                               # EH_LABEL
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
.Ltmp643:                               # EH_LABEL
# %bb.9:
.Ltmp645:                               # EH_LABEL
	movq	%rsp, %rdi
	leaq	32(%rsp), %rsi
	callq	_ZNKRSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv@PLT
.Ltmp646:                               # EH_LABEL
# %bb.10:
.Ltmp648:                               # EH_LABEL
	movq	%rsp, %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp649:                               # EH_LABEL
# %bb.11:
.LBB2_14:
.Ltmp650:                               # EH_LABEL
	movq	%rax, %rbx
	movq	(%rsp), %rdi
	leaq	16(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB2_16
# %bb.15:
	movq	16(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	leaq	32(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@PLT
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB2_13:
.Ltmp647:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	32(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@PLT
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB2_12:
.Ltmp644:                               # EH_LABEL
	movq	%rax, %rbx
.LBB2_16:
	leaq	32(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@PLT
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end2:
	.size	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i, .Lfunc_end2-_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table2:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp628-.Lfunc_begin1         #   Call between .Lfunc_begin1 and .Ltmp628
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp628-.Lfunc_begin1         # >> Call Site 2 <<
	.uleb128 .Ltmp643-.Ltmp628              #   Call between .Ltmp628 and .Ltmp643
	.uleb128 .Ltmp644-.Lfunc_begin1         #     jumps to .Ltmp644
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp645-.Lfunc_begin1         # >> Call Site 3 <<
	.uleb128 .Ltmp646-.Ltmp645              #   Call between .Ltmp645 and .Ltmp646
	.uleb128 .Ltmp647-.Lfunc_begin1         #     jumps to .Ltmp647
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp648-.Lfunc_begin1         # >> Call Site 4 <<
	.uleb128 .Ltmp649-.Ltmp648              #   Call between .Ltmp648 and .Ltmp649
	.uleb128 .Ltmp650-.Lfunc_begin1         #     jumps to .Ltmp650
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp649-.Lfunc_begin1         # >> Call Site 5 <<
	.uleb128 .Lfunc_end2-.Ltmp649           #   Call between .Ltmp649 and .Lfunc_end2
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end3, nop     # -- Begin function _ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.type	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,@function
_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE: # @_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception2
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rdi, %r14
	movl	$16, %edi
	callq	__cxa_allocate_exception@PLT
	movq	%rax, %rbx
.Ltmp651:                               # EH_LABEL
	movq	%rax, %rdi
	movq	%r14, %rsi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@PLT
.Ltmp652:                               # EH_LABEL
# %bb.1:
	movq	_ZTISt13runtime_error@GOTPCREL(%rip), %rsi
	movq	_ZNSt13runtime_errorD1Ev@GOTPCREL(%rip), %rdx
	movq	%rbx, %rdi
	callq	__cxa_throw@PLT
.LBB3_2:
.Ltmp653:                               # EH_LABEL
	movq	%rax, %r14
	movq	%rbx, %rdi
	callq	__cxa_free_exception@PLT
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end3:
	.size	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .Lfunc_end3-_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table3:
.Lexception2:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.uleb128 .Lfunc_begin2-.Lfunc_begin2    # >> Call Site 1 <<
	.uleb128 .Ltmp651-.Lfunc_begin2         #   Call between .Lfunc_begin2 and .Ltmp651
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp651-.Lfunc_begin2         # >> Call Site 2 <<
	.uleb128 .Ltmp652-.Ltmp651              #   Call between .Ltmp651 and .Ltmp652
	.uleb128 .Ltmp653-.Lfunc_begin2         #     jumps to .Ltmp653
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp652-.Lfunc_begin2         # >> Call Site 3 <<
	.uleb128 .Lfunc_end3-.Ltmp652           #   Call between .Ltmp652 and .Lfunc_end3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end2:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_,"axG",@progbits,_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_,comdat
	.weak	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_ # -- Begin function _ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_
	.prefalign	4, .Lfunc_end4, nop
	.type	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_,@function
_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_: # @_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %rbx
	movq	%rsi, %r14
	movq	%rdi, %r15
	movq	%rsi, %rdi
	callq	strlen@PLT
	movq	(%rbx), %rcx
	movq	8(%rbx), %r8
	leaq	15(%rsp), %r9
	movq	%r15, %rdi
	movq	%r14, %rsi
	movq	%rax, %rdx
	callq	_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE
	movq	%r15, %rax
	addq	$16, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_, .Lfunc_end4-_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end5, nop     # -- Begin function _ZN12_GLOBAL__N_19read_lineB5cxx11EPKc
	.type	_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc,@function
_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc: # @_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc
.Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception3
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$528, %rsp                      # imm = 0x210
	.cfi_def_cfa_offset 560
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdi, %rbx
	leaq	8(%rsp), %rdi
	movl	$8, %edx
	callq	_ZNSt14basic_ifstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@PLT
	leaq	16(%rbx), %r15
	movq	%r15, (%rbx)
	movq	$0, 8(%rbx)
	movb	$0, 16(%rbx)
	movq	8(%rsp), %rax
	movq	-24(%rax), %rax
	movq	248(%rsp,%rax), %r14
	testq	%r14, %r14
	je	.LBB5_1
# %bb.3:
	cmpb	$0, 56(%r14)
	je	.LBB5_5
# %bb.4:
	movzbl	67(%r14), %eax
	jmp	.LBB5_7
.LBB5_5:
.Ltmp654:                               # EH_LABEL
	movq	%r14, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
.Ltmp655:                               # EH_LABEL
# %bb.6:
	movq	(%r14), %rax
.Ltmp656:                               # EH_LABEL
	movq	%r14, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.Ltmp657:                               # EH_LABEL
.LBB5_7:
.Ltmp658:                               # EH_LABEL
	movsbl	%al, %edx
	leaq	8(%rsp), %rdi
	movq	%rbx, %rsi
	callq	_ZSt7getlineIcSt11char_traitsIcESaIcEERSt13basic_istreamIT_T0_ES7_RNSt7__cxx1112basic_stringIS4_S5_T1_EES4_@PLT
.Ltmp659:                               # EH_LABEL
# %bb.8:
	leaq	8(%rsp), %rdi
	callq	_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@PLT
	addq	$528, %rsp                      # imm = 0x210
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB5_1:
	.cfi_def_cfa_offset 560
.Ltmp660:                               # EH_LABEL
	callq	_ZSt16__throw_bad_castv@PLT
.Ltmp661:                               # EH_LABEL
# %bb.2:
.LBB5_9:
.Ltmp662:                               # EH_LABEL
	movq	%rax, %r14
	movq	(%rbx), %rdi
	cmpq	%r15, %rdi
	je	.LBB5_11
# %bb.10:
	movq	(%r15), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB5_11:
	leaq	8(%rsp), %rdi
	callq	_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@PLT
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end5:
	.size	_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc, .Lfunc_end5-_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table5:
.Lexception3:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end3-.Lcst_begin3
.Lcst_begin3:
	.uleb128 .Lfunc_begin3-.Lfunc_begin3    # >> Call Site 1 <<
	.uleb128 .Ltmp654-.Lfunc_begin3         #   Call between .Lfunc_begin3 and .Ltmp654
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp654-.Lfunc_begin3         # >> Call Site 2 <<
	.uleb128 .Ltmp661-.Ltmp654              #   Call between .Ltmp654 and .Ltmp661
	.uleb128 .Ltmp662-.Lfunc_begin3         #     jumps to .Ltmp662
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp661-.Lfunc_begin3         # >> Call Site 3 <<
	.uleb128 .Lfunc_end5-.Ltmp661           #   Call between .Ltmp661 and .Lfunc_end5
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end3:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.p2align	1                               # -- Begin function _ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev
	.prefalign	4, .Lfunc_end6, nop
	.type	_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev,@function
_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev: # @_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev
.Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception4
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	testq	%rdi, %rdi
	je	.LBB6_2
# %bb.1:
.Ltmp663:                               # EH_LABEL
	callq	hipFree@PLT
.Ltmp664:                               # EH_LABEL
.LBB6_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.LBB6_3:
	.cfi_def_cfa_offset 16
.Ltmp665:                               # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end6:
	.size	_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev, .Lfunc_end6-_ZN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16ED2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table6:
.Lexception4:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase1-.Lttbaseref1
.Lttbaseref1:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end4-.Lcst_begin4
.Lcst_begin4:
	.uleb128 .Ltmp663-.Lfunc_begin4         # >> Call Site 1 <<
	.uleb128 .Ltmp664-.Ltmp663              #   Call between .Ltmp663 and .Ltmp664
	.uleb128 .Ltmp665-.Lfunc_begin4         #     jumps to .Ltmp665
	.byte	1                               #   On action: 1
.Lcst_end4:
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
	.prefalign	4, .Lfunc_end7, nop     # -- Begin function _ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
	.type	_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t,@function
_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t: # @_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$160, %rsp
	.cfi_def_cfa_offset 192
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdi, %rbx
	callq	_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE
	movl	%eax, %ecx
	movl	$1, %eax
	testb	%cl, %cl
	je	.LBB7_10
# %bb.1:
	movabsq	$4294967552, %r14               # imm = 0x100000100
	leaq	-208(%r14), %r15
	movq	%r15, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	je	.LBB7_2
# %bb.3:
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB7_10
	jmp	.LBB7_4
.LBB7_2:
	movq	(%rbx), %rax
	movq	8(%rbx), %rcx
	movq	40(%rbx), %rdx
	movq	%rax, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	24(%rsp), %rdi
	leaq	8(%rsp), %rsi
	leaq	48(%rsp), %rdx
	leaq	40(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	24(%rsp), %rsi
	movl	32(%rsp), %edx
	movq	8(%rsp), %rcx
	movl	16(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	56(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB7_10
.LBB7_4:
	movq	%r15, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	je	.LBB7_5
# %bb.6:
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB7_10
	jmp	.LBB7_7
.LBB7_5:
	movq	(%rbx), %rax
	movq	16(%rbx), %rcx
	movq	48(%rbx), %rdx
	movq	%rax, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	24(%rsp), %rdi
	leaq	8(%rsp), %rsi
	leaq	48(%rsp), %rdx
	leaq	40(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	24(%rsp), %rsi
	movl	32(%rsp), %edx
	movq	8(%rsp), %rcx
	movl	16(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	56(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB7_10
.LBB7_7:
	leaq	-255(%r14), %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB7_9
# %bb.8:
	movq	40(%rbx), %rax
	movq	48(%rbx), %rcx
	movq	24(%rbx), %rdx
	movq	32(%rbx), %rsi
	movq	56(%rbx), %rdi
	movq	64(%rbx), %r8
	movq	%rax, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rsi, 48(%rsp)
	movq	%rdi, 40(%rsp)
	movq	%r8, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	152(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	24(%rsp), %rdi
	leaq	8(%rsp), %rsi
	leaq	144(%rsp), %rdx
	leaq	136(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	24(%rsp), %rsi
	movl	32(%rsp), %edx
	movq	8(%rsp), %rcx
	movl	16(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	152(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB7_9:
	callq	hipGetLastError@PLT
.LBB7_10:
	addq	$160, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end7:
	.size	_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t, .Lfunc_end7-_ZN12_GLOBAL__N_113launch_serialERKNS_9ArgumentsEP12ihipStream_t
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end8, nop     # -- Begin function _ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
	.type	_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t,@function
_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t: # @_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$152, %rsp
	.cfi_def_cfa_offset 176
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rdi, %rbx
	callq	_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE
	movl	%eax, %ecx
	movl	$1, %eax
	testb	%cl, %cl
	je	.LBB8_7
# %bb.1:
	movabsq	$4294967552, %r14               # imm = 0x100000100
	leaq	-160(%r14), %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	je	.LBB8_2
# %bb.3:
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	je	.LBB8_4
.LBB8_7:
	addq	$152, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB8_2:
	.cfi_def_cfa_offset 176
	movq	(%rbx), %rax
	movq	8(%rbx), %rcx
	movq	16(%rbx), %rdx
	movq	40(%rbx), %rsi
	movq	48(%rbx), %rdi
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdi, 56(%rsp)
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
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	48(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	56(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB8_7
.LBB8_4:
	leaq	-255(%r14), %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB8_6
# %bb.5:
	movq	40(%rbx), %rax
	movq	48(%rbx), %rcx
	movq	24(%rbx), %rdx
	movq	32(%rbx), %rsi
	movq	56(%rbx), %rdi
	movq	64(%rbx), %r8
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdi, 56(%rsp)
	movq	%r8, 48(%rsp)
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
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	leaq	144(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	144(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB8_6:
	callq	hipGetLastError@PLT
	addq	$152, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end8:
	.size	_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t, .Lfunc_end8-_ZN12_GLOBAL__N_115launch_combinedERKNS_9ArgumentsEP12ihipStream_t
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end9, nop     # -- Begin function _ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t
	.type	_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t,@function
_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t: # @_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$208, %rsp
	.cfi_def_cfa_offset 224
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	callq	_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE
	movl	%eax, %ecx
	movl	$1, %eax
	testb	%cl, %cl
	je	.LBB9_4
# %bb.1:
	movabsq	$4294967344, %rdi               # imm = 0x100000030
	leaq	208(%rdi), %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB9_3
# %bb.2:
	movq	(%rbx), %rax
	movq	8(%rbx), %rcx
	movq	16(%rbx), %rdx
	movq	24(%rbx), %rsi
	movq	32(%rbx), %rdi
	movq	40(%rbx), %r8
	movq	48(%rbx), %r9
	movq	56(%rbx), %r10
	movq	64(%rbx), %r11
	movq	%rax, 120(%rsp)
	movq	%rcx, 112(%rsp)
	movq	%rdx, 104(%rsp)
	movq	%rsi, 96(%rsp)
	movq	%rdi, 88(%rsp)
	movq	%r8, 80(%rsp)
	movq	%r9, 72(%rsp)
	movq	%r10, 64(%rsp)
	movq	%r11, 56(%rsp)
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
	leaq	64(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB9_3:
	callq	hipGetLastError@PLT
.LBB9_4:
	addq	$208, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end9:
	.size	_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t, .Lfunc_end9-_ZN12_GLOBAL__N_112launch_fusedERKNS_9ArgumentsEP12ihipStream_t
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end10, nop    # -- Begin function _ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm
	.type	_ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm,@function
_ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm: # @_ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm
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
	leaq	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end10:
	.size	_ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm, .Lfunc_end10-_ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function _ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE
.LCPI11_0:
	.quad	0x4048000000000000              # double 48
.LCPI11_1:
	.quad	0x3fc999999999999a              # double 0.20000000000000001
	.text
	.prefalign	4, .Lfunc_end11, nop
	.type	_ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE,@function
_ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE: # @_ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE
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
	subq	$5192, %rsp                     # imm = 0x1448
	.cfi_def_cfa_offset 5248
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movsd	%xmm1, 696(%rsp)                # 8-byte Spill
	movsd	%xmm0, 688(%rsp)                # 8-byte Spill
	movq	%r9, 712(%rsp)                  # 8-byte Spill
	movq	%r8, 680(%rsp)                  # 8-byte Spill
	movq	%rcx, %rbx
	movq	%rsi, 8(%rsp)                   # 8-byte Spill
	movq	5256(%rsp), %r15
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	movq	(%r15), %r12
	movq	8(%r15), %r13
	subq	%r12, %r13
	movq	%rdi, 672(%rsp)                 # 8-byte Spill
	movq	%rdx, 704(%rsp)                 # 8-byte Spill
	je	.LBB11_4
# %bb.1:
	cmpq	%rax, %r13
	ja	.LBB11_351
# %bb.2:
	movq	%r13, %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbp
	movq	%rax, %r14
	addq	%r13, %r14
	cmpq	$5, %r13
	jb	.LBB11_383
# %bb.3:
	movq	%rbp, %rdi
	movq	%r12, %rsi
	movq	%r13, %rdx
	callq	memmove@PLT
	jmp	.LBB11_5
.LBB11_4:
	xorl	%ebp, %ebp
	movq	%r13, %r14
.LBB11_5:
	movq	%r14, %r12
	subq	%rbp, %r12
	sarq	$2, %r12
	movq	%r14, %r13
	subq	%rbp, %r13
	je	.LBB11_8
# %bb.6:
	bsrq	%r12, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp666:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp667:                               # EH_LABEL
# %bb.7:
.Ltmp668:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp669:                               # EH_LABEL
.LBB11_8:
	andq	$-2, %r12
	movss	(%rbp,%r12,2), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 124(%rsp)                # 4-byte Spill
	movq	%rbp, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
	movq	24(%r15), %r12
	movq	32(%r15), %r13
	subq	%r12, %r13
	je	.LBB11_12
# %bb.9:
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r13
	ja	.LBB11_351
# %bb.10:
	movq	%r13, %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbp
	movq	%rax, %r14
	addq	%r13, %r14
	cmpq	$5, %r13
	jb	.LBB11_385
# %bb.11:
	movq	%rbp, %rdi
	movq	%r12, %rsi
	movq	%r13, %rdx
	callq	memmove@PLT
	jmp	.LBB11_13
.LBB11_12:
	xorl	%ebp, %ebp
	movq	%r13, %r14
.LBB11_13:
	movq	%r14, %r12
	subq	%rbp, %r12
	sarq	$2, %r12
	movq	%r14, %r13
	subq	%rbp, %r13
	je	.LBB11_16
# %bb.14:
	bsrq	%r12, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp671:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp672:                               # EH_LABEL
# %bb.15:
.Ltmp673:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp674:                               # EH_LABEL
.LBB11_16:
	andq	$-2, %r12
	movss	(%rbp,%r12,2), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rsp)                   # 4-byte Spill
	movq	%rbp, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
	movq	48(%r15), %r12
	movq	56(%r15), %r13
	subq	%r12, %r13
	je	.LBB11_20
# %bb.17:
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r13
	ja	.LBB11_351
# %bb.18:
	movq	%r13, %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbp
	movq	%rax, %r14
	addq	%r13, %r14
	cmpq	$5, %r13
	jb	.LBB11_387
# %bb.19:
	movq	%rbp, %rdi
	movq	%r12, %rsi
	movq	%r13, %rdx
	callq	memmove@PLT
	jmp	.LBB11_21
.LBB11_20:
	xorl	%ebp, %ebp
	movq	%r13, %r14
.LBB11_21:
	movq	%r14, %r12
	subq	%rbp, %r12
	sarq	$2, %r12
	movq	%r14, %r13
	subq	%rbp, %r13
	je	.LBB11_24
# %bb.22:
	bsrq	%r12, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp676:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp677:                               # EH_LABEL
# %bb.23:
.Ltmp678:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp679:                               # EH_LABEL
.LBB11_24:
	andq	$-2, %r12
	movss	(%rbp,%r12,2), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 52(%rsp)                 # 4-byte Spill
	movq	%rbp, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
	leaq	1096(%rsp), %r14
	movl	$4096, %edx                     # imm = 0x1000
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	.L.str.123(%rip), %rdi
	movl	$4095, %edx                     # imm = 0xFFF
	movq	%r14, %rsi
	callq	readlink@PLT
	testq	%rax, %rax
	jle	.LBB11_367
# %bb.25:
	movq	%rax, %r14
	leaq	104(%rsp), %r12
	movq	%r12, 88(%rsp)
	cmpq	$16, %rax
	jb	.LBB11_28
# %bb.26:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB11_352
# %bb.27:
	callq	_Znwm@PLT
	movq	%rax, %r12
	movq	%rax, 88(%rsp)
	movq	%r14, 104(%rsp)
	jmp	.LBB11_30
.LBB11_28:
	cmpq	$1, %r14
	jne	.LBB11_30
# %bb.29:
	movzbl	1096(%rsp), %eax
	movb	%al, 104(%rsp)
	jmp	.LBB11_31
.LBB11_30:
	leaq	1096(%rsp), %rsi
	movq	%r12, %rdi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB11_31:
	movq	%r14, 96(%rsp)
	movb	$0, (%r12,%r14)
.Ltmp681:                               # EH_LABEL
	leaq	720(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev@PLT
.Ltmp682:                               # EH_LABEL
# %bb.32:
	movq	720(%rsp), %rax
	movq	-24(%rax), %rcx
	movl	$-261, %edx                     # imm = 0xFEFB
	andl	744(%rsp,%rcx), %edx
	orl	$4, %edx
	movl	%edx, 744(%rsp,%rcx)
	movq	-24(%rax), %rax
	movq	$9, 728(%rsp,%rax)
.Ltmp684:                               # EH_LABEL
	leaq	.L.str.48(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$67, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp685:                               # EH_LABEL
	movq	712(%rsp), %r12                 # 8-byte Reload
# %bb.33:
.Ltmp686:                               # EH_LABEL
	leaq	.L.str.49(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp687:                               # EH_LABEL
# %bb.34:
	movq	(%rbx), %rsi
	movq	8(%rbx), %rdx
.Ltmp688:                               # EH_LABEL
	leaq	720(%rsp), %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp689:                               # EH_LABEL
# %bb.35:
.Ltmp690:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.50(%rip), %rsi
	movl	$3, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp691:                               # EH_LABEL
# %bb.36:
.Ltmp692:                               # EH_LABEL
	leaq	.L.str.51(%rip), %rsi
	movl	$31, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp693:                               # EH_LABEL
# %bb.37:
	leaq	144(%rsp), %r13
	movq	%r13, 128(%rsp)
	movq	%r12, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	cmpq	$16, %rax
	jb	.LBB11_42
# %bb.38:
	testq	%r14, %r14
	js	.LBB11_379
# %bb.39:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB11_353
# %bb.40:
.Ltmp695:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp696:                               # EH_LABEL
# %bb.41:
	movq	%rax, %r13
	movq	%rax, 128(%rsp)
	movq	%r14, 144(%rsp)
.LBB11_42:
	testq	%r14, %r14
	je	.LBB11_46
# %bb.43:
	cmpq	$1, %r14
	jne	.LBB11_45
# %bb.44:
	movzbl	(%r12), %eax
	movb	%al, (%r13)
	jmp	.LBB11_46
.LBB11_45:
	movq	%r13, %rdi
	movq	%r12, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB11_46:
	movq	%r14, 136(%rsp)
	movb	$0, (%r13,%r14)
	movq	128(%rsp), %rsi
	movq	136(%rsp), %rdx
.Ltmp697:                               # EH_LABEL
	leaq	16(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp698:                               # EH_LABEL
# %bb.47:
	movq	16(%rsp), %rsi
	movq	24(%rsp), %rdx
.Ltmp700:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp701:                               # EH_LABEL
# %bb.48:
.Ltmp702:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.52(%rip), %rsi
	movl	$16, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp703:                               # EH_LABEL
# %bb.49:
	leaq	1160(%r12), %rbp
	leaq	176(%rsp), %r13
	movq	%r13, 160(%rsp)
	movq	%rbp, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	cmpq	$16, %rax
	jb	.LBB11_54
# %bb.50:
	testq	%r14, %r14
	js	.LBB11_381
# %bb.51:
	movq	%r14, %rdi
	incq	%rdi
	js	.LBB11_355
# %bb.52:
.Ltmp705:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp706:                               # EH_LABEL
# %bb.53:
	movq	%rax, %r13
	movq	%rax, 160(%rsp)
	movq	%r14, 176(%rsp)
.LBB11_54:
	testq	%r14, %r14
	je	.LBB11_58
# %bb.55:
	cmpq	$1, %r14
	jne	.LBB11_57
# %bb.56:
	movzbl	(%rbp), %eax
	movb	%al, (%r13)
	jmp	.LBB11_58
.LBB11_57:
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB11_58:
	movq	%r14, 168(%rsp)
	movb	$0, (%r13,%r14)
	movq	160(%rsp), %rsi
	movq	168(%rsp), %rdx
.Ltmp707:                               # EH_LABEL
	leaq	640(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp708:                               # EH_LABEL
# %bb.59:
	movq	640(%rsp), %rsi
	movq	648(%rsp), %rdx
.Ltmp710:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp711:                               # EH_LABEL
	movq	704(%rsp), %r14                 # 8-byte Reload
# %bb.60:
.Ltmp712:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.53(%rip), %rsi
	movl	$17, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp713:                               # EH_LABEL
# %bb.61:
	movl	592(%r12), %esi
.Ltmp714:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSolsEi@PLT
.Ltmp715:                               # EH_LABEL
# %bb.62:
.Ltmp716:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.54(%rip), %rsi
	movl	$13, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp717:                               # EH_LABEL
# %bb.63:
	movl	584(%r12), %esi
.Ltmp718:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSolsEi@PLT
.Ltmp719:                               # EH_LABEL
# %bb.64:
.Ltmp720:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.55(%rip), %rsi
	movl	$16, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp721:                               # EH_LABEL
# %bb.65:
	movl	588(%r12), %esi
.Ltmp722:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSolsEi@PLT
.Ltmp723:                               # EH_LABEL
# %bb.66:
.Ltmp724:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.56(%rip), %rsi
	movl	$25, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp725:                               # EH_LABEL
# %bb.67:
.Ltmp726:                               # EH_LABEL
	movl	5248(%rsp), %esi
	movq	%rbx, %rdi
	callq	_ZNSolsEi@PLT
.Ltmp727:                               # EH_LABEL
# %bb.68:
.Ltmp728:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.57(%rip), %rsi
	movl	$25, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp729:                               # EH_LABEL
# %bb.69:
.Ltmp731:                               # EH_LABEL
	movl	$35, %edi
	callq	_Znwm@PLT
.Ltmp732:                               # EH_LABEL
# %bb.70:
	movq	%rax, %r12
	movups	.L.str.58+16(%rip), %xmm0
	movups	%xmm0, 16(%rax)
	movups	.L.str.58(%rip), %xmm0
	movups	%xmm0, (%rax)
	movw	$12582, 32(%rax)                # imm = 0x3126
	movb	$0, 34(%rax)
.Ltmp734:                               # EH_LABEL
	leaq	576(%rsp), %rdi
	movq	%rax, %rsi
	callq	_ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp735:                               # EH_LABEL
# %bb.71:
	movq	576(%rsp), %rsi
	movq	584(%rsp), %rdx
.Ltmp737:                               # EH_LABEL
	leaq	608(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp738:                               # EH_LABEL
# %bb.72:
	movq	608(%rsp), %rsi
	movq	616(%rsp), %rdx
.Ltmp740:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp741:                               # EH_LABEL
# %bb.73:
.Ltmp742:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.59(%rip), %rsi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp743:                               # EH_LABEL
# %bb.74:
.Ltmp744:                               # EH_LABEL
	leaq	.L.str.60(%rip), %rsi
	movl	$37, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp745:                               # EH_LABEL
# %bb.75:
	movq	88(%rsp), %rsi
	movq	96(%rsp), %rdx
.Ltmp747:                               # EH_LABEL
	leaq	544(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp748:                               # EH_LABEL
# %bb.76:
	movq	544(%rsp), %rsi
	movq	552(%rsp), %rdx
.Ltmp750:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp751:                               # EH_LABEL
# %bb.77:
.Ltmp752:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.61(%rip), %rsi
	movl	$25, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp753:                               # EH_LABEL
# %bb.78:
.Ltmp755:                               # EH_LABEL
	leaq	512(%rsp), %rdi
	leaq	88(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp756:                               # EH_LABEL
# %bb.79:
	movq	512(%rsp), %rsi
	movq	520(%rsp), %rdx
.Ltmp758:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp759:                               # EH_LABEL
# %bb.80:
.Ltmp760:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.62(%rip), %rsi
	movl	$80, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp761:                               # EH_LABEL
# %bb.81:
.Ltmp763:                               # EH_LABEL
	movl	$41, %edi
	callq	_Znwm@PLT
.Ltmp764:                               # EH_LABEL
# %bb.82:
	leaq	464(%rsp), %r13
	movq	%rax, 448(%rsp)
	movq	$40, 464(%rsp)
	movups	.L.str.63+16(%rip), %xmm0
	movups	%xmm0, 16(%rax)
	movups	.L.str.63(%rip), %xmm0
	movups	%xmm0, (%rax)
	movabsq	$8100119953398658417, %rcx      # imm = 0x7069682E6C617571
	movq	%rcx, 32(%rax)
	movq	$40, 456(%rsp)
	movb	$0, 40(%rax)
.Ltmp766:                               # EH_LABEL
	leaq	480(%rsp), %rdi
	leaq	448(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp767:                               # EH_LABEL
# %bb.83:
	movq	480(%rsp), %rsi
	movq	488(%rsp), %rdx
.Ltmp769:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp770:                               # EH_LABEL
# %bb.84:
.Ltmp771:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.64(%rip), %rsi
	movl	$89, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp772:                               # EH_LABEL
# %bb.85:
.Ltmp774:                               # EH_LABEL
	movl	$48, %edi
	callq	_Znwm@PLT
.Ltmp775:                               # EH_LABEL
# %bb.86:
	leaq	400(%rsp), %rbp
	movq	%rax, 384(%rsp)
	movq	$47, 400(%rsp)
	movups	.L.str.65+31(%rip), %xmm0
	movups	%xmm0, 31(%rax)
	movups	.L.str.65+16(%rip), %xmm0
	movups	%xmm0, 16(%rax)
	movups	.L.str.65(%rip), %xmm0
	movups	%xmm0, (%rax)
	movq	$47, 392(%rsp)
	movb	$0, 47(%rax)
.Ltmp777:                               # EH_LABEL
	leaq	416(%rsp), %rdi
	leaq	384(%rsp), %rsi
	callq	_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp778:                               # EH_LABEL
# %bb.87:
	movq	416(%rsp), %rsi
	movq	424(%rsp), %rdx
.Ltmp780:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp781:                               # EH_LABEL
# %bb.88:
.Ltmp782:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.66(%rip), %rsi
	movl	$21, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp783:                               # EH_LABEL
# %bb.89:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	(%rax), %rsi
	movq	8(%rax), %rdx
.Ltmp785:                               # EH_LABEL
	leaq	352(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp786:                               # EH_LABEL
# %bb.90:
	movq	352(%rsp), %rsi
	movq	360(%rsp), %rdx
.Ltmp788:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp789:                               # EH_LABEL
# %bb.91:
.Ltmp790:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.67(%rip), %rsi
	movl	$23, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp791:                               # EH_LABEL
# %bb.92:
.Ltmp793:                               # EH_LABEL
	leaq	320(%rsp), %rdi
	movq	8(%rsp), %rsi                   # 8-byte Reload
	callq	_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp794:                               # EH_LABEL
# %bb.93:
	movq	320(%rsp), %rsi
	movq	328(%rsp), %rdx
.Ltmp796:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp797:                               # EH_LABEL
# %bb.94:
.Ltmp798:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.68(%rip), %rsi
	movl	$27, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp799:                               # EH_LABEL
# %bb.95:
	movq	(%r14), %rsi
	movq	8(%r14), %rdx
.Ltmp801:                               # EH_LABEL
	leaq	288(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp802:                               # EH_LABEL
# %bb.96:
	movq	288(%rsp), %rsi
	movq	296(%rsp), %rdx
.Ltmp804:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp805:                               # EH_LABEL
# %bb.97:
.Ltmp806:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.69(%rip), %rsi
	movl	$29, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp807:                               # EH_LABEL
# %bb.98:
.Ltmp809:                               # EH_LABEL
	leaq	256(%rsp), %rdi
	movq	%r14, %rsi
	callq	_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp810:                               # EH_LABEL
# %bb.99:
	movq	256(%rsp), %rsi
	movq	264(%rsp), %rdx
.Ltmp812:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp813:                               # EH_LABEL
# %bb.100:
.Ltmp814:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.70(%rip), %rsi
	movl	$22, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp815:                               # EH_LABEL
# %bb.101:
.Ltmp817:                               # EH_LABEL
	leaq	1096(%rsp), %rdi
	movq	%r14, %rsi
	movl	$8, %edx
	callq	_ZNSt14basic_ifstreamIcSt11char_traitsIcEEC1ERKNSt7__cxx1112basic_stringIcS1_SaIcEEESt13_Ios_Openmode@PLT
.Ltmp818:                               # EH_LABEL
# %bb.102:
	movq	1096(%rsp), %rax
	movq	-24(%rax), %rax
	testb	$5, 1128(%rsp,%rax)
	jne	.LBB11_369
# %bb.103:
	movq	1328(%rsp,%rax), %rsi
	leaq	208(%rsp), %r14
	movq	%r14, 192(%rsp)
	movq	$0, 200(%rsp)
.Ltmp826:                               # EH_LABEL
	leaq	192(%rsp), %rdi
	movl	$-1, %edx
	xorl	%ecx, %ecx
	movl	$-1, %r8d
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag
.Ltmp827:                               # EH_LABEL
# %bb.104:
	leaq	1096(%rsp), %rdi
	callq	_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@PLT
	movq	192(%rsp), %rsi
	movq	200(%rsp), %rdx
.Ltmp829:                               # EH_LABEL
	leaq	224(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp830:                               # EH_LABEL
# %bb.105:
	movq	224(%rsp), %rsi
	movq	232(%rsp), %rdx
.Ltmp832:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp833:                               # EH_LABEL
# %bb.106:
.Ltmp834:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.59(%rip), %rsi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp835:                               # EH_LABEL
# %bb.107:
.Ltmp836:                               # EH_LABEL
	leaq	.L.str.71(%rip), %rsi
	movl	$186, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp837:                               # EH_LABEL
# %bb.108:
.Ltmp838:                               # EH_LABEL
	leaq	.L.str.72(%rip), %rsi
	movl	$70, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp839:                               # EH_LABEL
# %bb.109:
.Ltmp840:                               # EH_LABEL
	leaq	.L.str.73(%rip), %rsi
	movl	$73, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp841:                               # EH_LABEL
# %bb.110:
.Ltmp842:                               # EH_LABEL
	leaq	.L.str.74(%rip), %rsi
	movl	$31, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp843:                               # EH_LABEL
# %bb.111:
.Ltmp844:                               # EH_LABEL
	movl	$983040, %esi                   # imm = 0xF0000
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp845:                               # EH_LABEL
# %bb.112:
.Ltmp846:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.75(%rip), %rsi
	movl	$15, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp847:                               # EH_LABEL
# %bb.113:
.Ltmp848:                               # EH_LABEL
	movl	$10240, %esi                    # imm = 0x2800
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp849:                               # EH_LABEL
# %bb.114:
.Ltmp850:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.76(%rip), %rsi
	movl	$23, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp851:                               # EH_LABEL
# %bb.115:
.Ltmp852:                               # EH_LABEL
	movl	$192, %esi
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp853:                               # EH_LABEL
# %bb.116:
.Ltmp854:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.77(%rip), %rsi
	movl	$24, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp855:                               # EH_LABEL
# %bb.117:
.Ltmp856:                               # EH_LABEL
	movl	$384, %esi                      # imm = 0x180
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp857:                               # EH_LABEL
# %bb.118:
.Ltmp858:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.78(%rip), %rsi
	movl	$20, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp859:                               # EH_LABEL
# %bb.119:
.Ltmp860:                               # EH_LABEL
	movl	$384, %esi                      # imm = 0x180
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp861:                               # EH_LABEL
# %bb.120:
.Ltmp862:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.79(%rip), %rsi
	movl	$104, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp863:                               # EH_LABEL
# %bb.121:
.Ltmp864:                               # EH_LABEL
	leaq	.L.str.80(%rip), %rsi
	movl	$19, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp865:                               # EH_LABEL
# %bb.122:
.Ltmp866:                               # EH_LABEL
	movl	$1004672, %esi                  # imm = 0xF5480
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp867:                               # EH_LABEL
# %bb.123:
.Ltmp868:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.81(%rip), %rsi
	movl	$100, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp869:                               # EH_LABEL
# %bb.124:
.Ltmp870:                               # EH_LABEL
	leaq	.L.str.82(%rip), %rsi
	movl	$18, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp871:                               # EH_LABEL
# %bb.125:
.Ltmp872:                               # EH_LABEL
	movl	$994432, %esi                   # imm = 0xF2C80
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp873:                               # EH_LABEL
# %bb.126:
.Ltmp874:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.83(%rip), %rsi
	movl	$81, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp875:                               # EH_LABEL
# %bb.127:
.Ltmp876:                               # EH_LABEL
	leaq	.L.str.84(%rip), %rsi
	movl	$15, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp877:                               # EH_LABEL
# %bb.128:
.Ltmp878:                               # EH_LABEL
	movl	$994240, %esi                   # imm = 0xF2BC0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp879:                               # EH_LABEL
# %bb.129:
.Ltmp880:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.85(%rip), %rsi
	movl	$35, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp881:                               # EH_LABEL
# %bb.130:
.Ltmp882:                               # EH_LABEL
	leaq	.L.str.86(%rip), %rsi
	movl	$45, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp883:                               # EH_LABEL
# %bb.131:
.Ltmp884:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	696(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp885:                               # EH_LABEL
# %bb.132:
.Ltmp886:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.87(%rip), %rsi
	movl	$26, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp887:                               # EH_LABEL
# %bb.133:
.Ltmp888:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	688(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp889:                               # EH_LABEL
# %bb.134:
.Ltmp890:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.88(%rip), %rsi
	movl	$100, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp891:                               # EH_LABEL
# %bb.135:
.Ltmp892:                               # EH_LABEL
	leaq	.L.str.89(%rip), %rsi
	movl	$43, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp893:                               # EH_LABEL
# %bb.136:
.Ltmp894:                               # EH_LABEL
	movss	124(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movq	%rbx, %rdi
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp895:                               # EH_LABEL
# %bb.137:
.Ltmp896:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.90(%rip), %rsi
	movl	$21, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp897:                               # EH_LABEL
# %bb.138:
	movss	(%rsp), %xmm0                   # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp898:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp899:                               # EH_LABEL
# %bb.139:
.Ltmp900:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.91(%rip), %rsi
	movl	$18, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp901:                               # EH_LABEL
# %bb.140:
	movss	52(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp902:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp903:                               # EH_LABEL
# %bb.141:
.Ltmp904:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.92(%rip), %rsi
	movl	$3, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp905:                               # EH_LABEL
# %bb.142:
.Ltmp906:                               # EH_LABEL
	leaq	.L.str.93(%rip), %rsi
	movl	$60, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp907:                               # EH_LABEL
# %bb.143:
.Ltmp908:                               # EH_LABEL
	movss	52(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	minss	(%rsp), %xmm0                   # 4-byte Folded Reload
	cvtss2sd	%xmm0, %xmm1
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	subsd	%xmm1, %xmm0
	xorpd	%xmm1, %xmm1
	maxsd	%xmm1, %xmm0
	mulsd	.LCPI11_0(%rip), %xmm0
	movq	%rbx, %rdi
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp909:                               # EH_LABEL
# %bb.144:
.Ltmp910:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.94(%rip), %rsi
	movl	$40, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp911:                               # EH_LABEL
# %bb.145:
.Ltmp912:                               # EH_LABEL
	leaq	.L.str.95(%rip), %rsi
	movl	$12, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp913:                               # EH_LABEL
# %bb.146:
	xorl	%edx, %edx
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	ucomisd	.LCPI11_1(%rip), %xmm0
	leaq	.L.str.96(%rip), %rax
	leaq	.L.str.97(%rip), %rsi
	cmovaeq	%rax, %rsi
	adcq	$4, %rdx
.Ltmp914:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp915:                               # EH_LABEL
# %bb.147:
.Ltmp916:                               # EH_LABEL
	leaq	.L.str.92(%rip), %rsi
	movl	$3, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp917:                               # EH_LABEL
# %bb.148:
.Ltmp918:                               # EH_LABEL
	leaq	.L.str.98(%rip), %rsi
	movl	$90, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp919:                               # EH_LABEL
# %bb.149:
.Ltmp920:                               # EH_LABEL
	leaq	.L.str.99(%rip), %rsi
	movl	$95, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp921:                               # EH_LABEL
# %bb.150:
.Ltmp922:                               # EH_LABEL
	leaq	.L.str.100(%rip), %rsi
	movl	$48, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp923:                               # EH_LABEL
# %bb.151:
.Ltmp924:                               # EH_LABEL
	leaq	.L.str.101(%rip), %rsi
	movl	$18, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp925:                               # EH_LABEL
# %bb.152:
	movq	224(%rsp), %rdi
	leaq	240(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_154
# %bb.153:
	movq	240(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_154:
	movq	192(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.LBB11_156
# %bb.155:
	movq	208(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_156:
	movq	256(%rsp), %rdi
	leaq	272(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_158
# %bb.157:
	movq	272(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_158:
	movq	288(%rsp), %rdi
	leaq	304(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_160
# %bb.159:
	movq	304(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_160:
	movq	320(%rsp), %rdi
	leaq	336(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_162
# %bb.161:
	movq	336(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_162:
	movq	352(%rsp), %rdi
	leaq	368(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_164
# %bb.163:
	movq	368(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_164:
	movq	416(%rsp), %rdi
	leaq	432(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_166
# %bb.165:
	movq	432(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_166:
	movq	384(%rsp), %rdi
	cmpq	%rbp, %rdi
	je	.LBB11_168
# %bb.167:
	movq	400(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_168:
	movq	480(%rsp), %rdi
	leaq	496(%rsp), %rax
	cmpq	%rax, %rdi
	movabsq	$9223372036854775804, %rbp      # imm = 0x7FFFFFFFFFFFFFFC
	je	.LBB11_170
# %bb.169:
	movq	496(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_170:
	movq	448(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.LBB11_172
# %bb.171:
	movq	464(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_172:
	movq	512(%rsp), %rdi
	leaq	528(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_174
# %bb.173:
	movq	528(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_174:
	movq	544(%rsp), %rdi
	leaq	560(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_176
# %bb.175:
	movq	560(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_176:
	movq	608(%rsp), %rdi
	leaq	624(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_178
# %bb.177:
	movq	624(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_178:
	movq	576(%rsp), %rdi
	leaq	592(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_180
# %bb.179:
	movq	592(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_180:
	movl	$35, %esi
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
	movq	640(%rsp), %rdi
	leaq	656(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_182
# %bb.181:
	movq	656(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_182:
	movq	160(%rsp), %rdi
	leaq	176(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_184
# %bb.183:
	movq	176(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_184:
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_186
# %bb.185:
	movq	32(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_186:
	movq	128(%rsp), %rdi
	leaq	144(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_188
# %bb.187:
	movq	144(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_188:
.Ltmp927:                               # EH_LABEL
	leaq	.L.str.105(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$5, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp928:                               # EH_LABEL
# %bb.189:
.Ltmp929:                               # EH_LABEL
	leaq	.L.str.102(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$6, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp930:                               # EH_LABEL
# %bb.190:
.Ltmp931:                               # EH_LABEL
	leaq	.L.str.106(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$4, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp932:                               # EH_LABEL
# %bb.191:
	movq	(%r15), %rax
	cmpq	%rax, 8(%r15)
	je	.LBB11_198
# %bb.192:
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp933:                               # EH_LABEL
	leaq	720(%rsp), %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp934:                               # EH_LABEL
# %bb.193:
	movq	8(%r15), %rax
	subq	(%r15), %rax
	cmpq	$5, %rax
	jb	.LBB11_198
# %bb.194:
	movl	$1, %r12d
	leaq	.L.str.107(%rip), %rbx
	leaq	720(%rsp), %r14
	.p2align	4
.LBB11_195:                             # =>This Inner Loop Header: Depth=1
.Ltmp935:                               # EH_LABEL
	movl	$2, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp936:                               # EH_LABEL
# %bb.196:                              #   in Loop: Header=BB11_195 Depth=1
	movq	(%r15), %rax
	movss	(%rax,%r12,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp937:                               # EH_LABEL
	movq	%r14, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp938:                               # EH_LABEL
# %bb.197:                              #   in Loop: Header=BB11_195 Depth=1
	incq	%r12
	movq	8(%r15), %rax
	subq	(%r15), %rax
	sarq	$2, %rax
	cmpq	%rax, %r12
	jb	.LBB11_195
.LBB11_198:
.Ltmp940:                               # EH_LABEL
	leaq	.L.str.108(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp941:                               # EH_LABEL
# %bb.199:
.Ltmp942:                               # EH_LABEL
	leaq	.L.str.110(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$2, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp943:                               # EH_LABEL
# %bb.200:
.Ltmp944:                               # EH_LABEL
	leaq	.L.str.105(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$5, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp945:                               # EH_LABEL
# %bb.201:
.Ltmp946:                               # EH_LABEL
	leaq	.L.str.103(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$8, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp947:                               # EH_LABEL
# %bb.202:
.Ltmp948:                               # EH_LABEL
	leaq	.L.str.106(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$4, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp949:                               # EH_LABEL
# %bb.203:
	movq	24(%r15), %rax
	cmpq	%rax, 32(%r15)
	je	.LBB11_210
# %bb.204:
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp950:                               # EH_LABEL
	leaq	720(%rsp), %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp951:                               # EH_LABEL
# %bb.205:
	movq	32(%r15), %rax
	subq	24(%r15), %rax
	cmpq	$5, %rax
	jb	.LBB11_210
# %bb.206:
	movl	$1, %r12d
	leaq	.L.str.107(%rip), %rbx
	leaq	720(%rsp), %r14
	.p2align	4
.LBB11_207:                             # =>This Inner Loop Header: Depth=1
.Ltmp952:                               # EH_LABEL
	movl	$2, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp953:                               # EH_LABEL
# %bb.208:                              #   in Loop: Header=BB11_207 Depth=1
	movq	24(%r15), %rax
	movss	(%rax,%r12,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp954:                               # EH_LABEL
	movq	%r14, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp955:                               # EH_LABEL
# %bb.209:                              #   in Loop: Header=BB11_207 Depth=1
	incq	%r12
	movq	32(%r15), %rax
	subq	24(%r15), %rax
	sarq	$2, %rax
	cmpq	%rax, %r12
	jb	.LBB11_207
.LBB11_210:
.Ltmp957:                               # EH_LABEL
	leaq	.L.str.108(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp958:                               # EH_LABEL
# %bb.211:
.Ltmp959:                               # EH_LABEL
	leaq	.L.str.110(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$2, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp960:                               # EH_LABEL
# %bb.212:
.Ltmp961:                               # EH_LABEL
	leaq	.L.str.105(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$5, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp962:                               # EH_LABEL
# %bb.213:
.Ltmp963:                               # EH_LABEL
	leaq	.L.str.104(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$5, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp964:                               # EH_LABEL
# %bb.214:
.Ltmp965:                               # EH_LABEL
	leaq	.L.str.106(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$4, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp966:                               # EH_LABEL
# %bb.215:
	movq	48(%r15), %rax
	cmpq	%rax, 56(%r15)
	je	.LBB11_222
# %bb.216:
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp967:                               # EH_LABEL
	leaq	720(%rsp), %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp968:                               # EH_LABEL
# %bb.217:
	movq	56(%r15), %rax
	subq	48(%r15), %rax
	cmpq	$5, %rax
	jb	.LBB11_222
# %bb.218:
	movl	$1, %r12d
	leaq	.L.str.107(%rip), %rbx
	leaq	720(%rsp), %r14
	.p2align	4
.LBB11_219:                             # =>This Inner Loop Header: Depth=1
.Ltmp970:                               # EH_LABEL
	movl	$2, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp971:                               # EH_LABEL
# %bb.220:                              #   in Loop: Header=BB11_219 Depth=1
	movq	48(%r15), %rax
	movss	(%rax,%r12,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp972:                               # EH_LABEL
	movq	%r14, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp973:                               # EH_LABEL
# %bb.221:                              #   in Loop: Header=BB11_219 Depth=1
	incq	%r12
	movq	56(%r15), %rax
	subq	48(%r15), %rax
	sarq	$2, %rax
	cmpq	%rax, %r12
	jb	.LBB11_219
.LBB11_222:
.Ltmp975:                               # EH_LABEL
	leaq	.L.str.108(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp976:                               # EH_LABEL
# %bb.223:
.Ltmp977:                               # EH_LABEL
	leaq	.L.str.109(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp978:                               # EH_LABEL
# %bb.224:
.Ltmp980:                               # EH_LABEL
	leaq	.L.str.111(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$32, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp981:                               # EH_LABEL
# %bb.225:
	movq	$0, 8(%rsp)                     # 8-byte Folded Spill
	leaq	720(%rsp), %r14
	.p2align	4
.LBB11_226:                             # =>This Loop Header: Depth=1
                                        #     Child Loop BB11_231 Depth 2
                                        #     Child Loop BB11_257 Depth 2
                                        #     Child Loop BB11_283 Depth 2
.Ltmp982:                               # EH_LABEL
	movl	$5, %edx
	movq	%r14, %rdi
	leaq	.L.str.105(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp983:                               # EH_LABEL
# %bb.227:                              #   in Loop: Header=BB11_226 Depth=1
	movq	8(%rsp), %rax                   # 8-byte Reload
	leaq	.L__const._ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE.names(%rip), %rcx
	movq	(%rcx,%rax,8), %rbx
	movq	%rbx, %rdi
	callq	strlen@PLT
.Ltmp984:                               # EH_LABEL
	movq	%r14, %rdi
	movq	%rbx, %rsi
	movq	%rax, %rdx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp985:                               # EH_LABEL
# %bb.228:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp986:                               # EH_LABEL
	movl	$4, %edx
	movq	%r14, %rdi
	leaq	.L.str.106(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp987:                               # EH_LABEL
# %bb.229:                              #   in Loop: Header=BB11_226 Depth=1
	movq	5264(%rsp), %rax
	movq	(%rax), %r14
	movq	8(%rax), %rbx
	cmpq	%rbx, %r14
	je	.LBB11_248
# %bb.230:                              #   in Loop: Header=BB11_226 Depth=1
	movq	$0, (%rsp)                      # 8-byte Folded Spill
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	.p2align	4
.LBB11_231:                             #   Parent Loop BB11_226 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	8(%rsp), %rax                   # 8-byte Reload
	cmpl	%eax, 24(%r14)
	jne	.LBB11_235
# %bb.232:                              #   in Loop: Header=BB11_231 Depth=2
	cmpq	$0, 32(%r14)
	jne	.LBB11_235
# %bb.233:                              #   in Loop: Header=BB11_231 Depth=2
	movq	(%rsp), %rax                    # 8-byte Reload
	cmpq	%rax, %r12
	je	.LBB11_237
# %bb.234:                              #   in Loop: Header=BB11_231 Depth=2
	movss	40(%r14), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%r12)
	addq	$4, %r12
	.p2align	4
.LBB11_235:                             #   in Loop: Header=BB11_231 Depth=2
	movq	%r13, %r15
.LBB11_236:                             #   in Loop: Header=BB11_231 Depth=2
	addq	$48, %r14
	movq	%r15, %r13
	cmpq	%rbx, %r14
	jne	.LBB11_231
	jmp	.LBB11_249
.LBB11_237:                             #   in Loop: Header=BB11_231 Depth=2
	movq	%rax, %r12
	subq	%r13, %r12
	cmpq	%rbp, %r12
	je	.LBB11_357
# %bb.238:                              #   in Loop: Header=BB11_231 Depth=2
	movq	%r12, %rax
	sarq	$2, %rax
	cmpq	$1, %rax
	movq	%rax, %rcx
	adcq	$0, %rcx
	leaq	(%rcx,%rax), %rdx
	movabsq	$2305843009213693951, %rsi      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rsi, %rdx
	jb	.LBB11_240
# %bb.239:                              #   in Loop: Header=BB11_231 Depth=2
	movq	%rsi, %rdx
.LBB11_240:                             #   in Loop: Header=BB11_231 Depth=2
	addq	%rax, %rcx
	jb	.LBB11_242
# %bb.241:                              #   in Loop: Header=BB11_231 Depth=2
	movq	%rdx, %rsi
.LBB11_242:                             #   in Loop: Header=BB11_231 Depth=2
	movq	%rsi, %rbp
	leaq	(,%rsi,4), %rdi
.Ltmp988:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp989:                               # EH_LABEL
# %bb.243:                              #   in Loop: Header=BB11_231 Depth=2
	movq	%rax, %r15
	movss	40(%r14), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r12)
	testq	%r12, %r12
	jle	.LBB11_245
# %bb.244:                              #   in Loop: Header=BB11_231 Depth=2
	movq	%r15, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
.LBB11_245:                             #   in Loop: Header=BB11_231 Depth=2
	testq	%r13, %r13
	je	.LBB11_247
# %bb.246:                              #   in Loop: Header=BB11_231 Depth=2
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB11_247:                             #   in Loop: Header=BB11_231 Depth=2
	addq	%r15, %r12
	addq	$4, %r12
	leaq	(%r15,%rbp,4), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	movabsq	$9223372036854775804, %rbp      # imm = 0x7FFFFFFFFFFFFFFC
	jmp	.LBB11_236
	.p2align	4
.LBB11_248:                             #   in Loop: Header=BB11_226 Depth=1
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	movq	$0, (%rsp)                      # 8-byte Folded Spill
.LBB11_249:                             #   in Loop: Header=BB11_226 Depth=1
	movq	%r12, %rax
	subq	%r15, %rax
	cmpq	$16, %rax
	jne	.LBB11_359
# %bb.250:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp991:                               # EH_LABEL
	movl	$4, %edx
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp992:                               # EH_LABEL
	leaq	720(%rsp), %rbx
# %bb.251:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp993:                               # EH_LABEL
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp994:                               # EH_LABEL
# %bb.252:                              #   in Loop: Header=BB11_226 Depth=1
	movss	8(%r15), %xmm0                  # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp995:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp996:                               # EH_LABEL
# %bb.253:                              #   in Loop: Header=BB11_226 Depth=1
	movq	(%rsp), %rsi                    # 8-byte Reload
	subq	%r15, %rsi
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
	movq	5264(%rsp), %rax
	movq	(%rax), %rbx
	movq	8(%rax), %r14
	cmpq	%r14, %rbx
	je	.LBB11_273
# %bb.254:                              #   in Loop: Header=BB11_226 Depth=1
	movq	$0, (%rsp)                      # 8-byte Folded Spill
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	.p2align	4
.LBB11_257:                             #   Parent Loop BB11_226 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	8(%rsp), %rax                   # 8-byte Reload
	cmpl	%eax, 24(%rbx)
	jne	.LBB11_255
# %bb.258:                              #   in Loop: Header=BB11_257 Depth=2
	cmpq	$1, 32(%rbx)
	jne	.LBB11_255
# %bb.259:                              #   in Loop: Header=BB11_257 Depth=2
	movq	(%rsp), %rax                    # 8-byte Reload
	cmpq	%rax, %r12
	je	.LBB11_262
# %bb.260:                              #   in Loop: Header=BB11_257 Depth=2
	movss	40(%rbx), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%r12)
	addq	$4, %r12
	.p2align	4
.LBB11_255:                             #   in Loop: Header=BB11_257 Depth=2
	movq	%r13, %r15
	addq	$48, %rbx
	movq	%r15, %r13
	cmpq	%r14, %rbx
	jne	.LBB11_257
	jmp	.LBB11_274
.LBB11_262:                             #   in Loop: Header=BB11_257 Depth=2
	movq	%rax, %r12
	subq	%r13, %r12
	cmpq	%rbp, %r12
	je	.LBB11_357
# %bb.263:                              #   in Loop: Header=BB11_257 Depth=2
	movq	%r12, %rax
	sarq	$2, %rax
	cmpq	$1, %rax
	movq	%rax, %rcx
	adcq	$0, %rcx
	leaq	(%rcx,%rax), %rdx
	movabsq	$2305843009213693951, %rbp      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rbp, %rdx
	jb	.LBB11_265
# %bb.264:                              #   in Loop: Header=BB11_257 Depth=2
	movq	%rbp, %rdx
.LBB11_265:                             #   in Loop: Header=BB11_257 Depth=2
	addq	%rax, %rcx
	jb	.LBB11_267
# %bb.266:                              #   in Loop: Header=BB11_257 Depth=2
	movq	%rdx, %rbp
.LBB11_267:                             #   in Loop: Header=BB11_257 Depth=2
	leaq	(,%rbp,4), %rdi
.Ltmp997:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp998:                               # EH_LABEL
# %bb.268:                              #   in Loop: Header=BB11_257 Depth=2
	movq	%rax, %r15
	movss	40(%rbx), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r12)
	testq	%r12, %r12
	jle	.LBB11_270
# %bb.269:                              #   in Loop: Header=BB11_257 Depth=2
	movq	%r15, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
.LBB11_270:                             #   in Loop: Header=BB11_257 Depth=2
	testq	%r13, %r13
	je	.LBB11_272
# %bb.271:                              #   in Loop: Header=BB11_257 Depth=2
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB11_272:                             #   in Loop: Header=BB11_257 Depth=2
	addq	%r15, %r12
	addq	$4, %r12
	leaq	(%r15,%rbp,4), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	movabsq	$9223372036854775804, %rbp      # imm = 0x7FFFFFFFFFFFFFFC
	addq	$48, %rbx
	movq	%r15, %r13
	cmpq	%r14, %rbx
	jne	.LBB11_257
	jmp	.LBB11_274
	.p2align	4
.LBB11_273:                             #   in Loop: Header=BB11_226 Depth=1
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	movq	$0, (%rsp)                      # 8-byte Folded Spill
.LBB11_274:                             #   in Loop: Header=BB11_226 Depth=1
	movq	%r12, %rax
	subq	%r15, %rax
	cmpq	$16, %rax
	jne	.LBB11_359
# %bb.275:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp1000:                              # EH_LABEL
	movl	$2, %edx
	leaq	720(%rsp), %rbx
	movq	%rbx, %rdi
	leaq	.L.str.107(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1001:                              # EH_LABEL
# %bb.276:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp1002:                              # EH_LABEL
	movl	$4, %edx
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp1003:                              # EH_LABEL
# %bb.277:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp1004:                              # EH_LABEL
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp1005:                              # EH_LABEL
# %bb.278:                              #   in Loop: Header=BB11_226 Depth=1
	movss	8(%r15), %xmm0                  # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp1006:                              # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp1007:                              # EH_LABEL
# %bb.279:                              #   in Loop: Header=BB11_226 Depth=1
	movq	(%rsp), %rsi                    # 8-byte Reload
	subq	%r15, %rsi
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
	movq	5264(%rsp), %rax
	movq	(%rax), %rbx
	movq	8(%rax), %r14
	cmpq	%r14, %rbx
	je	.LBB11_299
# %bb.280:                              #   in Loop: Header=BB11_226 Depth=1
	movq	$0, (%rsp)                      # 8-byte Folded Spill
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	.p2align	4
.LBB11_283:                             #   Parent Loop BB11_226 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	8(%rsp), %rax                   # 8-byte Reload
	cmpl	%eax, 24(%rbx)
	jne	.LBB11_281
# %bb.284:                              #   in Loop: Header=BB11_283 Depth=2
	cmpq	$2, 32(%rbx)
	jne	.LBB11_281
# %bb.285:                              #   in Loop: Header=BB11_283 Depth=2
	movq	(%rsp), %rax                    # 8-byte Reload
	cmpq	%rax, %r12
	je	.LBB11_288
# %bb.286:                              #   in Loop: Header=BB11_283 Depth=2
	movss	40(%rbx), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%r12)
	addq	$4, %r12
	.p2align	4
.LBB11_281:                             #   in Loop: Header=BB11_283 Depth=2
	movq	%r13, %r15
	addq	$48, %rbx
	movq	%r15, %r13
	cmpq	%r14, %rbx
	jne	.LBB11_283
	jmp	.LBB11_300
.LBB11_288:                             #   in Loop: Header=BB11_283 Depth=2
	movq	%rax, %r12
	subq	%r13, %r12
	cmpq	%rbp, %r12
	je	.LBB11_357
# %bb.289:                              #   in Loop: Header=BB11_283 Depth=2
	movq	%r12, %rax
	sarq	$2, %rax
	cmpq	$1, %rax
	movq	%rax, %rcx
	adcq	$0, %rcx
	leaq	(%rcx,%rax), %rdx
	movabsq	$2305843009213693951, %rbp      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rbp, %rdx
	jb	.LBB11_291
# %bb.290:                              #   in Loop: Header=BB11_283 Depth=2
	movq	%rbp, %rdx
.LBB11_291:                             #   in Loop: Header=BB11_283 Depth=2
	addq	%rax, %rcx
	jb	.LBB11_293
# %bb.292:                              #   in Loop: Header=BB11_283 Depth=2
	movq	%rdx, %rbp
.LBB11_293:                             #   in Loop: Header=BB11_283 Depth=2
	leaq	(,%rbp,4), %rdi
.Ltmp1008:                              # EH_LABEL
	callq	_Znwm@PLT
.Ltmp1009:                              # EH_LABEL
# %bb.294:                              #   in Loop: Header=BB11_283 Depth=2
	movq	%rax, %r15
	movss	40(%rbx), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r12)
	testq	%r12, %r12
	jle	.LBB11_296
# %bb.295:                              #   in Loop: Header=BB11_283 Depth=2
	movq	%r15, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
.LBB11_296:                             #   in Loop: Header=BB11_283 Depth=2
	testq	%r13, %r13
	je	.LBB11_298
# %bb.297:                              #   in Loop: Header=BB11_283 Depth=2
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB11_298:                             #   in Loop: Header=BB11_283 Depth=2
	addq	%r15, %r12
	addq	$4, %r12
	leaq	(%r15,%rbp,4), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	movabsq	$9223372036854775804, %rbp      # imm = 0x7FFFFFFFFFFFFFFC
	addq	$48, %rbx
	movq	%r15, %r13
	cmpq	%r14, %rbx
	jne	.LBB11_283
	jmp	.LBB11_300
	.p2align	4
.LBB11_299:                             #   in Loop: Header=BB11_226 Depth=1
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	movq	$0, (%rsp)                      # 8-byte Folded Spill
.LBB11_300:                             #   in Loop: Header=BB11_226 Depth=1
	movq	%r12, %rax
	subq	%r15, %rax
	cmpq	$16, %rax
	jne	.LBB11_359
# %bb.301:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp1020:                              # EH_LABEL
	movl	$2, %edx
	leaq	720(%rsp), %r14
	movq	%r14, %rdi
	leaq	.L.str.107(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1021:                              # EH_LABEL
# %bb.302:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp1023:                              # EH_LABEL
	movl	$4, %edx
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp1024:                              # EH_LABEL
# %bb.303:                              #   in Loop: Header=BB11_226 Depth=1
.Ltmp1025:                              # EH_LABEL
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp1026:                              # EH_LABEL
# %bb.304:                              #   in Loop: Header=BB11_226 Depth=1
	movss	8(%r15), %xmm0                  # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp1028:                              # EH_LABEL
	movq	%r14, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp1029:                              # EH_LABEL
# %bb.305:                              #   in Loop: Header=BB11_226 Depth=1
	movq	(%rsp), %rsi                    # 8-byte Reload
	subq	%r15, %rsi
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
.Ltmp1031:                              # EH_LABEL
	movl	$1, %edx
	movq	%r14, %rdi
	leaq	.L.str.108(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1032:                              # EH_LABEL
# %bb.306:                              #   in Loop: Header=BB11_226 Depth=1
	xorl	%edx, %edx
	cmpq	$2, 8(%rsp)                     # 8-byte Folded Reload
	leaq	.L.str.110(%rip), %rsi
	leaq	.L.str.109(%rip), %rax
	cmoveq	%rax, %rsi
	setne	%dl
	incq	%rdx
.Ltmp1033:                              # EH_LABEL
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1034:                              # EH_LABEL
# %bb.307:                              #   in Loop: Header=BB11_226 Depth=1
	movq	8(%rsp), %rcx                   # 8-byte Reload
	incq	%rcx
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	cmpq	$3, %rcx
	jne	.LBB11_226
# %bb.308:
.Ltmp1036:                              # EH_LABEL
	leaq	.L.str.113(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$26, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1037:                              # EH_LABEL
# %bb.309:
	xorl	%r14d, %r14d
	movabsq	$-6148914691236517205, %r12     # imm = 0xAAAAAAAAAAAAAAAB
	leaq	720(%rsp), %rbp
	xorl	%ebx, %ebx
	.p2align	4
.LBB11_310:                             # =>This Inner Loop Header: Depth=1
	movq	5264(%rsp), %rax
	movq	(%rax), %r15
	movq	8(%rax), %rax
	subq	%r15, %rax
	sarq	$4, %rax
	imulq	%r12, %rax
	cmpq	%rax, %rbx
	jae	.LBB11_325
# %bb.311:                              #   in Loop: Header=BB11_310 Depth=1
.Ltmp1088:                              # EH_LABEL
	movl	$14, %edx
	movq	%rbp, %rdi
	leaq	.L.str.114(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1089:                              # EH_LABEL
# %bb.312:                              #   in Loop: Header=BB11_310 Depth=1
	movq	(%r15,%r14), %rsi
.Ltmp1090:                              # EH_LABEL
	movq	%rbp, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp1091:                              # EH_LABEL
# %bb.313:                              #   in Loop: Header=BB11_310 Depth=1
.Ltmp1092:                              # EH_LABEL
	movq	%rax, %r13
	movl	$11, %edx
	movq	%rax, %rdi
	leaq	.L.str.115(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1093:                              # EH_LABEL
# %bb.314:                              #   in Loop: Header=BB11_310 Depth=1
	movq	8(%r15,%r14), %rsi
.Ltmp1094:                              # EH_LABEL
	movq	%r13, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp1095:                              # EH_LABEL
# %bb.315:                              #   in Loop: Header=BB11_310 Depth=1
.Ltmp1096:                              # EH_LABEL
	movq	%rax, %r13
	movl	$14, %edx
	movq	%rax, %rdi
	leaq	.L.str.116(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1097:                              # EH_LABEL
# %bb.316:                              #   in Loop: Header=BB11_310 Depth=1
	movq	16(%r15,%r14), %rsi
.Ltmp1098:                              # EH_LABEL
	movq	%r13, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp1099:                              # EH_LABEL
# %bb.317:                              #   in Loop: Header=BB11_310 Depth=1
.Ltmp1100:                              # EH_LABEL
	movq	%rax, %r13
	movl	$11, %edx
	movq	%rax, %rdi
	leaq	.L.str.117(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1101:                              # EH_LABEL
# %bb.318:                              #   in Loop: Header=BB11_310 Depth=1
	movl	24(%r15,%r14), %esi
.Ltmp1102:                              # EH_LABEL
	movq	%r13, %rdi
	callq	_ZNSolsEi@PLT
.Ltmp1103:                              # EH_LABEL
# %bb.319:                              #   in Loop: Header=BB11_310 Depth=1
.Ltmp1104:                              # EH_LABEL
	movq	%rax, %r13
	movl	$10, %edx
	movq	%rax, %rdi
	leaq	.L.str.118(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1105:                              # EH_LABEL
# %bb.320:                              #   in Loop: Header=BB11_310 Depth=1
	movq	32(%r15,%r14), %rsi
.Ltmp1106:                              # EH_LABEL
	movq	%r13, %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp1107:                              # EH_LABEL
# %bb.321:                              #   in Loop: Header=BB11_310 Depth=1
.Ltmp1108:                              # EH_LABEL
	movq	%rax, %r13
	movl	$18, %edx
	movq	%rax, %rdi
	leaq	.L.str.119(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1109:                              # EH_LABEL
# %bb.322:                              #   in Loop: Header=BB11_310 Depth=1
	movss	40(%r15,%r14), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp1110:                              # EH_LABEL
	movq	%r13, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp1111:                              # EH_LABEL
# %bb.323:                              #   in Loop: Header=BB11_310 Depth=1
.Ltmp1112:                              # EH_LABEL
	movq	%rax, %r13
	movl	$1, %edx
	movq	%rax, %rdi
	leaq	.L.str.120(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1113:                              # EH_LABEL
# %bb.324:                              #   in Loop: Header=BB11_310 Depth=1
	movq	5264(%rsp), %rcx
	movq	8(%rcx), %rax
	subq	(%rcx), %rax
	incq	%rbx
	sarq	$4, %rax
	imulq	%r12, %rax
	xorl	%edx, %edx
	cmpq	%rax, %rbx
	leaq	.L.str.110(%rip), %rsi
	leaq	.L.str.109(%rip), %rax
	cmoveq	%rax, %rsi
	setne	%dl
	incq	%rdx
	addq	$48, %r14
.Ltmp1114:                              # EH_LABEL
	movq	%r13, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1115:                              # EH_LABEL
	jmp	.LBB11_310
.LBB11_325:
.Ltmp1038:                              # EH_LABEL
	leaq	.L.str.121(%rip), %rsi
	leaq	720(%rsp), %rdi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1039:                              # EH_LABEL
# %bb.326:
	movq	680(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rsi
.Ltmp1041:                              # EH_LABEL
	leaq	1096(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_19read_lineB5cxx11EPKc
.Ltmp1042:                              # EH_LABEL
# %bb.327:
	movq	1096(%rsp), %rsi
	movq	1104(%rsp), %rdx
.Ltmp1044:                              # EH_LABEL
	leaq	720(%rsp), %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1045:                              # EH_LABEL
# %bb.328:
.Ltmp1046:                              # EH_LABEL
	leaq	.L.str.122(%rip), %rsi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp1047:                              # EH_LABEL
# %bb.329:
	movq	1096(%rsp), %rdi
	leaq	1112(%rsp), %r14
	cmpq	%r14, %rdi
	je	.LBB11_331
# %bb.330:
	movq	1112(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_331:
	leaq	32(%rsp), %rax
	movq	%rax, 16(%rsp)
	movq	$0, 24(%rsp)
	movb	$0, 32(%rsp)
	movq	752(%rsp), %r8
	movq	768(%rsp), %rax
	cmpq	%r8, %rax
	cmovaq	%rax, %r8
	testq	%rax, %rax
	je	.LBB11_334
# %bb.332:
	testq	%r8, %r8
	je	.LBB11_334
# %bb.333:
	movq	760(%rsp), %rcx
	subq	%rcx, %r8
.Ltmp1049:                              # EH_LABEL
	leaq	16(%rsp), %rdi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm
.Ltmp1050:                              # EH_LABEL
	jmp	.LBB11_335
.LBB11_334:
	leaq	800(%rsp), %rsi
.Ltmp1051:                              # EH_LABEL
	leaq	16(%rsp), %rdi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_
.Ltmp1052:                              # EH_LABEL
.LBB11_335:
	movq	672(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rdi
.Ltmp1054:                              # EH_LABEL
	movl	$193, %esi
	movl	$420, %edx                      # imm = 0x1A4
	xorl	%eax, %eax
	callq	open@PLT
.Ltmp1055:                              # EH_LABEL
# %bb.336:
	movl	%eax, %ebx
	testl	%eax, %eax
	js	.LBB11_372
# %bb.337:
	movq	24(%rsp), %rdx
	testq	%rdx, %rdx
	je	.LBB11_342
# %bb.338:
	xorl	%r15d, %r15d
	.p2align	4
.LBB11_339:                             # =>This Inner Loop Header: Depth=1
	movq	16(%rsp), %rsi
	addq	%r15, %rsi
	subq	%r15, %rdx
.Ltmp1056:                              # EH_LABEL
	movl	%ebx, %edi
	callq	write@PLT
.Ltmp1057:                              # EH_LABEL
# %bb.340:                              #   in Loop: Header=BB11_339 Depth=1
	testq	%rax, %rax
	js	.LBB11_362
# %bb.341:                              #   in Loop: Header=BB11_339 Depth=1
	addq	%rax, %r15
	movq	24(%rsp), %rdx
	cmpq	%rdx, %r15
	jb	.LBB11_339
.LBB11_342:
.Ltmp1070:                              # EH_LABEL
	movl	%ebx, %edi
	callq	close@PLT
.Ltmp1071:                              # EH_LABEL
# %bb.343:
	testl	%eax, %eax
	jne	.LBB11_376
# %bb.344:
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_346
# %bb.345:
	movq	32(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_346:
	movq	_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE@GOTPCREL(%rip), %rax
	movq	(%rax), %rcx
	movq	24(%rax), %rax
	movq	%rcx, 720(%rsp)
	movq	-24(%rcx), %rcx
	movq	%rax, 720(%rsp,%rcx)
	movq	_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE@GOTPCREL(%rip), %rax
	addq	$16, %rax
	movq	%rax, 728(%rsp)
	movq	800(%rsp), %rdi
	leaq	816(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_348
# %bb.347:
	movq	816(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_348:
	movq	_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GOTPCREL(%rip), %rax
	addq	$16, %rax
	movq	%rax, 728(%rsp)
	leaq	784(%rsp), %rdi
	callq	_ZNSt6localeD1Ev@PLT
	leaq	832(%rsp), %rdi
	callq	_ZNSt8ios_baseD2Ev@PLT
	movq	88(%rsp), %rdi
	leaq	104(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_350
# %bb.349:
	movq	104(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_350:
	addq	$5192, %rsp                     # imm = 0x1448
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
.LBB11_351:
	.cfi_def_cfa_offset 5248
	callq	_ZSt28__throw_bad_array_new_lengthv@PLT
.LBB11_352:
	callq	_ZSt17__throw_bad_allocv@PLT
.LBB11_353:
.Ltmp1122:                              # EH_LABEL
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp1123:                              # EH_LABEL
# %bb.354:
.LBB11_355:
.Ltmp1117:                              # EH_LABEL
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp1118:                              # EH_LABEL
# %bb.356:
.LBB11_357:
.Ltmp1011:                              # EH_LABEL
	leaq	.L.str.47(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1012:                              # EH_LABEL
# %bb.358:
.LBB11_359:
	movq	%r15, %r13
.Ltmp1014:                              # EH_LABEL
	leaq	.L.str.112(%rip), %rsi
	leaq	1096(%rsp), %rdi
	leaq	56(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1015:                              # EH_LABEL
# %bb.360:
.Ltmp1017:                              # EH_LABEL
	leaq	1096(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1018:                              # EH_LABEL
# %bb.361:
.LBB11_362:
	callq	__errno_location@PLT
	movl	(%rax), %ebp
.Ltmp1059:                              # EH_LABEL
	movl	%ebx, %edi
	callq	close@PLT
.Ltmp1060:                              # EH_LABEL
# %bb.363:
	movl	%ebp, %edi
	callq	strerror@PLT
.Ltmp1061:                              # EH_LABEL
	leaq	56(%rsp), %rdi
	leaq	128(%rsp), %rdx
	movq	%rax, %rsi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1062:                              # EH_LABEL
# %bb.364:
.Ltmp1064:                              # EH_LABEL
	leaq	.L.str.137(%rip), %rsi
	leaq	1096(%rsp), %rdi
	leaq	56(%rsp), %rdx
	callq	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_
.Ltmp1065:                              # EH_LABEL
# %bb.365:
.Ltmp1067:                              # EH_LABEL
	leaq	1096(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1068:                              # EH_LABEL
# %bb.366:
.LBB11_367:
	leaq	.L.str.124(%rip), %rsi
	leaq	720(%rsp), %rbx
	leaq	56(%rsp), %rdx
	movq	%rbx, %rdi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1127:                              # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1128:                              # EH_LABEL
# %bb.368:
.LBB11_369:
.Ltmp820:                               # EH_LABEL
	leaq	.L.str.135(%rip), %rsi
	leaq	56(%rsp), %rdi
	movq	%r14, %rdx
	callq	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_
.Ltmp821:                               # EH_LABEL
# %bb.370:
.Ltmp823:                               # EH_LABEL
	leaq	56(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp824:                               # EH_LABEL
# %bb.371:
.LBB11_372:
	callq	__errno_location@PLT
	movl	(%rax), %edi
	callq	strerror@PLT
.Ltmp1079:                              # EH_LABEL
	leaq	56(%rsp), %rdi
	leaq	128(%rsp), %rdx
	movq	%rax, %rsi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1080:                              # EH_LABEL
# %bb.373:
.Ltmp1082:                              # EH_LABEL
	leaq	.L.str.136(%rip), %rsi
	leaq	1096(%rsp), %rdi
	leaq	56(%rsp), %rdx
	callq	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_
.Ltmp1083:                              # EH_LABEL
# %bb.374:
.Ltmp1085:                              # EH_LABEL
	leaq	1096(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1086:                              # EH_LABEL
# %bb.375:
.LBB11_376:
.Ltmp1073:                              # EH_LABEL
	leaq	.L.str.138(%rip), %rsi
	leaq	1096(%rsp), %rdi
	leaq	56(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1074:                              # EH_LABEL
# %bb.377:
.Ltmp1076:                              # EH_LABEL
	leaq	1096(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1077:                              # EH_LABEL
# %bb.378:
.LBB11_379:
.Ltmp1124:                              # EH_LABEL
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1125:                              # EH_LABEL
# %bb.380:
.LBB11_381:
.Ltmp1119:                              # EH_LABEL
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1120:                              # EH_LABEL
# %bb.382:
.LBB11_383:
	cmpq	$4, %r13
	jne	.LBB11_5
# %bb.384:
	movss	(%r12), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rbp)
	jmp	.LBB11_5
.LBB11_385:
	cmpq	$4, %r13
	jne	.LBB11_13
# %bb.386:
	movss	(%r12), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rbp)
	jmp	.LBB11_13
.LBB11_387:
	cmpq	$4, %r13
	jne	.LBB11_21
# %bb.388:
	movss	(%r12), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rbp)
	jmp	.LBB11_21
.LBB11_389:
.Ltmp1087:                              # EH_LABEL
	jmp	.LBB11_450
.LBB11_390:
.Ltmp1084:                              # EH_LABEL
	jmp	.LBB11_453
.LBB11_391:
.Ltmp1078:                              # EH_LABEL
	movq	%rax, %rbx
	movq	1096(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.LBB11_467
# %bb.392:
	movq	1112(%rsp), %rsi
	jmp	.LBB11_456
.LBB11_393:
.Ltmp1075:                              # EH_LABEL
	jmp	.LBB11_466
.LBB11_394:
.Ltmp1081:                              # EH_LABEL
	jmp	.LBB11_466
.LBB11_395:
.Ltmp825:                               # EH_LABEL
	movq	%rax, %rbx
	movq	56(%rsp), %rdi
	leaq	72(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_403
# %bb.396:
	movq	72(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB11_403
.LBB11_397:
.Ltmp822:                               # EH_LABEL
	jmp	.LBB11_402
.LBB11_398:
.Ltmp1053:                              # EH_LABEL
	jmp	.LBB11_466
.LBB11_399:
.Ltmp1043:                              # EH_LABEL
	jmp	.LBB11_529
.LBB11_400:
.Ltmp831:                               # EH_LABEL
	movq	%rax, %rbx
	movq	192(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.LBB11_475
	jmp	.LBB11_496
.LBB11_401:
.Ltmp828:                               # EH_LABEL
.LBB11_402:
	movq	%rax, %rbx
.LBB11_403:
	leaq	1096(%rsp), %rdi
	callq	_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@PLT
	movq	256(%rsp), %rdi
	leaq	272(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_476
	jmp	.LBB11_497
.LBB11_404:
.Ltmp819:                               # EH_LABEL
	jmp	.LBB11_432
.LBB11_405:
.Ltmp811:                               # EH_LABEL
	jmp	.LBB11_434
.LBB11_406:
.Ltmp803:                               # EH_LABEL
	jmp	.LBB11_436
.LBB11_407:
.Ltmp795:                               # EH_LABEL
	jmp	.LBB11_438
.LBB11_408:
.Ltmp787:                               # EH_LABEL
	jmp	.LBB11_440
.LBB11_409:
.Ltmp779:                               # EH_LABEL
	movq	%rax, %rbx
	movq	384(%rsp), %rdi
	cmpq	%rbp, %rdi
	je	.LBB11_481
	jmp	.LBB11_502
.LBB11_410:
.Ltmp776:                               # EH_LABEL
	jmp	.LBB11_442
.LBB11_411:
.Ltmp768:                               # EH_LABEL
	movq	%rax, %rbx
	movq	448(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.LBB11_483
	jmp	.LBB11_504
.LBB11_412:
.Ltmp765:                               # EH_LABEL
	jmp	.LBB11_444
.LBB11_413:
.Ltmp757:                               # EH_LABEL
	jmp	.LBB11_446
.LBB11_414:
.Ltmp749:                               # EH_LABEL
	jmp	.LBB11_459
.LBB11_415:
.Ltmp739:                               # EH_LABEL
	movq	%rax, %rbx
	movq	576(%rsp), %rdi
	leaq	592(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_487
	jmp	.LBB11_488
.LBB11_416:
.Ltmp736:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_488
.LBB11_417:
.Ltmp733:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_489
.LBB11_418:
.Ltmp709:                               # EH_LABEL
	movq	%rax, %rbx
	movq	160(%rsp), %rdi
	leaq	176(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_491
	jmp	.LBB11_494
.LBB11_419:
.Ltmp699:                               # EH_LABEL
	movq	%rax, %rbx
	movq	128(%rsp), %rdi
	leaq	144(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_524
	jmp	.LBB11_530
.LBB11_420:
.Ltmp683:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_531
.LBB11_421:
.Ltmp1129:                              # EH_LABEL
	movq	%rax, %rbx
	movq	720(%rsp), %rdi
	leaq	736(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_535
# %bb.422:
	movq	736(%rsp), %rsi
	jmp	.LBB11_533
.LBB11_423:
.Ltmp680:                               # EH_LABEL
	jmp	.LBB11_426
.LBB11_424:
.Ltmp675:                               # EH_LABEL
	jmp	.LBB11_426
.LBB11_425:
.Ltmp670:                               # EH_LABEL
.LBB11_426:
	movq	%rax, %rbx
	testq	%rbp, %rbp
	je	.LBB11_535
# %bb.427:
	movq	%rbp, %rdi
	movq	%r13, %rsi
	jmp	.LBB11_534
.LBB11_428:
.Ltmp969:                               # EH_LABEL
	jmp	.LBB11_529
.LBB11_429:
.Ltmp1048:                              # EH_LABEL
	movq	%rax, %rbx
	movq	1096(%rsp), %rdi
	leaq	1112(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_530
# %bb.430:
	movq	1112(%rsp), %rsi
	jmp	.LBB11_525
.LBB11_431:
.Ltmp816:                               # EH_LABEL
.LBB11_432:
	movq	%rax, %rbx
	movq	256(%rsp), %rdi
	leaq	272(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_476
	jmp	.LBB11_497
.LBB11_433:
.Ltmp808:                               # EH_LABEL
.LBB11_434:
	movq	%rax, %rbx
	movq	288(%rsp), %rdi
	leaq	304(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_477
	jmp	.LBB11_498
.LBB11_435:
.Ltmp800:                               # EH_LABEL
.LBB11_436:
	movq	%rax, %rbx
	movq	320(%rsp), %rdi
	leaq	336(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_478
	jmp	.LBB11_499
.LBB11_437:
.Ltmp792:                               # EH_LABEL
.LBB11_438:
	movq	%rax, %rbx
	movq	352(%rsp), %rdi
	leaq	368(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_479
	jmp	.LBB11_500
.LBB11_439:
.Ltmp784:                               # EH_LABEL
.LBB11_440:
	movq	%rax, %rbx
	movq	416(%rsp), %rdi
	leaq	432(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_480
	jmp	.LBB11_501
.LBB11_441:
.Ltmp773:                               # EH_LABEL
.LBB11_442:
	movq	%rax, %rbx
	movq	480(%rsp), %rdi
	leaq	496(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_482
	jmp	.LBB11_503
.LBB11_443:
.Ltmp762:                               # EH_LABEL
.LBB11_444:
	movq	%rax, %rbx
	movq	512(%rsp), %rdi
	leaq	528(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_484
	jmp	.LBB11_505
.LBB11_445:
.Ltmp754:                               # EH_LABEL
.LBB11_446:
	movq	%rax, %rbx
	movq	544(%rsp), %rdi
	leaq	560(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_485
	jmp	.LBB11_506
.LBB11_447:
.Ltmp704:                               # EH_LABEL
	jmp	.LBB11_448
.LBB11_449:
.Ltmp1069:                              # EH_LABEL
.LBB11_450:
	movq	%rax, %rbx
	movq	1096(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.LBB11_454
# %bb.451:
	movq	1112(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB11_454
.LBB11_452:
.Ltmp1066:                              # EH_LABEL
.LBB11_453:
	movq	%rax, %rbx
.LBB11_454:
	movq	56(%rsp), %rdi
	leaq	72(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_467
# %bb.455:
	movq	72(%rsp), %rsi
.LBB11_456:
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB11_467
.LBB11_457:
.Ltmp1040:                              # EH_LABEL
	jmp	.LBB11_529
.LBB11_458:
.Ltmp746:                               # EH_LABEL
.LBB11_459:
	movq	%rax, %rbx
	movq	608(%rsp), %rdi
	leaq	624(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_486
	jmp	.LBB11_507
.LBB11_460:
.Ltmp1063:                              # EH_LABEL
	jmp	.LBB11_466
.LBB11_461:
.Ltmp694:                               # EH_LABEL
	jmp	.LBB11_529
.LBB11_462:
.Ltmp730:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_489
.LBB11_463:
.Ltmp1072:                              # EH_LABEL
	jmp	.LBB11_466
.LBB11_464:
.Ltmp979:                               # EH_LABEL
	jmp	.LBB11_529
.LBB11_465:
.Ltmp1058:                              # EH_LABEL
.LBB11_466:
	movq	%rax, %rbx
.LBB11_467:
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_530
# %bb.468:
	movq	32(%rsp), %rsi
	jmp	.LBB11_525
.LBB11_469:
.Ltmp974:                               # EH_LABEL
	jmp	.LBB11_529
.LBB11_470:
.Ltmp956:                               # EH_LABEL
	jmp	.LBB11_529
.LBB11_471:
.Ltmp939:                               # EH_LABEL
	jmp	.LBB11_529
.LBB11_472:
.Ltmp990:                               # EH_LABEL
	jmp	.LBB11_517
.LBB11_473:
.Ltmp926:                               # EH_LABEL
	movq	%rax, %rbx
	movq	224(%rsp), %rdi
	leaq	240(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_495
# %bb.474:
	movq	192(%rsp), %rdi
	cmpq	%r14, %rdi
	jne	.LBB11_496
.LBB11_475:
	movq	256(%rsp), %rdi
	leaq	272(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_497
.LBB11_476:
	movq	288(%rsp), %rdi
	leaq	304(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_498
.LBB11_477:
	movq	320(%rsp), %rdi
	leaq	336(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_499
.LBB11_478:
	movq	352(%rsp), %rdi
	leaq	368(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_500
.LBB11_479:
	movq	416(%rsp), %rdi
	leaq	432(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_501
.LBB11_480:
	movq	384(%rsp), %rdi
	cmpq	%rbp, %rdi
	jne	.LBB11_502
.LBB11_481:
	movq	480(%rsp), %rdi
	leaq	496(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_503
.LBB11_482:
	movq	448(%rsp), %rdi
	cmpq	%r13, %rdi
	jne	.LBB11_504
.LBB11_483:
	movq	512(%rsp), %rdi
	leaq	528(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_505
.LBB11_484:
	movq	544(%rsp), %rdi
	leaq	560(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_506
.LBB11_485:
	movq	608(%rsp), %rdi
	leaq	624(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_507
.LBB11_486:
	movq	576(%rsp), %rdi
	leaq	592(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_488
.LBB11_487:
	movq	592(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB11_488:
	movl	$35, %esi
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
.LBB11_489:
	movq	640(%rsp), %rdi
	leaq	656(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_493
# %bb.490:
	movq	160(%rsp), %rdi
	leaq	176(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_494
.LBB11_491:
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_523
.LBB11_492:
	movq	128(%rsp), %rdi
	leaq	144(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_530
	jmp	.LBB11_524
.LBB11_493:
	movq	656(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	160(%rsp), %rdi
	leaq	176(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_491
.LBB11_494:
	movq	176(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_492
	jmp	.LBB11_523
.LBB11_495:
	movq	240(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	192(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.LBB11_475
.LBB11_496:
	movq	208(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	256(%rsp), %rdi
	leaq	272(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_476
.LBB11_497:
	movq	272(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	288(%rsp), %rdi
	leaq	304(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_477
.LBB11_498:
	movq	304(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	320(%rsp), %rdi
	leaq	336(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_478
.LBB11_499:
	movq	336(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	352(%rsp), %rdi
	leaq	368(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_479
.LBB11_500:
	movq	368(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	416(%rsp), %rdi
	leaq	432(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_480
.LBB11_501:
	movq	432(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	384(%rsp), %rdi
	cmpq	%rbp, %rdi
	je	.LBB11_481
.LBB11_502:
	movq	400(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	480(%rsp), %rdi
	leaq	496(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_482
.LBB11_503:
	movq	496(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	448(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.LBB11_483
.LBB11_504:
	movq	464(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	512(%rsp), %rdi
	leaq	528(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_484
.LBB11_505:
	movq	528(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	544(%rsp), %rdi
	leaq	560(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_485
.LBB11_506:
	movq	560(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	608(%rsp), %rdi
	leaq	624(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_486
.LBB11_507:
	movq	624(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	576(%rsp), %rdi
	leaq	592(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_487
	jmp	.LBB11_488
.LBB11_508:
.Ltmp1019:                              # EH_LABEL
	movq	%rax, %rbx
	movq	1096(%rsp), %rdi
	leaq	1112(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_518
# %bb.509:
	movq	1112(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB11_518
.LBB11_510:
.Ltmp1016:                              # EH_LABEL
	jmp	.LBB11_517
.LBB11_511:
.Ltmp1010:                              # EH_LABEL
	jmp	.LBB11_517
.LBB11_512:
.Ltmp999:                               # EH_LABEL
	jmp	.LBB11_517
.LBB11_513:
.Ltmp1022:                              # EH_LABEL
	movq	%rax, %rbx
	movq	%r15, %r13
	jmp	.LBB11_518
.LBB11_514:
.Ltmp1030:                              # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_521
.LBB11_515:
.Ltmp1035:                              # EH_LABEL
	jmp	.LBB11_529
.LBB11_516:
.Ltmp1013:                              # EH_LABEL
.LBB11_517:
	movq	%rax, %rbx
.LBB11_518:
	testq	%r13, %r13
	je	.LBB11_530
# %bb.519:
	movq	(%rsp), %rsi                    # 8-byte Reload
	subq	%r13, %rsi
	movq	%r13, %rdi
	jmp	.LBB11_526
.LBB11_520:
.Ltmp1027:                              # EH_LABEL
	movq	%rax, %rbx
	testq	%r15, %r15
	je	.LBB11_530
.LBB11_521:
	movq	(%rsp), %rsi                    # 8-byte Reload
	subq	%r15, %rsi
	movq	%r15, %rdi
	jmp	.LBB11_526
.LBB11_522:
.Ltmp1121:                              # EH_LABEL
.LBB11_448:
	movq	%rax, %rbx
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_492
.LBB11_523:
	movq	32(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	movq	128(%rsp), %rdi
	leaq	144(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_530
.LBB11_524:
	movq	144(%rsp), %rsi
.LBB11_525:
	incq	%rsi
.LBB11_526:
	callq	_ZdlPvm@PLT
	jmp	.LBB11_530
.LBB11_527:
.Ltmp1126:                              # EH_LABEL
	jmp	.LBB11_529
.LBB11_528:
.Ltmp1116:                              # EH_LABEL
.LBB11_529:
	movq	%rax, %rbx
.LBB11_530:
	leaq	720(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@PLT
.LBB11_531:
	movq	88(%rsp), %rdi
	leaq	104(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB11_535
# %bb.532:
	movq	104(%rsp), %rsi
.LBB11_533:
	incq	%rsi
.LBB11_534:
	callq	_ZdlPvm@PLT
.LBB11_535:
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end11:
	.size	_ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE, .Lfunc_end11-_ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table11:
.Lexception5:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end5-.Lcst_begin5
.Lcst_begin5:
	.uleb128 .Lfunc_begin5-.Lfunc_begin5    # >> Call Site 1 <<
	.uleb128 .Ltmp666-.Lfunc_begin5         #   Call between .Lfunc_begin5 and .Ltmp666
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp666-.Lfunc_begin5         # >> Call Site 2 <<
	.uleb128 .Ltmp669-.Ltmp666              #   Call between .Ltmp666 and .Ltmp669
	.uleb128 .Ltmp670-.Lfunc_begin5         #     jumps to .Ltmp670
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp669-.Lfunc_begin5         # >> Call Site 3 <<
	.uleb128 .Ltmp671-.Ltmp669              #   Call between .Ltmp669 and .Ltmp671
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp671-.Lfunc_begin5         # >> Call Site 4 <<
	.uleb128 .Ltmp674-.Ltmp671              #   Call between .Ltmp671 and .Ltmp674
	.uleb128 .Ltmp675-.Lfunc_begin5         #     jumps to .Ltmp675
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp674-.Lfunc_begin5         # >> Call Site 5 <<
	.uleb128 .Ltmp676-.Ltmp674              #   Call between .Ltmp674 and .Ltmp676
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp676-.Lfunc_begin5         # >> Call Site 6 <<
	.uleb128 .Ltmp679-.Ltmp676              #   Call between .Ltmp676 and .Ltmp679
	.uleb128 .Ltmp680-.Lfunc_begin5         #     jumps to .Ltmp680
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp679-.Lfunc_begin5         # >> Call Site 7 <<
	.uleb128 .Ltmp681-.Ltmp679              #   Call between .Ltmp679 and .Ltmp681
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp681-.Lfunc_begin5         # >> Call Site 8 <<
	.uleb128 .Ltmp682-.Ltmp681              #   Call between .Ltmp681 and .Ltmp682
	.uleb128 .Ltmp683-.Lfunc_begin5         #     jumps to .Ltmp683
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp684-.Lfunc_begin5         # >> Call Site 9 <<
	.uleb128 .Ltmp693-.Ltmp684              #   Call between .Ltmp684 and .Ltmp693
	.uleb128 .Ltmp694-.Lfunc_begin5         #     jumps to .Ltmp694
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp695-.Lfunc_begin5         # >> Call Site 10 <<
	.uleb128 .Ltmp696-.Ltmp695              #   Call between .Ltmp695 and .Ltmp696
	.uleb128 .Ltmp1126-.Lfunc_begin5        #     jumps to .Ltmp1126
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp696-.Lfunc_begin5         # >> Call Site 11 <<
	.uleb128 .Ltmp697-.Ltmp696              #   Call between .Ltmp696 and .Ltmp697
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp697-.Lfunc_begin5         # >> Call Site 12 <<
	.uleb128 .Ltmp698-.Ltmp697              #   Call between .Ltmp697 and .Ltmp698
	.uleb128 .Ltmp699-.Lfunc_begin5         #     jumps to .Ltmp699
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp700-.Lfunc_begin5         # >> Call Site 13 <<
	.uleb128 .Ltmp703-.Ltmp700              #   Call between .Ltmp700 and .Ltmp703
	.uleb128 .Ltmp704-.Lfunc_begin5         #     jumps to .Ltmp704
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp705-.Lfunc_begin5         # >> Call Site 14 <<
	.uleb128 .Ltmp706-.Ltmp705              #   Call between .Ltmp705 and .Ltmp706
	.uleb128 .Ltmp1121-.Lfunc_begin5        #     jumps to .Ltmp1121
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp706-.Lfunc_begin5         # >> Call Site 15 <<
	.uleb128 .Ltmp707-.Ltmp706              #   Call between .Ltmp706 and .Ltmp707
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp707-.Lfunc_begin5         # >> Call Site 16 <<
	.uleb128 .Ltmp708-.Ltmp707              #   Call between .Ltmp707 and .Ltmp708
	.uleb128 .Ltmp709-.Lfunc_begin5         #     jumps to .Ltmp709
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp710-.Lfunc_begin5         # >> Call Site 17 <<
	.uleb128 .Ltmp729-.Ltmp710              #   Call between .Ltmp710 and .Ltmp729
	.uleb128 .Ltmp730-.Lfunc_begin5         #     jumps to .Ltmp730
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp731-.Lfunc_begin5         # >> Call Site 18 <<
	.uleb128 .Ltmp732-.Ltmp731              #   Call between .Ltmp731 and .Ltmp732
	.uleb128 .Ltmp733-.Lfunc_begin5         #     jumps to .Ltmp733
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp734-.Lfunc_begin5         # >> Call Site 19 <<
	.uleb128 .Ltmp735-.Ltmp734              #   Call between .Ltmp734 and .Ltmp735
	.uleb128 .Ltmp736-.Lfunc_begin5         #     jumps to .Ltmp736
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp737-.Lfunc_begin5         # >> Call Site 20 <<
	.uleb128 .Ltmp738-.Ltmp737              #   Call between .Ltmp737 and .Ltmp738
	.uleb128 .Ltmp739-.Lfunc_begin5         #     jumps to .Ltmp739
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp740-.Lfunc_begin5         # >> Call Site 21 <<
	.uleb128 .Ltmp745-.Ltmp740              #   Call between .Ltmp740 and .Ltmp745
	.uleb128 .Ltmp746-.Lfunc_begin5         #     jumps to .Ltmp746
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp747-.Lfunc_begin5         # >> Call Site 22 <<
	.uleb128 .Ltmp748-.Ltmp747              #   Call between .Ltmp747 and .Ltmp748
	.uleb128 .Ltmp749-.Lfunc_begin5         #     jumps to .Ltmp749
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp750-.Lfunc_begin5         # >> Call Site 23 <<
	.uleb128 .Ltmp753-.Ltmp750              #   Call between .Ltmp750 and .Ltmp753
	.uleb128 .Ltmp754-.Lfunc_begin5         #     jumps to .Ltmp754
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp755-.Lfunc_begin5         # >> Call Site 24 <<
	.uleb128 .Ltmp756-.Ltmp755              #   Call between .Ltmp755 and .Ltmp756
	.uleb128 .Ltmp757-.Lfunc_begin5         #     jumps to .Ltmp757
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp758-.Lfunc_begin5         # >> Call Site 25 <<
	.uleb128 .Ltmp761-.Ltmp758              #   Call between .Ltmp758 and .Ltmp761
	.uleb128 .Ltmp762-.Lfunc_begin5         #     jumps to .Ltmp762
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp763-.Lfunc_begin5         # >> Call Site 26 <<
	.uleb128 .Ltmp764-.Ltmp763              #   Call between .Ltmp763 and .Ltmp764
	.uleb128 .Ltmp765-.Lfunc_begin5         #     jumps to .Ltmp765
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp766-.Lfunc_begin5         # >> Call Site 27 <<
	.uleb128 .Ltmp767-.Ltmp766              #   Call between .Ltmp766 and .Ltmp767
	.uleb128 .Ltmp768-.Lfunc_begin5         #     jumps to .Ltmp768
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp769-.Lfunc_begin5         # >> Call Site 28 <<
	.uleb128 .Ltmp772-.Ltmp769              #   Call between .Ltmp769 and .Ltmp772
	.uleb128 .Ltmp773-.Lfunc_begin5         #     jumps to .Ltmp773
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp774-.Lfunc_begin5         # >> Call Site 29 <<
	.uleb128 .Ltmp775-.Ltmp774              #   Call between .Ltmp774 and .Ltmp775
	.uleb128 .Ltmp776-.Lfunc_begin5         #     jumps to .Ltmp776
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp777-.Lfunc_begin5         # >> Call Site 30 <<
	.uleb128 .Ltmp778-.Ltmp777              #   Call between .Ltmp777 and .Ltmp778
	.uleb128 .Ltmp779-.Lfunc_begin5         #     jumps to .Ltmp779
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp780-.Lfunc_begin5         # >> Call Site 31 <<
	.uleb128 .Ltmp783-.Ltmp780              #   Call between .Ltmp780 and .Ltmp783
	.uleb128 .Ltmp784-.Lfunc_begin5         #     jumps to .Ltmp784
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp785-.Lfunc_begin5         # >> Call Site 32 <<
	.uleb128 .Ltmp786-.Ltmp785              #   Call between .Ltmp785 and .Ltmp786
	.uleb128 .Ltmp787-.Lfunc_begin5         #     jumps to .Ltmp787
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp788-.Lfunc_begin5         # >> Call Site 33 <<
	.uleb128 .Ltmp791-.Ltmp788              #   Call between .Ltmp788 and .Ltmp791
	.uleb128 .Ltmp792-.Lfunc_begin5         #     jumps to .Ltmp792
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp793-.Lfunc_begin5         # >> Call Site 34 <<
	.uleb128 .Ltmp794-.Ltmp793              #   Call between .Ltmp793 and .Ltmp794
	.uleb128 .Ltmp795-.Lfunc_begin5         #     jumps to .Ltmp795
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp796-.Lfunc_begin5         # >> Call Site 35 <<
	.uleb128 .Ltmp799-.Ltmp796              #   Call between .Ltmp796 and .Ltmp799
	.uleb128 .Ltmp800-.Lfunc_begin5         #     jumps to .Ltmp800
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp801-.Lfunc_begin5         # >> Call Site 36 <<
	.uleb128 .Ltmp802-.Ltmp801              #   Call between .Ltmp801 and .Ltmp802
	.uleb128 .Ltmp803-.Lfunc_begin5         #     jumps to .Ltmp803
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp804-.Lfunc_begin5         # >> Call Site 37 <<
	.uleb128 .Ltmp807-.Ltmp804              #   Call between .Ltmp804 and .Ltmp807
	.uleb128 .Ltmp808-.Lfunc_begin5         #     jumps to .Ltmp808
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp809-.Lfunc_begin5         # >> Call Site 38 <<
	.uleb128 .Ltmp810-.Ltmp809              #   Call between .Ltmp809 and .Ltmp810
	.uleb128 .Ltmp811-.Lfunc_begin5         #     jumps to .Ltmp811
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp812-.Lfunc_begin5         # >> Call Site 39 <<
	.uleb128 .Ltmp815-.Ltmp812              #   Call between .Ltmp812 and .Ltmp815
	.uleb128 .Ltmp816-.Lfunc_begin5         #     jumps to .Ltmp816
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp817-.Lfunc_begin5         # >> Call Site 40 <<
	.uleb128 .Ltmp818-.Ltmp817              #   Call between .Ltmp817 and .Ltmp818
	.uleb128 .Ltmp819-.Lfunc_begin5         #     jumps to .Ltmp819
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp826-.Lfunc_begin5         # >> Call Site 41 <<
	.uleb128 .Ltmp827-.Ltmp826              #   Call between .Ltmp826 and .Ltmp827
	.uleb128 .Ltmp828-.Lfunc_begin5         #     jumps to .Ltmp828
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp829-.Lfunc_begin5         # >> Call Site 42 <<
	.uleb128 .Ltmp830-.Ltmp829              #   Call between .Ltmp829 and .Ltmp830
	.uleb128 .Ltmp831-.Lfunc_begin5         #     jumps to .Ltmp831
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp832-.Lfunc_begin5         # >> Call Site 43 <<
	.uleb128 .Ltmp925-.Ltmp832              #   Call between .Ltmp832 and .Ltmp925
	.uleb128 .Ltmp926-.Lfunc_begin5         #     jumps to .Ltmp926
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp927-.Lfunc_begin5         # >> Call Site 44 <<
	.uleb128 .Ltmp932-.Ltmp927              #   Call between .Ltmp927 and .Ltmp932
	.uleb128 .Ltmp979-.Lfunc_begin5         #     jumps to .Ltmp979
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp933-.Lfunc_begin5         # >> Call Site 45 <<
	.uleb128 .Ltmp934-.Ltmp933              #   Call between .Ltmp933 and .Ltmp934
	.uleb128 .Ltmp969-.Lfunc_begin5         #     jumps to .Ltmp969
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp935-.Lfunc_begin5         # >> Call Site 46 <<
	.uleb128 .Ltmp938-.Ltmp935              #   Call between .Ltmp935 and .Ltmp938
	.uleb128 .Ltmp939-.Lfunc_begin5         #     jumps to .Ltmp939
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp940-.Lfunc_begin5         # >> Call Site 47 <<
	.uleb128 .Ltmp949-.Ltmp940              #   Call between .Ltmp940 and .Ltmp949
	.uleb128 .Ltmp979-.Lfunc_begin5         #     jumps to .Ltmp979
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp950-.Lfunc_begin5         # >> Call Site 48 <<
	.uleb128 .Ltmp951-.Ltmp950              #   Call between .Ltmp950 and .Ltmp951
	.uleb128 .Ltmp969-.Lfunc_begin5         #     jumps to .Ltmp969
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp952-.Lfunc_begin5         # >> Call Site 49 <<
	.uleb128 .Ltmp955-.Ltmp952              #   Call between .Ltmp952 and .Ltmp955
	.uleb128 .Ltmp956-.Lfunc_begin5         #     jumps to .Ltmp956
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp957-.Lfunc_begin5         # >> Call Site 50 <<
	.uleb128 .Ltmp966-.Ltmp957              #   Call between .Ltmp957 and .Ltmp966
	.uleb128 .Ltmp979-.Lfunc_begin5         #     jumps to .Ltmp979
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp967-.Lfunc_begin5         # >> Call Site 51 <<
	.uleb128 .Ltmp968-.Ltmp967              #   Call between .Ltmp967 and .Ltmp968
	.uleb128 .Ltmp969-.Lfunc_begin5         #     jumps to .Ltmp969
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp970-.Lfunc_begin5         # >> Call Site 52 <<
	.uleb128 .Ltmp973-.Ltmp970              #   Call between .Ltmp970 and .Ltmp973
	.uleb128 .Ltmp974-.Lfunc_begin5         #     jumps to .Ltmp974
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp975-.Lfunc_begin5         # >> Call Site 53 <<
	.uleb128 .Ltmp978-.Ltmp975              #   Call between .Ltmp975 and .Ltmp978
	.uleb128 .Ltmp979-.Lfunc_begin5         #     jumps to .Ltmp979
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp980-.Lfunc_begin5         # >> Call Site 54 <<
	.uleb128 .Ltmp981-.Ltmp980              #   Call between .Ltmp980 and .Ltmp981
	.uleb128 .Ltmp1040-.Lfunc_begin5        #     jumps to .Ltmp1040
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp982-.Lfunc_begin5         # >> Call Site 55 <<
	.uleb128 .Ltmp987-.Ltmp982              #   Call between .Ltmp982 and .Ltmp987
	.uleb128 .Ltmp1035-.Lfunc_begin5        #     jumps to .Ltmp1035
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp988-.Lfunc_begin5         # >> Call Site 56 <<
	.uleb128 .Ltmp989-.Ltmp988              #   Call between .Ltmp988 and .Ltmp989
	.uleb128 .Ltmp990-.Lfunc_begin5         #     jumps to .Ltmp990
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp989-.Lfunc_begin5         # >> Call Site 57 <<
	.uleb128 .Ltmp991-.Ltmp989              #   Call between .Ltmp989 and .Ltmp991
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp991-.Lfunc_begin5         # >> Call Site 58 <<
	.uleb128 .Ltmp994-.Ltmp991              #   Call between .Ltmp991 and .Ltmp994
	.uleb128 .Ltmp1027-.Lfunc_begin5        #     jumps to .Ltmp1027
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp995-.Lfunc_begin5         # >> Call Site 59 <<
	.uleb128 .Ltmp996-.Ltmp995              #   Call between .Ltmp995 and .Ltmp996
	.uleb128 .Ltmp1030-.Lfunc_begin5        #     jumps to .Ltmp1030
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp997-.Lfunc_begin5         # >> Call Site 60 <<
	.uleb128 .Ltmp998-.Ltmp997              #   Call between .Ltmp997 and .Ltmp998
	.uleb128 .Ltmp999-.Lfunc_begin5         #     jumps to .Ltmp999
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp998-.Lfunc_begin5         # >> Call Site 61 <<
	.uleb128 .Ltmp1000-.Ltmp998             #   Call between .Ltmp998 and .Ltmp1000
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1000-.Lfunc_begin5        # >> Call Site 62 <<
	.uleb128 .Ltmp1001-.Ltmp1000            #   Call between .Ltmp1000 and .Ltmp1001
	.uleb128 .Ltmp1022-.Lfunc_begin5        #     jumps to .Ltmp1022
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1002-.Lfunc_begin5        # >> Call Site 63 <<
	.uleb128 .Ltmp1005-.Ltmp1002            #   Call between .Ltmp1002 and .Ltmp1005
	.uleb128 .Ltmp1027-.Lfunc_begin5        #     jumps to .Ltmp1027
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1006-.Lfunc_begin5        # >> Call Site 64 <<
	.uleb128 .Ltmp1007-.Ltmp1006            #   Call between .Ltmp1006 and .Ltmp1007
	.uleb128 .Ltmp1030-.Lfunc_begin5        #     jumps to .Ltmp1030
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1008-.Lfunc_begin5        # >> Call Site 65 <<
	.uleb128 .Ltmp1009-.Ltmp1008            #   Call between .Ltmp1008 and .Ltmp1009
	.uleb128 .Ltmp1010-.Lfunc_begin5        #     jumps to .Ltmp1010
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1009-.Lfunc_begin5        # >> Call Site 66 <<
	.uleb128 .Ltmp1020-.Ltmp1009            #   Call between .Ltmp1009 and .Ltmp1020
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1020-.Lfunc_begin5        # >> Call Site 67 <<
	.uleb128 .Ltmp1021-.Ltmp1020            #   Call between .Ltmp1020 and .Ltmp1021
	.uleb128 .Ltmp1022-.Lfunc_begin5        #     jumps to .Ltmp1022
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1023-.Lfunc_begin5        # >> Call Site 68 <<
	.uleb128 .Ltmp1026-.Ltmp1023            #   Call between .Ltmp1023 and .Ltmp1026
	.uleb128 .Ltmp1027-.Lfunc_begin5        #     jumps to .Ltmp1027
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1028-.Lfunc_begin5        # >> Call Site 69 <<
	.uleb128 .Ltmp1029-.Ltmp1028            #   Call between .Ltmp1028 and .Ltmp1029
	.uleb128 .Ltmp1030-.Lfunc_begin5        #     jumps to .Ltmp1030
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1031-.Lfunc_begin5        # >> Call Site 70 <<
	.uleb128 .Ltmp1034-.Ltmp1031            #   Call between .Ltmp1031 and .Ltmp1034
	.uleb128 .Ltmp1035-.Lfunc_begin5        #     jumps to .Ltmp1035
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1036-.Lfunc_begin5        # >> Call Site 71 <<
	.uleb128 .Ltmp1037-.Ltmp1036            #   Call between .Ltmp1036 and .Ltmp1037
	.uleb128 .Ltmp1040-.Lfunc_begin5        #     jumps to .Ltmp1040
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1088-.Lfunc_begin5        # >> Call Site 72 <<
	.uleb128 .Ltmp1115-.Ltmp1088            #   Call between .Ltmp1088 and .Ltmp1115
	.uleb128 .Ltmp1116-.Lfunc_begin5        #     jumps to .Ltmp1116
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1038-.Lfunc_begin5        # >> Call Site 73 <<
	.uleb128 .Ltmp1039-.Ltmp1038            #   Call between .Ltmp1038 and .Ltmp1039
	.uleb128 .Ltmp1040-.Lfunc_begin5        #     jumps to .Ltmp1040
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1041-.Lfunc_begin5        # >> Call Site 74 <<
	.uleb128 .Ltmp1042-.Ltmp1041            #   Call between .Ltmp1041 and .Ltmp1042
	.uleb128 .Ltmp1043-.Lfunc_begin5        #     jumps to .Ltmp1043
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1044-.Lfunc_begin5        # >> Call Site 75 <<
	.uleb128 .Ltmp1047-.Ltmp1044            #   Call between .Ltmp1044 and .Ltmp1047
	.uleb128 .Ltmp1048-.Lfunc_begin5        #     jumps to .Ltmp1048
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1049-.Lfunc_begin5        # >> Call Site 76 <<
	.uleb128 .Ltmp1052-.Ltmp1049            #   Call between .Ltmp1049 and .Ltmp1052
	.uleb128 .Ltmp1053-.Lfunc_begin5        #     jumps to .Ltmp1053
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1054-.Lfunc_begin5        # >> Call Site 77 <<
	.uleb128 .Ltmp1055-.Ltmp1054            #   Call between .Ltmp1054 and .Ltmp1055
	.uleb128 .Ltmp1072-.Lfunc_begin5        #     jumps to .Ltmp1072
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1056-.Lfunc_begin5        # >> Call Site 78 <<
	.uleb128 .Ltmp1057-.Ltmp1056            #   Call between .Ltmp1056 and .Ltmp1057
	.uleb128 .Ltmp1058-.Lfunc_begin5        #     jumps to .Ltmp1058
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1070-.Lfunc_begin5        # >> Call Site 79 <<
	.uleb128 .Ltmp1071-.Ltmp1070            #   Call between .Ltmp1070 and .Ltmp1071
	.uleb128 .Ltmp1072-.Lfunc_begin5        #     jumps to .Ltmp1072
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1071-.Lfunc_begin5        # >> Call Site 80 <<
	.uleb128 .Ltmp1122-.Ltmp1071            #   Call between .Ltmp1071 and .Ltmp1122
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1122-.Lfunc_begin5        # >> Call Site 81 <<
	.uleb128 .Ltmp1123-.Ltmp1122            #   Call between .Ltmp1122 and .Ltmp1123
	.uleb128 .Ltmp1126-.Lfunc_begin5        #     jumps to .Ltmp1126
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1117-.Lfunc_begin5        # >> Call Site 82 <<
	.uleb128 .Ltmp1118-.Ltmp1117            #   Call between .Ltmp1117 and .Ltmp1118
	.uleb128 .Ltmp1121-.Lfunc_begin5        #     jumps to .Ltmp1121
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1011-.Lfunc_begin5        # >> Call Site 83 <<
	.uleb128 .Ltmp1012-.Ltmp1011            #   Call between .Ltmp1011 and .Ltmp1012
	.uleb128 .Ltmp1013-.Lfunc_begin5        #     jumps to .Ltmp1013
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1014-.Lfunc_begin5        # >> Call Site 84 <<
	.uleb128 .Ltmp1015-.Ltmp1014            #   Call between .Ltmp1014 and .Ltmp1015
	.uleb128 .Ltmp1016-.Lfunc_begin5        #     jumps to .Ltmp1016
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1017-.Lfunc_begin5        # >> Call Site 85 <<
	.uleb128 .Ltmp1018-.Ltmp1017            #   Call between .Ltmp1017 and .Ltmp1018
	.uleb128 .Ltmp1019-.Lfunc_begin5        #     jumps to .Ltmp1019
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1059-.Lfunc_begin5        # >> Call Site 86 <<
	.uleb128 .Ltmp1060-.Ltmp1059            #   Call between .Ltmp1059 and .Ltmp1060
	.uleb128 .Ltmp1072-.Lfunc_begin5        #     jumps to .Ltmp1072
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1061-.Lfunc_begin5        # >> Call Site 87 <<
	.uleb128 .Ltmp1062-.Ltmp1061            #   Call between .Ltmp1061 and .Ltmp1062
	.uleb128 .Ltmp1063-.Lfunc_begin5        #     jumps to .Ltmp1063
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1064-.Lfunc_begin5        # >> Call Site 88 <<
	.uleb128 .Ltmp1065-.Ltmp1064            #   Call between .Ltmp1064 and .Ltmp1065
	.uleb128 .Ltmp1066-.Lfunc_begin5        #     jumps to .Ltmp1066
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1067-.Lfunc_begin5        # >> Call Site 89 <<
	.uleb128 .Ltmp1068-.Ltmp1067            #   Call between .Ltmp1067 and .Ltmp1068
	.uleb128 .Ltmp1069-.Lfunc_begin5        #     jumps to .Ltmp1069
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1068-.Lfunc_begin5        # >> Call Site 90 <<
	.uleb128 .Ltmp1127-.Ltmp1068            #   Call between .Ltmp1068 and .Ltmp1127
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1127-.Lfunc_begin5        # >> Call Site 91 <<
	.uleb128 .Ltmp1128-.Ltmp1127            #   Call between .Ltmp1127 and .Ltmp1128
	.uleb128 .Ltmp1129-.Lfunc_begin5        #     jumps to .Ltmp1129
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp820-.Lfunc_begin5         # >> Call Site 92 <<
	.uleb128 .Ltmp821-.Ltmp820              #   Call between .Ltmp820 and .Ltmp821
	.uleb128 .Ltmp822-.Lfunc_begin5         #     jumps to .Ltmp822
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp823-.Lfunc_begin5         # >> Call Site 93 <<
	.uleb128 .Ltmp824-.Ltmp823              #   Call between .Ltmp823 and .Ltmp824
	.uleb128 .Ltmp825-.Lfunc_begin5         #     jumps to .Ltmp825
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1079-.Lfunc_begin5        # >> Call Site 94 <<
	.uleb128 .Ltmp1080-.Ltmp1079            #   Call between .Ltmp1079 and .Ltmp1080
	.uleb128 .Ltmp1081-.Lfunc_begin5        #     jumps to .Ltmp1081
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1082-.Lfunc_begin5        # >> Call Site 95 <<
	.uleb128 .Ltmp1083-.Ltmp1082            #   Call between .Ltmp1082 and .Ltmp1083
	.uleb128 .Ltmp1084-.Lfunc_begin5        #     jumps to .Ltmp1084
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1085-.Lfunc_begin5        # >> Call Site 96 <<
	.uleb128 .Ltmp1086-.Ltmp1085            #   Call between .Ltmp1085 and .Ltmp1086
	.uleb128 .Ltmp1087-.Lfunc_begin5        #     jumps to .Ltmp1087
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1073-.Lfunc_begin5        # >> Call Site 97 <<
	.uleb128 .Ltmp1074-.Ltmp1073            #   Call between .Ltmp1073 and .Ltmp1074
	.uleb128 .Ltmp1075-.Lfunc_begin5        #     jumps to .Ltmp1075
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1076-.Lfunc_begin5        # >> Call Site 98 <<
	.uleb128 .Ltmp1077-.Ltmp1076            #   Call between .Ltmp1076 and .Ltmp1077
	.uleb128 .Ltmp1078-.Lfunc_begin5        #     jumps to .Ltmp1078
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1124-.Lfunc_begin5        # >> Call Site 99 <<
	.uleb128 .Ltmp1125-.Ltmp1124            #   Call between .Ltmp1124 and .Ltmp1125
	.uleb128 .Ltmp1126-.Lfunc_begin5        #     jumps to .Ltmp1126
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1119-.Lfunc_begin5        # >> Call Site 100 <<
	.uleb128 .Ltmp1120-.Ltmp1119            #   Call between .Ltmp1119 and .Ltmp1120
	.uleb128 .Ltmp1121-.Lfunc_begin5        #     jumps to .Ltmp1121
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1120-.Lfunc_begin5        # >> Call Site 101 <<
	.uleb128 .Lfunc_end11-.Ltmp1120         #   Call between .Ltmp1120 and .Lfunc_end11
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end5:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev,"axG",@progbits,_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev,comdat
	.weak	_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev # -- Begin function _ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev
	.p2align	1
	.prefalign	4, .Lfunc_end12, nop
	.type	_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev,@function
_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev: # @_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	movq	48(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB12_2
# %bb.1:
	movq	64(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB12_2:
	movq	24(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB12_4
# %bb.3:
	movq	40(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB12_4:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB12_5
# %bb.6:
	movq	16(%rbx), %rsi
	subq	%rdi, %rsi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB12_5:
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end12:
	.size	_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev, .Lfunc_end12-_ZNSt5arrayISt6vectorIfSaIfEELm3EED2Ev
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	1                               # -- Begin function _ZN12_GLOBAL__N_112DeviceBufferIjED2Ev
	.prefalign	4, .Lfunc_end13, nop
	.type	_ZN12_GLOBAL__N_112DeviceBufferIjED2Ev,@function
_ZN12_GLOBAL__N_112DeviceBufferIjED2Ev: # @_ZN12_GLOBAL__N_112DeviceBufferIjED2Ev
.Lfunc_begin6:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception6
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	testq	%rdi, %rdi
	je	.LBB13_2
# %bb.1:
.Ltmp1130:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1131:                              # EH_LABEL
.LBB13_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.LBB13_3:
	.cfi_def_cfa_offset 16
.Ltmp1132:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end13:
	.size	_ZN12_GLOBAL__N_112DeviceBufferIjED2Ev, .Lfunc_end13-_ZN12_GLOBAL__N_112DeviceBufferIjED2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table13:
.Lexception6:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase2-.Lttbaseref2
.Lttbaseref2:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end6-.Lcst_begin6
.Lcst_begin6:
	.uleb128 .Ltmp1130-.Lfunc_begin6        # >> Call Site 1 <<
	.uleb128 .Ltmp1131-.Ltmp1130            #   Call between .Ltmp1130 and .Ltmp1131
	.uleb128 .Ltmp1132-.Lfunc_begin6        #     jumps to .Ltmp1132
	.byte	1                               #   On action: 1
.Lcst_end6:
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
	.p2align	1                               # -- Begin function _ZNSt5arrayIN12_GLOBAL__N_17OutputsELm3EED2Ev
	.prefalign	4, .Lfunc_end14, nop
	.type	_ZNSt5arrayIN12_GLOBAL__N_17OutputsELm3EED2Ev,@function
_ZNSt5arrayIN12_GLOBAL__N_17OutputsELm3EED2Ev: # @_ZNSt5arrayIN12_GLOBAL__N_17OutputsELm3EED2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	addq	$192, %rdi
	callq	_ZN12_GLOBAL__N_17OutputsD2Ev
	leaq	96(%rbx), %rdi
	callq	_ZN12_GLOBAL__N_17OutputsD2Ev
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZN12_GLOBAL__N_17OutputsD2Ev   # TAILCALL
.Lfunc_end14:
	.size	_ZNSt5arrayIN12_GLOBAL__N_17OutputsELm3EED2Ev, .Lfunc_end14-_ZNSt5arrayIN12_GLOBAL__N_17OutputsELm3EED2Ev
	.cfi_endproc
                                        # -- End function
	.p2align	1                               # -- Begin function _ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev
	.prefalign	4, .Lfunc_end15, nop
	.type	_ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev,@function
_ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev: # @_ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev
.Lfunc_begin7:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception7
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	movq	32(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB15_2
# %bb.1:
.Ltmp1133:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1134:                              # EH_LABEL
.LBB15_2:
	movq	16(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB15_4
# %bb.3:
.Ltmp1135:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1136:                              # EH_LABEL
.LBB15_4:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB15_6
# %bb.5:
.Ltmp1137:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1138:                              # EH_LABEL
.LBB15_6:
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB15_7:
	.cfi_def_cfa_offset 16
.Ltmp1139:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end15:
	.size	_ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev, .Lfunc_end15-_ZNSt5arrayIN12_GLOBAL__N_112DeviceBufferI12hip_bfloat16EELm3EED2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table15:
.Lexception7:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase3-.Lttbaseref3
.Lttbaseref3:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end7-.Lcst_begin7
.Lcst_begin7:
	.uleb128 .Ltmp1133-.Lfunc_begin7        # >> Call Site 1 <<
	.uleb128 .Ltmp1138-.Ltmp1133            #   Call between .Ltmp1133 and .Ltmp1138
	.uleb128 .Ltmp1139-.Lfunc_begin7        #     jumps to .Ltmp1139
	.byte	1                               #   On action: 1
.Lcst_end7:
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
	.p2align	1                               # -- Begin function _ZN12_GLOBAL__N_112DeviceBufferIfED2Ev
	.prefalign	4, .Lfunc_end16, nop
	.type	_ZN12_GLOBAL__N_112DeviceBufferIfED2Ev,@function
_ZN12_GLOBAL__N_112DeviceBufferIfED2Ev: # @_ZN12_GLOBAL__N_112DeviceBufferIfED2Ev
.Lfunc_begin8:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception8
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	testq	%rdi, %rdi
	je	.LBB16_2
# %bb.1:
.Ltmp1140:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1141:                              # EH_LABEL
.LBB16_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.LBB16_3:
	.cfi_def_cfa_offset 16
.Ltmp1142:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end16:
	.size	_ZN12_GLOBAL__N_112DeviceBufferIfED2Ev, .Lfunc_end16-_ZN12_GLOBAL__N_112DeviceBufferIfED2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table16:
.Lexception8:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase4-.Lttbaseref4
.Lttbaseref4:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end8-.Lcst_begin8
.Lcst_begin8:
	.uleb128 .Ltmp1140-.Lfunc_begin8        # >> Call Site 1 <<
	.uleb128 .Ltmp1141-.Ltmp1140            #   Call between .Ltmp1140 and .Ltmp1141
	.uleb128 .Ltmp1142-.Lfunc_begin8        #     jumps to .Ltmp1142
	.byte	1                               #   On action: 1
.Lcst_end8:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase4:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text.__clang_call_terminate,"axG",@progbits,__clang_call_terminate,comdat
	.hidden	__clang_call_terminate          # -- Begin function __clang_call_terminate
	.weak	__clang_call_terminate
	.prefalign	4, .Lfunc_end17, nop
	.type	__clang_call_terminate,@function
__clang_call_terminate:                 # @__clang_call_terminate
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__cxa_begin_catch@PLT
	callq	_ZSt9terminatev@PLT
.Lfunc_end17:
	.size	__clang_call_terminate, .Lfunc_end17-__clang_call_terminate
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm,comdat
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm # -- Begin function _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm
	.p2align	1
	.prefalign	4, .Lfunc_end18, nop
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm,@function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm: # @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdi, %rbx
	movq	8(%rdi), %rax
	movq	%rdx, %rdi
	subq	%rax, %rdi
	movabsq	$9223372036854775807, %r9       # imm = 0x7FFFFFFFFFFFFFFF
	addq	%rdi, %r9
	cmpq	%r8, %r9
	jb	.LBB18_17
# %bb.1:
	movq	%r8, %r15
	subq	%rdx, %r15
	addq	%rax, %r15
	movq	(%rbx), %rdi
	leaq	16(%rbx), %r10
	movl	$15, %r9d
	cmpq	%r10, %rdi
	je	.LBB18_3
# %bb.2:
	movq	16(%rbx), %r9
.LBB18_3:
	cmpq	%r9, %r15
	jbe	.LBB18_4
# %bb.15:
	movq	%rbx, %rdi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	jmp	.LBB18_16
.LBB18_4:
	leaq	(%rdi,%rsi), %r14
	addq	%rdx, %rsi
	movq	%rax, %r9
	subq	%rsi, %r9
	cmpq	%rdi, %rcx
	setb	%r10b
	addq	%rax, %rdi
	cmpq	%rcx, %rdi
	setb	%dil
	orb	%r10b, %dil
	cmpb	$1, %dil
	jne	.LBB18_14
# %bb.5:
	cmpq	%rdx, %r8
	je	.LBB18_10
# %bb.6:
	cmpq	%rsi, %rax
	je	.LBB18_10
# %bb.7:
	leaq	(%r14,%r8), %rdi
	addq	%r14, %rdx
	cmpq	$1, %r9
	jne	.LBB18_9
# %bb.8:
	movzbl	(%rdx), %eax
	movb	%al, (%rdi)
.LBB18_10:
	testq	%r8, %r8
	je	.LBB18_16
.LBB18_11:
	cmpq	$1, %r8
	jne	.LBB18_13
# %bb.12:
	movzbl	(%rcx), %eax
	movb	%al, (%r14)
	jmp	.LBB18_16
.LBB18_13:
	movq	%r14, %rdi
	movq	%rcx, %rsi
	movq	%r8, %rdx
	callq	memcpy@PLT
	jmp	.LBB18_16
.LBB18_9:
	movq	%rdx, %rsi
	movq	%r9, %rdx
	movq	%r8, %r12
	movq	%rcx, %r13
	callq	memmove@PLT
	movq	%r13, %rcx
	movq	%r12, %r8
	testq	%r8, %r8
	jne	.LBB18_11
.LBB18_16:
	movq	%r15, 8(%rbx)
	movq	(%rbx), %rax
	movb	$0, (%rax,%r15)
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB18_14:
	.cfi_def_cfa_offset 48
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_replace_coldEPcmPKcmm@PLT
	jmp	.LBB18_16
.LBB18_17:
	leaq	.L.str.37(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end18:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm, .Lfunc_end18-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm,comdat
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm # -- Begin function _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	.p2align	1
	.prefalign	4, .Lfunc_end19, nop
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm,@function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm: # @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
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
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	movq	(%rdi), %rbp
	movq	8(%rdi), %r13
	movq	%r8, %r15
	movq	%rdx, 24(%rsp)                  # 8-byte Spill
	subq	%rdx, %r15
	leaq	16(%rdi), %rax
	cmpq	%rax, %rbp
	movq	16(%rdi), %rcx
	movl	$15, %eax
	cmovneq	%rcx, %rax
	addq	%r13, %r15
	js	.LBB19_20
# %bb.1:
	movq	%r8, %r14
	movq	%rsi, %r12
	movq	%rdi, %rbx
	cmpq	%rax, %r15
	jbe	.LBB19_4
# %bb.2:
	addq	%rax, %rax
	cmpq	%rax, %r15
	jae	.LBB19_4
# %bb.3:
	movabsq	$9223372036854775807, %r15      # imm = 0x7FFFFFFFFFFFFFFF
	cmpq	%r15, %rax
	cmovbq	%rax, %r15
.LBB19_4:
	movq	%r15, %rdi
	incq	%rdi
	js	.LBB19_21
# %bb.5:
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	callq	_Znwm@PLT
	movq	%rbp, %rdx
	movq	%rax, %rsi
	testq	%r12, %r12
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	je	.LBB19_9
# %bb.6:
	cmpq	$1, %r12
	jne	.LBB19_8
# %bb.7:
	movzbl	(%rdx), %eax
	movb	%al, (%rsi)
	jmp	.LBB19_9
.LBB19_8:
	movq	%rsi, %rdi
	movq	%rbp, %rsi
	movq	%r12, %rdx
	callq	memcpy@PLT
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	%rbp, %rdx
.LBB19_9:
	movq	24(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r12), %r8
	cmpq	$0, 8(%rsp)                     # 8-byte Folded Reload
	sete	%al
	testq	%r14, %r14
	sete	%cl
	orb	%al, %cl
	je	.LBB19_10
# %bb.13:
	cmpq	%r8, %r13
	jne	.LBB19_14
.LBB19_17:
	leaq	16(%rbx), %rax
	cmpq	%rax, %rdx
	je	.LBB19_19
.LBB19_18:
	movq	32(%rsp), %rsi                  # 8-byte Reload
	incq	%rsi
	movq	%rdx, %rdi
	callq	_ZdlPvm@PLT
	movq	16(%rsp), %rsi                  # 8-byte Reload
.LBB19_19:
	movq	%rsi, (%rbx)
	movq	%r15, 16(%rbx)
	addq	$40, %rsp
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
.LBB19_10:
	.cfi_def_cfa_offset 96
	leaq	(%rsi,%r12), %rdi
	cmpq	$1, %r14
	jne	.LBB19_12
# %bb.11:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movzbl	(%rax), %eax
	movb	%al, (%rdi)
	cmpq	%r8, %r13
	je	.LBB19_17
	jmp	.LBB19_14
.LBB19_12:
	movq	8(%rsp), %rsi                   # 8-byte Reload
	movq	%r14, %rdx
	movq	%r8, 8(%rsp)                    # 8-byte Spill
	callq	memcpy@PLT
	movq	8(%rsp), %r8                    # 8-byte Reload
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	%rbp, %rdx
	cmpq	%r8, %r13
	je	.LBB19_17
.LBB19_14:
	subq	%r8, %r13
	movq	%rsi, %rdi
	addq	%r12, %rdi
	addq	%r14, %rdi
	addq	%rdx, %r12
	addq	24(%rsp), %r12                  # 8-byte Folded Reload
	cmpq	$1, %r13
	jne	.LBB19_16
# %bb.15:
	movzbl	(%r12), %eax
	movb	%al, (%rdi)
	leaq	16(%rbx), %rax
	cmpq	%rax, %rdx
	jne	.LBB19_18
	jmp	.LBB19_19
.LBB19_16:
	movq	%r12, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	%rbp, %rdx
	leaq	16(%rbx), %rax
	cmpq	%rax, %rdx
	jne	.LBB19_18
	jmp	.LBB19_19
.LBB19_21:
	callq	_ZSt17__throw_bad_allocv@PLT
.LBB19_20:
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end19:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm, .Lfunc_end19-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_,comdat
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_ # -- Begin function _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_
	.p2align	1
	.prefalign	4, .Lfunc_end20, nop
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_,@function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_: # @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_
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
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	cmpq	%rsi, %rdi
	je	.LBB20_13
# %bb.1:
	movq	%rdi, %rbx
	movq	8(%rsi), %r14
	movq	(%rdi), %r15
	movq	16(%rdi), %r12
	leaq	16(%rdi), %rax
	cmpq	%rax, %r15
	movl	$15, %eax
	cmovneq	%r12, %rax
	cmpq	%rax, %r14
	jbe	.LBB20_7
# %bb.2:
	testq	%r14, %r14
	js	.LBB20_14
# %bb.3:
	addq	%rax, %rax
	movabsq	$9223372036854775807, %rbp      # imm = 0x7FFFFFFFFFFFFFFF
	cmpq	%rbp, %rax
	cmovbq	%rax, %rbp
	cmpq	%rax, %r14
	cmovaeq	%r14, %rbp
	movq	%rbp, %rdi
	incq	%rdi
	js	.LBB20_15
# %bb.4:
	movq	%rsi, (%rsp)                    # 8-byte Spill
	callq	_Znwm@PLT
	movq	%rax, %r13
	leaq	16(%rbx), %rax
	cmpq	%rax, %r15
	je	.LBB20_6
# %bb.5:
	incq	%r12
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB20_6:
	movq	%r13, (%rbx)
	movq	%rbp, 16(%rbx)
	movq	(%rsp), %rsi                    # 8-byte Reload
	movq	(%rsi), %rsi
	cmpq	$1, %r14
	jne	.LBB20_11
.LBB20_10:
	movzbl	(%rsi), %eax
	movb	%al, (%r13)
	jmp	.LBB20_12
.LBB20_7:
	testq	%r14, %r14
	je	.LBB20_16
# %bb.8:
	movq	%r15, %r13
	movq	(%rsi), %rsi
	cmpq	$1, %r14
	je	.LBB20_10
.LBB20_11:
	movq	%r13, %rdi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB20_12:
	movq	%r14, 8(%rbx)
	movq	(%rbx), %rax
	movb	$0, (%rax,%r14)
.LBB20_13:
	addq	$8, %rsp
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
.LBB20_16:
	.cfi_def_cfa_offset 64
	movq	$0, 8(%rbx)
	movb	$0, (%r15)
	jmp	.LBB20_13
.LBB20_15:
	callq	_ZSt17__throw_bad_allocv@PLT
.LBB20_14:
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end20:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_, .Lfunc_end20-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE,"axG",@progbits,_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE,comdat
	.weak	_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE # -- Begin function _ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE
	.prefalign	4, .Lfunc_end21, nop
	.type	_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE,@function
_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE: # @_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE
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
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r8, %r14
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	movq	%rdx, %r12
	movq	%rsi, 16(%rsp)                  # 8-byte Spill
	movq	%rdi, %rbx
	leaq	16(%rdi), %rbp
	movq	%rbp, (%rdi)
	movq	$0, 8(%rdi)
	movb	$0, 16(%rdi)
	leaq	(%r8,%rdx), %rsi
.Ltmp1143:                              # EH_LABEL
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
.Ltmp1144:                              # EH_LABEL
# %bb.1:
	movabsq	$9223372036854775807, %r15      # imm = 0x7FFFFFFFFFFFFFFF
	movq	8(%rbx), %rsi
	movq	%r15, %rax
	subq	%rsi, %rax
	cmpq	%r12, %rax
	jb	.LBB21_11
# %bb.2:
	leaq	(%rsi,%r12), %r13
	movq	(%rbx), %rdi
	movl	$15, %eax
	cmpq	%rbp, %rdi
	je	.LBB21_4
# %bb.3:
	movq	(%rbp), %rax
.LBB21_4:
	cmpq	%rax, %r13
	jbe	.LBB21_5
# %bb.9:
.Ltmp1145:                              # EH_LABEL
	movq	%rbx, %rdi
	xorl	%edx, %edx
	movq	16(%rsp), %rcx                  # 8-byte Reload
	movq	%r12, %r8
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp1146:                              # EH_LABEL
	jmp	.LBB21_10
.LBB21_5:
	testq	%r12, %r12
	je	.LBB21_10
# %bb.6:
	addq	%rsi, %rdi
	cmpq	$1, %r12
	jne	.LBB21_8
# %bb.7:
	movq	16(%rsp), %rax                  # 8-byte Reload
	movzbl	(%rax), %eax
	movb	%al, (%rdi)
	jmp	.LBB21_10
.LBB21_8:
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	%r12, %rdx
	callq	memcpy@PLT
.LBB21_10:
	movq	%r13, 8(%rbx)
	movq	(%rbx), %rax
	movb	$0, (%rax,%r13)
	movq	8(%rbx), %rsi
	subq	%rsi, %r15
	cmpq	%r14, %r15
	jb	.LBB21_11
# %bb.13:
	leaq	(%rsi,%r14), %r15
	movq	(%rbx), %rdi
	movl	$15, %eax
	cmpq	%rbp, %rdi
	je	.LBB21_15
# %bb.14:
	movq	(%rbp), %rax
.LBB21_15:
	cmpq	%rax, %r15
	jbe	.LBB21_16
# %bb.19:
.Ltmp1147:                              # EH_LABEL
	movq	%rbx, %rdi
	xorl	%edx, %edx
	movq	8(%rsp), %rcx                   # 8-byte Reload
	movq	%r14, %r8
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp1148:                              # EH_LABEL
	jmp	.LBB21_24
.LBB21_16:
	testq	%r14, %r14
	je	.LBB21_24
# %bb.17:
	addq	%rsi, %rdi
	cmpq	$1, %r14
	jne	.LBB21_23
# %bb.18:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movzbl	(%rax), %eax
	movb	%al, (%rdi)
	jmp	.LBB21_24
.LBB21_23:
	movq	8(%rsp), %rsi                   # 8-byte Reload
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB21_24:
	movq	%r15, 8(%rbx)
	movq	(%rbx), %rax
	movb	$0, (%rax,%r15)
	movq	%rbx, %rax
	addq	$24, %rsp
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
.LBB21_11:
	.cfi_def_cfa_offset 80
.Ltmp1149:                              # EH_LABEL
	leaq	.L.str.39(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1150:                              # EH_LABEL
# %bb.12:
.LBB21_20:
.Ltmp1151:                              # EH_LABEL
	movq	%rax, %r14
	movq	(%rbx), %rdi
	cmpq	%rbp, %rdi
	je	.LBB21_22
# %bb.21:
	movq	(%rbp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB21_22:
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end21:
	.size	_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE, .Lfunc_end21-_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE
	.cfi_endproc
	.section	.gcc_except_table._ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE,"aG",@progbits,_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE,comdat
	.p2align	2, 0x0
GCC_except_table21:
.Lexception9:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end9-.Lcst_begin9
.Lcst_begin9:
	.uleb128 .Ltmp1143-.Lfunc_begin9        # >> Call Site 1 <<
	.uleb128 .Ltmp1146-.Ltmp1143            #   Call between .Ltmp1143 and .Ltmp1146
	.uleb128 .Ltmp1151-.Lfunc_begin9        #     jumps to .Ltmp1151
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1146-.Lfunc_begin9        # >> Call Site 2 <<
	.uleb128 .Ltmp1147-.Ltmp1146            #   Call between .Ltmp1146 and .Ltmp1147
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1147-.Lfunc_begin9        # >> Call Site 3 <<
	.uleb128 .Ltmp1148-.Ltmp1147            #   Call between .Ltmp1147 and .Ltmp1148
	.uleb128 .Ltmp1151-.Lfunc_begin9        #     jumps to .Ltmp1151
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1148-.Lfunc_begin9        # >> Call Site 4 <<
	.uleb128 .Ltmp1149-.Ltmp1148            #   Call between .Ltmp1148 and .Ltmp1149
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1149-.Lfunc_begin9        # >> Call Site 5 <<
	.uleb128 .Ltmp1150-.Ltmp1149            #   Call between .Ltmp1149 and .Ltmp1150
	.uleb128 .Ltmp1151-.Lfunc_begin9        #     jumps to .Ltmp1151
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1150-.Lfunc_begin9        # >> Call Site 6 <<
	.uleb128 .Lfunc_end21-.Ltmp1150         #   Call between .Ltmp1150 and .Lfunc_end21
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end9:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm,comdat
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm # -- Begin function _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	.p2align	1
	.prefalign	4, .Lfunc_end22, nop
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm,@function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm: # @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
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
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	(%rdi), %r14
	movq	16(%rdi), %r15
	leaq	16(%rdi), %r13
	cmpq	%r13, %r14
	movl	$15, %eax
	cmovneq	%r15, %rax
	cmpq	%rax, %rsi
	jbe	.LBB22_10
# %bb.1:
	testq	%rsi, %rsi
	js	.LBB22_11
# %bb.2:
	movq	%rdi, %rbx
	addq	%rax, %rax
	movabsq	$9223372036854775807, %rbp      # imm = 0x7FFFFFFFFFFFFFFF
	cmpq	%rbp, %rax
	cmovbq	%rax, %rbp
	cmpq	%rax, %rsi
	cmovaeq	%rsi, %rbp
	movq	%rbp, %rdi
	incq	%rdi
	js	.LBB22_12
# %bb.3:
	callq	_Znwm@PLT
	movq	%rax, %r12
	movq	8(%rbx), %rdx
	incq	%rdx
	je	.LBB22_7
# %bb.4:
	cmpq	$1, %rdx
	jne	.LBB22_6
# %bb.5:
	movzbl	(%r14), %eax
	movb	%al, (%r12)
.LBB22_7:
	cmpq	%r13, %r14
	je	.LBB22_9
.LBB22_8:
	incq	%r15
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB22_9:
	movq	%r12, (%rbx)
	movq	%rbp, 16(%rbx)
.LBB22_10:
	addq	$8, %rsp
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
.LBB22_6:
	.cfi_def_cfa_offset 64
	movq	%r12, %rdi
	movq	%r14, %rsi
	callq	memcpy@PLT
	cmpq	%r13, %r14
	jne	.LBB22_8
	jmp	.LBB22_9
.LBB22_12:
	callq	_ZSt17__throw_bad_allocv@PLT
.LBB22_11:
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end22:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm, .Lfunc_end22-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	1                               # -- Begin function _ZN12_GLOBAL__N_17OutputsD2Ev
	.prefalign	4, .Lfunc_end23, nop
	.type	_ZN12_GLOBAL__N_17OutputsD2Ev,@function
_ZN12_GLOBAL__N_17OutputsD2Ev:          # @_ZN12_GLOBAL__N_17OutputsD2Ev
.Lfunc_begin10:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception10
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	movq	72(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB23_2
# %bb.1:
.Ltmp1152:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1153:                              # EH_LABEL
.LBB23_2:
	movq	48(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB23_4
# %bb.3:
.Ltmp1155:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1156:                              # EH_LABEL
.LBB23_4:
	movq	24(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB23_6
# %bb.5:
.Ltmp1158:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1159:                              # EH_LABEL
.LBB23_6:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB23_8
# %bb.7:
.Ltmp1161:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1162:                              # EH_LABEL
.LBB23_8:
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB23_12:
	.cfi_def_cfa_offset 16
.Ltmp1163:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB23_11:
.Ltmp1160:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB23_10:
.Ltmp1157:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.LBB23_9:
.Ltmp1154:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end23:
	.size	_ZN12_GLOBAL__N_17OutputsD2Ev, .Lfunc_end23-_ZN12_GLOBAL__N_17OutputsD2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table23:
.Lexception10:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase5-.Lttbaseref5
.Lttbaseref5:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end10-.Lcst_begin10
.Lcst_begin10:
	.uleb128 .Ltmp1152-.Lfunc_begin10       # >> Call Site 1 <<
	.uleb128 .Ltmp1153-.Ltmp1152            #   Call between .Ltmp1152 and .Ltmp1153
	.uleb128 .Ltmp1154-.Lfunc_begin10       #     jumps to .Ltmp1154
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp1155-.Lfunc_begin10       # >> Call Site 2 <<
	.uleb128 .Ltmp1156-.Ltmp1155            #   Call between .Ltmp1155 and .Ltmp1156
	.uleb128 .Ltmp1157-.Lfunc_begin10       #     jumps to .Ltmp1157
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp1158-.Lfunc_begin10       # >> Call Site 3 <<
	.uleb128 .Ltmp1159-.Ltmp1158            #   Call between .Ltmp1158 and .Ltmp1159
	.uleb128 .Ltmp1160-.Lfunc_begin10       #     jumps to .Ltmp1160
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp1161-.Lfunc_begin10       # >> Call Site 4 <<
	.uleb128 .Ltmp1162-.Ltmp1161            #   Call between .Ltmp1161 and .Ltmp1162
	.uleb128 .Ltmp1163-.Lfunc_begin10       #     jumps to .Ltmp1163
	.byte	1                               #   On action: 1
.Lcst_end10:
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
	.p2align	1                               # -- Begin function _ZN12_GLOBAL__N_112GuardedFloatD2Ev
	.prefalign	4, .Lfunc_end24, nop
	.type	_ZN12_GLOBAL__N_112GuardedFloatD2Ev,@function
_ZN12_GLOBAL__N_112GuardedFloatD2Ev:    # @_ZN12_GLOBAL__N_112GuardedFloatD2Ev
.Lfunc_begin11:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception11
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	testq	%rdi, %rdi
	je	.LBB24_2
# %bb.1:
.Ltmp1164:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1165:                              # EH_LABEL
.LBB24_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.LBB24_3:
	.cfi_def_cfa_offset 16
.Ltmp1166:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end24:
	.size	_ZN12_GLOBAL__N_112GuardedFloatD2Ev, .Lfunc_end24-_ZN12_GLOBAL__N_112GuardedFloatD2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table24:
.Lexception11:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase6-.Lttbaseref6
.Lttbaseref6:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end11-.Lcst_begin11
.Lcst_begin11:
	.uleb128 .Ltmp1164-.Lfunc_begin11       # >> Call Site 1 <<
	.uleb128 .Ltmp1165-.Ltmp1164            #   Call between .Ltmp1164 and .Ltmp1165
	.uleb128 .Ltmp1166-.Lfunc_begin11       #     jumps to .Ltmp1166
	.byte	1                               #   On action: 1
.Lcst_end11:
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
	.p2align	1                               # -- Begin function _ZN12_GLOBAL__N_111GuardedBf16D2Ev
	.prefalign	4, .Lfunc_end25, nop
	.type	_ZN12_GLOBAL__N_111GuardedBf16D2Ev,@function
_ZN12_GLOBAL__N_111GuardedBf16D2Ev:     # @_ZN12_GLOBAL__N_111GuardedBf16D2Ev
.Lfunc_begin12:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception12
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	testq	%rdi, %rdi
	je	.LBB25_2
# %bb.1:
.Ltmp1167:                              # EH_LABEL
	callq	hipFree@PLT
.Ltmp1168:                              # EH_LABEL
.LBB25_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.LBB25_3:
	.cfi_def_cfa_offset 16
.Ltmp1169:                              # EH_LABEL
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end25:
	.size	_ZN12_GLOBAL__N_111GuardedBf16D2Ev, .Lfunc_end25-_ZN12_GLOBAL__N_111GuardedBf16D2Ev
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table25:
.Lexception12:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase7-.Lttbaseref7
.Lttbaseref7:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end12-.Lcst_begin12
.Lcst_begin12:
	.uleb128 .Ltmp1167-.Lfunc_begin12       # >> Call Site 1 <<
	.uleb128 .Ltmp1168-.Ltmp1167            #   Call between .Ltmp1167 and .Ltmp1168
	.uleb128 .Ltmp1169-.Lfunc_begin12       #     jumps to .Ltmp1169
	.byte	1                               #   On action: 1
.Lcst_end12:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase7:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function _ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.LCPI26_0:
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.short	32602                           # 0x7f5a
	.text
	.prefalign	4, .Lfunc_end26, nop
	.type	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_,@function
_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_: # @_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
.Lfunc_begin13:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception13
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	2(%rsi), %r15
	movq	%r15, %rax
	shrq	$62, %rax
	jne	.LBB26_22
# %bb.1:
	testq	%r15, %r15
	je	.LBB26_2
# %bb.3:
	movq	%rsi, %r14
	movq	%rdi, %r12
	leaq	(%r15,%r15), %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbx
	movw	$0, (%rax)
	leaq	2(%rax), %rdx
	incq	%r14
	je	.LBB26_5
# %bb.4:
	leaq	(%r14,%r14), %rax
	movq	%rdx, %rdi
	xorl	%esi, %esi
	movq	%rdx, %r13
	movq	%rax, %rdx
	callq	memset@PLT
	leaq	(,%r14,2), %rdx
	addq	%r13, %rdx
.LBB26_5:
	leaq	(%rbx,%r15,2), %r14
	movq	%rdx, %rsi
	subq	%rbx, %rsi
	addq	$-2, %rsi
	movq	%rbx, %rax
	cmpq	$6, %rsi
	movq	%r12, %rdi
	jb	.LBB26_23
# %bb.6:
	movq	%rsi, %rcx
	shrq	%rcx
	incq	%rcx
	cmpq	$30, %rsi
	jae	.LBB26_8
# %bb.7:
	xorl	%esi, %esi
	jmp	.LBB26_12
.LBB26_2:
	xorl	%edx, %edx
	xorl	%ebx, %ebx
	xorl	%r14d, %r14d
	jmp	.LBB26_15
.LBB26_8:
	movq	%rcx, %rsi
	andq	$-16, %rsi
	leaq	(%rbx,%rsi,2), %rax
	xorl	%r8d, %r8d
	movaps	.LCPI26_0(%rip), %xmm0          # xmm0 = [32602,32602,32602,32602,32602,32602,32602,32602]
	.p2align	4
.LBB26_9:                               # =>This Inner Loop Header: Depth=1
	movups	%xmm0, (%rbx,%r8,2)
	movups	%xmm0, 16(%rbx,%r8,2)
	addq	$16, %r8
	cmpq	%r8, %rsi
	jne	.LBB26_9
# %bb.10:
	cmpq	%rsi, %rcx
	je	.LBB26_15
# %bb.11:
	testb	$12, %cl
	je	.LBB26_23
.LBB26_12:
	movq	%rcx, %r8
	andq	$-4, %r8
	leaq	(%rbx,%r8,2), %rax
	movabsq	$9176787217381228378, %r9       # imm = 0x7F5A7F5A7F5A7F5A
	.p2align	4
.LBB26_13:                              # =>This Inner Loop Header: Depth=1
	movq	%r9, (%rbx,%rsi,2)
	addq	$4, %rsi
	cmpq	%rsi, %r8
	jne	.LBB26_13
# %bb.14:
	cmpq	%r8, %rcx
	je	.LBB26_15
	.p2align	4
.LBB26_23:                              # =>This Inner Loop Header: Depth=1
	movw	$32602, (%rax)                  # imm = 0x7F5A
	addq	$2, %rax
	cmpq	%rdx, %rax
	jne	.LBB26_23
.LBB26_15:
	subq	%rbx, %rdx
.Ltmp1170:                              # EH_LABEL
	movq	%rbx, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp1171:                              # EH_LABEL
# %bb.16:
.Ltmp1172:                              # EH_LABEL
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp1173:                              # EH_LABEL
# %bb.17:
	testq	%rbx, %rbx
	je	.LBB26_18
# %bb.24:
	subq	%rbx, %r14
	movq	%rbx, %rdi
	movq	%r14, %rsi
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB26_18:
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB26_22:
	.cfi_def_cfa_offset 48
	leaq	.L.str.40(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB26_19:
.Ltmp1174:                              # EH_LABEL
	movq	%rax, %r15
	testq	%rbx, %rbx
	je	.LBB26_21
# %bb.20:
	subq	%rbx, %r14
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
.LBB26_21:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end26:
	.size	_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_, .Lfunc_end26-_ZN12_GLOBAL__N_116initialize_guardINS_11GuardedBf16EEEvRT_
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table26:
.Lexception13:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end13-.Lcst_begin13
.Lcst_begin13:
	.uleb128 .Lfunc_begin13-.Lfunc_begin13  # >> Call Site 1 <<
	.uleb128 .Ltmp1170-.Lfunc_begin13       #   Call between .Lfunc_begin13 and .Ltmp1170
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1170-.Lfunc_begin13       # >> Call Site 2 <<
	.uleb128 .Ltmp1173-.Ltmp1170            #   Call between .Ltmp1170 and .Ltmp1173
	.uleb128 .Ltmp1174-.Lfunc_begin13       #     jumps to .Ltmp1174
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1173-.Lfunc_begin13       # >> Call Site 3 <<
	.uleb128 .Lfunc_end26-.Ltmp1173         #   Call between .Ltmp1173 and .Lfunc_end26
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end13:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function _ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.LCPI27_0:
	.long	0xc7f12060                      # float -123456.75
	.long	0xc7f12060                      # float -123456.75
	.long	0xc7f12060                      # float -123456.75
	.long	0xc7f12060                      # float -123456.75
	.text
	.prefalign	4, .Lfunc_end27, nop
	.type	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_,@function
_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_: # @_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
.Lfunc_begin14:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception14
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	2(%rsi), %r15
	movq	%r15, %rax
	shrq	$61, %rax
	jne	.LBB27_16
# %bb.1:
	testq	%r15, %r15
	je	.LBB27_2
# %bb.3:
	movq	%rsi, %r14
	movq	%rdi, %r12
	leaq	(,%r15,4), %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbx
	movl	$0, (%rax)
	leaq	4(%rax), %rdx
	incq	%r14
	je	.LBB27_5
# %bb.4:
	leaq	(,%r14,4), %rax
	movq	%rdx, %rdi
	xorl	%esi, %esi
	movq	%rdx, %r13
	movq	%rax, %rdx
	callq	memset@PLT
	leaq	(,%r14,4), %rdx
	addq	%r13, %rdx
.LBB27_5:
	leaq	(%rbx,%r15,4), %r14
	movq	%rdx, %rcx
	subq	%rbx, %rcx
	addq	$-4, %rcx
	movq	%rbx, %rax
	cmpq	$28, %rcx
	movq	%r12, %rdi
	jb	.LBB27_17
# %bb.6:
	shrq	$2, %rcx
	incq	%rcx
	movq	%rcx, %rsi
	andq	$-8, %rsi
	leaq	(%rbx,%rsi,4), %rax
	xorl	%r8d, %r8d
	movaps	.LCPI27_0(%rip), %xmm0          # xmm0 = [-1.2345675E+5,-1.2345675E+5,-1.2345675E+5,-1.2345675E+5]
	.p2align	4
.LBB27_7:                               # =>This Inner Loop Header: Depth=1
	movups	%xmm0, (%rbx,%r8,4)
	movups	%xmm0, 16(%rbx,%r8,4)
	addq	$8, %r8
	cmpq	%r8, %rsi
	jne	.LBB27_7
# %bb.8:
	cmpq	%rsi, %rcx
	je	.LBB27_9
	.p2align	4
.LBB27_17:                              # =>This Inner Loop Header: Depth=1
	movl	$-940498848, (%rax)             # imm = 0xC7F12060
	addq	$4, %rax
	cmpq	%rdx, %rax
	jne	.LBB27_17
	jmp	.LBB27_9
.LBB27_2:
	xorl	%edx, %edx
	xorl	%ebx, %ebx
	xorl	%r14d, %r14d
.LBB27_9:
	subq	%rbx, %rdx
.Ltmp1175:                              # EH_LABEL
	movq	%rbx, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp1176:                              # EH_LABEL
# %bb.10:
.Ltmp1177:                              # EH_LABEL
	leaq	.L.str.140(%rip), %rsi
	movl	%eax, %edi
	movl	$286, %edx                      # imm = 0x11E
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp1178:                              # EH_LABEL
# %bb.11:
	testq	%rbx, %rbx
	je	.LBB27_12
# %bb.18:
	subq	%rbx, %r14
	movq	%rbx, %rdi
	movq	%r14, %rsi
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB27_12:
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB27_16:
	.cfi_def_cfa_offset 48
	leaq	.L.str.40(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB27_13:
.Ltmp1179:                              # EH_LABEL
	movq	%rax, %r15
	testq	%rbx, %rbx
	je	.LBB27_15
# %bb.14:
	subq	%rbx, %r14
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
.LBB27_15:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end27:
	.size	_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_, .Lfunc_end27-_ZN12_GLOBAL__N_116initialize_guardINS_12GuardedFloatEEEvRT_
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table27:
.Lexception14:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end14-.Lcst_begin14
.Lcst_begin14:
	.uleb128 .Lfunc_begin14-.Lfunc_begin14  # >> Call Site 1 <<
	.uleb128 .Ltmp1175-.Lfunc_begin14       #   Call between .Lfunc_begin14 and .Ltmp1175
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1175-.Lfunc_begin14       # >> Call Site 2 <<
	.uleb128 .Ltmp1178-.Ltmp1175            #   Call between .Ltmp1175 and .Ltmp1178
	.uleb128 .Ltmp1179-.Lfunc_begin14       #     jumps to .Ltmp1179
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1178-.Lfunc_begin14       # >> Call Site 3 <<
	.uleb128 .Lfunc_end27-.Ltmp1178         #   Call between .Ltmp1178 and .Lfunc_end27
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end14:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end28, nop    # -- Begin function _ZN12_GLOBAL__N_15validERKNS_9ArgumentsE
	.type	_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE,@function
_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE: # @_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE
	.cfi_startproc
# %bb.0:
	movq	(%rdi), %r8
	testq	%r8, %r8
	je	.LBB28_1
# %bb.3:
	movq	8(%rdi), %rsi
	testq	%rsi, %rsi
	je	.LBB28_1
# %bb.4:
	movq	16(%rdi), %rdx
	testq	%rdx, %rdx
	je	.LBB28_1
# %bb.5:
	movq	24(%rdi), %rcx
	testq	%rcx, %rcx
	je	.LBB28_1
# %bb.6:
	movq	32(%rdi), %rax
	testq	%rax, %rax
	je	.LBB28_1
# %bb.7:
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
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	40(%rdi), %rbp
	testq	%rbp, %rbp
	je	.LBB28_37
# %bb.8:
	movq	48(%rdi), %rbx
	testq	%rbx, %rbx
	je	.LBB28_37
# %bb.9:
	movq	56(%rdi), %r10
	testq	%r10, %r10
	je	.LBB28_37
# %bb.10:
	movq	64(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB28_37
# %bb.11:
	leaq	96(%rbp), %r13
	leaq	96(%rbx), %r15
	cmpq	%rbp, %r15
	seta	%r9b
	cmpq	%rbx, %r13
	seta	%r11b
	testb	%r9b, %r11b
	jne	.LBB28_37
# %bb.12:
	leaq	192(%r10), %r11
	cmpq	%rbp, %r11
	seta	%r9b
	cmpq	%r10, %r13
	seta	%r14b
	testb	%r9b, %r14b
	jne	.LBB28_37
# %bb.13:
	leaq	192(%rdi), %r9
	cmpq	%rbp, %r9
	seta	%r14b
	cmpq	%rdi, %r13
	seta	%r12b
	testb	%r14b, %r12b
	jne	.LBB28_37
# %bb.14:
	cmpq	%rbx, %r11
	seta	%r14b
	cmpq	%r10, %r15
	seta	%r12b
	testb	%r14b, %r12b
	jne	.LBB28_37
# %bb.15:
	cmpq	%rbx, %r9
	seta	%r14b
	cmpq	%rdi, %r15
	seta	%r12b
	testb	%r14b, %r12b
	jne	.LBB28_37
# %bb.16:
	cmpq	%r10, %r9
	seta	%r14b
	cmpq	%rdi, %r11
	seta	%r12b
	testb	%r14b, %r12b
	jne	.LBB28_37
# %bb.17:
	leaq	10240(%r8), %r14
	cmpq	%rbp, %r14
	setbe	%r12b
	cmpq	%r8, %r13
	setbe	%r14b
	orb	%r12b, %r14b
	je	.LBB28_37
# %bb.18:
	leaq	491520(%rsi), %r14
	cmpq	%rbp, %r14
	seta	%r14b
	cmpq	%rsi, %r13
	seta	%r12b
	testb	%r14b, %r12b
	jne	.LBB28_37
# %bb.19:
	leaq	491520(%rdx), %r14
	cmpq	%rbp, %r14
	seta	%r14b
	cmpq	%rdx, %r13
	seta	%r12b
	testb	%r14b, %r12b
	jne	.LBB28_37
# %bb.20:
	leaq	192(%rcx), %r14
	cmpq	%rbp, %r14
	seta	%r14b
	cmpq	%rcx, %r13
	seta	%r12b
	testb	%r14b, %r12b
	jne	.LBB28_37
# %bb.21:
	leaq	192(%rax), %r12
	cmpq	%rbp, %r12
	seta	%bpl
	cmpq	%rax, %r13
	seta	%r14b
	testb	%bpl, %r14b
	jne	.LBB28_37
# %bb.22:
	leaq	10240(%r8), %r14
	cmpq	%rbx, %r14
	seta	%bpl
	cmpq	%r8, %r15
	seta	%r14b
	testb	%bpl, %r14b
	jne	.LBB28_37
# %bb.23:
	leaq	491520(%rsi), %r14
	cmpq	%rbx, %r14
	seta	%bpl
	cmpq	%rsi, %r15
	seta	%r14b
	testb	%bpl, %r14b
	jne	.LBB28_37
# %bb.24:
	leaq	491520(%rdx), %r14
	cmpq	%rbx, %r14
	seta	%bpl
	cmpq	%rdx, %r15
	seta	%r14b
	testb	%bpl, %r14b
	jne	.LBB28_37
# %bb.25:
	leaq	192(%rcx), %r14
	cmpq	%rbx, %r14
	seta	%bpl
	cmpq	%rcx, %r15
	seta	%r14b
	testb	%bpl, %r14b
	jne	.LBB28_37
# %bb.26:
	cmpq	%rbx, %r12
	seta	%bl
	cmpq	%rax, %r15
	seta	%bpl
	testb	%bl, %bpl
	jne	.LBB28_37
# %bb.27:
	leaq	10240(%r8), %rbx
	cmpq	%r10, %rbx
	seta	%bl
	cmpq	%r8, %r11
	seta	%bpl
	testb	%bl, %bpl
	jne	.LBB28_37
# %bb.28:
	leaq	491520(%rsi), %rbx
	cmpq	%r10, %rbx
	seta	%bl
	cmpq	%rsi, %r11
	seta	%bpl
	testb	%bl, %bpl
	jne	.LBB28_37
# %bb.29:
	leaq	491520(%rdx), %rbx
	cmpq	%r10, %rbx
	seta	%bl
	cmpq	%rdx, %r11
	seta	%bpl
	testb	%bl, %bpl
	jne	.LBB28_37
# %bb.30:
	leaq	192(%rcx), %rbx
	cmpq	%r10, %rbx
	seta	%bl
	cmpq	%rcx, %r11
	seta	%bpl
	testb	%bl, %bpl
	jne	.LBB28_37
# %bb.31:
	cmpq	%r10, %r12
	seta	%r10b
	cmpq	%rax, %r11
	seta	%r11b
	testb	%r10b, %r11b
	jne	.LBB28_37
# %bb.32:
	leaq	10240(%r8), %r10
	cmpq	%rdi, %r10
	seta	%r10b
	cmpq	%r8, %r9
	seta	%r8b
	testb	%r10b, %r8b
	jne	.LBB28_37
# %bb.33:
	leaq	491520(%rsi), %r8
	cmpq	%rdi, %r8
	seta	%r8b
	cmpq	%rsi, %r9
	seta	%sil
	testb	%r8b, %sil
	jne	.LBB28_37
# %bb.34:
	leaq	491520(%rdx), %rsi
	cmpq	%rdi, %rsi
	seta	%sil
	cmpq	%rdx, %r9
	seta	%dl
	testb	%sil, %dl
	jne	.LBB28_37
# %bb.35:
	leaq	192(%rcx), %rdx
	cmpq	%rdi, %rdx
	seta	%dl
	cmpq	%rcx, %r9
	seta	%cl
	testb	%dl, %cl
	jne	.LBB28_37
# %bb.36:
	cmpq	%rdi, %r12
	setbe	%cl
	cmpq	%rax, %r9
	setbe	%al
	orb	%cl, %al
	movb	$1, %al
	jne	.LBB28_38
.LBB28_37:
	xorl	%eax, %eax
.LBB28_38:
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
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r13
	.cfi_restore %r14
	.cfi_restore %r15
	.cfi_restore %rbp
                                        # kill: def $al killed $al killed $eax
	retq
.LBB28_1:
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	retq
.Lfunc_end28:
	.size	_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE, .Lfunc_end28-_ZN12_GLOBAL__N_15validERKNS_9ArgumentsE
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end29, nop    # -- Begin function _ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.type	_ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_,@function
_ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_: # @_ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
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
	leaq	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end29:
	.size	_ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_, .Lfunc_end29-_ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end30, nop    # -- Begin function _ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.type	_ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_,@function
_ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_: # @_ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
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
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end30:
	.size	_ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_, .Lfunc_end30-_ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end31, nop    # -- Begin function _ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.type	_ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_,@function
_ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_: # @_ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.cfi_startproc
# %bb.0:
	subq	$136, %rsp
	.cfi_def_cfa_offset 144
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
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
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$152, %rsp
	.cfi_adjust_cfa_offset -152
	retq
.Lfunc_end31:
	.size	_ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_, .Lfunc_end31-_ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end32, nop    # -- Begin function _ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.type	_ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_,@function
_ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_: # @_ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
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
	leaq	192(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end32:
	.size	_ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_, .Lfunc_end32-_ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end33, nop    # -- Begin function _ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
	.type	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_,@function
_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_: # @_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
.Lfunc_begin15:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception15
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
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	16(%rsi), %rbp
	leaq	2(%rbp), %rbx
	movq	%rbx, %rax
	shrq	$62, %rax
	jne	.LBB33_29
# %bb.1:
	movq	%rsi, %r13
	movq	%rdi, %r15
	testq	%rbx, %rbx
	je	.LBB33_2
# %bb.3:
	leaq	(%rbx,%rbx), %rdi
	callq	_Znwm@PLT
	movq	%rax, %r14
	leaq	(%rax,%rbx,2), %rbx
	movw	$0, (%rax)
	movq	%rax, %r12
	addq	$2, %r12
	incq	%rbp
	je	.LBB33_5
# %bb.4:
	leaq	(,%rbp,2), %rdx
	movq	%r12, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	(%r12,%rbp,2), %r12
	jmp	.LBB33_5
.LBB33_2:
	xorl	%ebx, %ebx
	xorl	%r14d, %r14d
	xorl	%r12d, %r12d
.LBB33_5:
	movq	(%r13), %rsi
	movq	8(%r13), %rdx
	addq	%rdx, %rdx
.Ltmp1180:                              # EH_LABEL
	movq	%r14, %rdi
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp1181:                              # EH_LABEL
# %bb.6:
.Ltmp1182:                              # EH_LABEL
	leaq	.L.str.44(%rip), %rsi
	movl	%eax, %edi
	movl	$303, %edx                      # imm = 0x12F
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp1183:                              # EH_LABEL
# %bb.7:
	cmpw	$32602, (%r14)                  # imm = 0x7F5A
	jne	.LBB33_9
# %bb.8:
	cmpw	$32602, -2(%r12)                # imm = 0x7F5A
	jne	.LBB33_9
# %bb.15:
	addq	$-2, %r12
	leaq	2(%r14), %r13
	xorps	%xmm0, %xmm0
	movups	%xmm0, (%r15)
	movq	$0, 16(%r15)
	movq	%r12, %rbp
	subq	%r13, %rbp
	movabsq	$9223372036854775807, %rax      # imm = 0x7FFFFFFFFFFFFFFF
	cmpq	%rax, %rbp
	jae	.LBB33_16
# %bb.18:
	cmpq	%r13, %r12
	je	.LBB33_19
# %bb.20:
.Ltmp1191:                              # EH_LABEL
	movq	%rbp, %rdi
	callq	_Znwm@PLT
.Ltmp1192:                              # EH_LABEL
# %bb.21:
	movq	%rax, (%r15)
	movq	%rax, %r12
	addq	%rbp, %r12
	movq	%r12, 16(%r15)
	cmpq	$3, %rbp
	jb	.LBB33_23
# %bb.22:
	movq	%rax, %rdi
	movq	%r13, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
	jmp	.LBB33_25
.LBB33_19:
	movq	%rbp, 16(%r15)
	movq	%rbp, %r12
.LBB33_25:
	movq	%r12, 8(%r15)
	subq	%r14, %rbx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	addq	$40, %rsp
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
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB33_9:
	.cfi_def_cfa_offset 96
.Ltmp1185:                              # EH_LABEL
	leaq	.L.str.45(%rip), %rsi
	leaq	8(%rsp), %rdi
	leaq	7(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1186:                              # EH_LABEL
# %bb.10:
.Ltmp1188:                              # EH_LABEL
	leaq	8(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1189:                              # EH_LABEL
# %bb.11:
.LBB33_29:
	leaq	.L.str.40(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB33_16:
.Ltmp1193:                              # EH_LABEL
	leaq	.L.str.40(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1194:                              # EH_LABEL
# %bb.17:
.LBB33_23:
	cmpq	$2, %rbp
	jne	.LBB33_25
# %bb.24:
	movzwl	(%r13), %ecx
	movw	%cx, (%rax)
	jmp	.LBB33_25
.LBB33_13:
.Ltmp1190:                              # EH_LABEL
	movq	%rax, %r15
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB33_27
# %bb.14:
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB33_27
.LBB33_12:
.Ltmp1187:                              # EH_LABEL
	movq	%rax, %r15
	jmp	.LBB33_27
.LBB33_30:
.Ltmp1195:                              # EH_LABEL
	movq	%rax, %r15
	jmp	.LBB33_27
.LBB33_26:
.Ltmp1184:                              # EH_LABEL
	movq	%rax, %r15
	testq	%r14, %r14
	je	.LBB33_28
.LBB33_27:
	subq	%r14, %rbx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
.LBB33_28:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end33:
	.size	_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_, .Lfunc_end33-_ZN12_GLOBAL__N_114download_guardINS_11GuardedBf16EEEDaRKT_
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table33:
.Lexception15:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end15-.Lcst_begin15
.Lcst_begin15:
	.uleb128 .Lfunc_begin15-.Lfunc_begin15  # >> Call Site 1 <<
	.uleb128 .Ltmp1180-.Lfunc_begin15       #   Call between .Lfunc_begin15 and .Ltmp1180
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1180-.Lfunc_begin15       # >> Call Site 2 <<
	.uleb128 .Ltmp1183-.Ltmp1180            #   Call between .Ltmp1180 and .Ltmp1183
	.uleb128 .Ltmp1184-.Lfunc_begin15       #     jumps to .Ltmp1184
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1191-.Lfunc_begin15       # >> Call Site 3 <<
	.uleb128 .Ltmp1192-.Ltmp1191            #   Call between .Ltmp1191 and .Ltmp1192
	.uleb128 .Ltmp1195-.Lfunc_begin15       #     jumps to .Ltmp1195
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1192-.Lfunc_begin15       # >> Call Site 4 <<
	.uleb128 .Ltmp1185-.Ltmp1192            #   Call between .Ltmp1192 and .Ltmp1185
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1185-.Lfunc_begin15       # >> Call Site 5 <<
	.uleb128 .Ltmp1186-.Ltmp1185            #   Call between .Ltmp1185 and .Ltmp1186
	.uleb128 .Ltmp1187-.Lfunc_begin15       #     jumps to .Ltmp1187
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1188-.Lfunc_begin15       # >> Call Site 6 <<
	.uleb128 .Ltmp1189-.Ltmp1188            #   Call between .Ltmp1188 and .Ltmp1189
	.uleb128 .Ltmp1190-.Lfunc_begin15       #     jumps to .Ltmp1190
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1189-.Lfunc_begin15       # >> Call Site 7 <<
	.uleb128 .Ltmp1193-.Ltmp1189            #   Call between .Ltmp1189 and .Ltmp1193
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1193-.Lfunc_begin15       # >> Call Site 8 <<
	.uleb128 .Ltmp1194-.Ltmp1193            #   Call between .Ltmp1193 and .Ltmp1194
	.uleb128 .Ltmp1195-.Lfunc_begin15       #     jumps to .Ltmp1195
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1194-.Lfunc_begin15       # >> Call Site 9 <<
	.uleb128 .Lfunc_end33-.Ltmp1194         #   Call between .Ltmp1194 and .Lfunc_end33
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end15:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.LCPI34_0:
	.long	0xc7f12060                      # float -123456.75
	.text
	.prefalign	4, .Lfunc_end34, nop
	.type	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_,@function
_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_: # @_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
.Lfunc_begin16:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception16
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
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	16(%rsi), %rbp
	leaq	2(%rbp), %rbx
	movq	%rbx, %rax
	shrq	$61, %rax
	jne	.LBB34_29
# %bb.1:
	movq	%rsi, %r13
	movq	%rdi, %r15
	testq	%rbx, %rbx
	je	.LBB34_2
# %bb.3:
	leaq	(,%rbx,4), %rdi
	callq	_Znwm@PLT
	movq	%rax, %r14
	leaq	(%rax,%rbx,4), %rbx
	movl	$0, (%rax)
	movq	%rax, %r12
	addq	$4, %r12
	incq	%rbp
	je	.LBB34_5
# %bb.4:
	leaq	(,%rbp,4), %rdx
	movq	%r12, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	(%r12,%rbp,4), %r12
	jmp	.LBB34_5
.LBB34_2:
	xorl	%ebx, %ebx
	xorl	%r14d, %r14d
	xorl	%r12d, %r12d
.LBB34_5:
	movq	(%r13), %rsi
	movq	8(%r13), %rdx
	shlq	$2, %rdx
.Ltmp1196:                              # EH_LABEL
	movq	%r14, %rdi
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp1197:                              # EH_LABEL
# %bb.6:
.Ltmp1198:                              # EH_LABEL
	leaq	.L.str.44(%rip), %rsi
	movl	%eax, %edi
	movl	$303, %edx                      # imm = 0x12F
	callq	_ZN12_GLOBAL__N_19hip_checkE10hipError_tPKcS2_i
.Ltmp1199:                              # EH_LABEL
# %bb.7:
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	.LCPI34_0(%rip), %xmm0
	jne	.LBB34_9
	jp	.LBB34_9
# %bb.8:
	movss	-4(%r12), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	ucomiss	.LCPI34_0(%rip), %xmm0
	jne	.LBB34_9
	jp	.LBB34_9
# %bb.15:
	addq	$-4, %r12
	leaq	4(%r14), %r13
	xorps	%xmm0, %xmm0
	movups	%xmm0, (%r15)
	movq	$0, 16(%r15)
	movq	%r12, %rbp
	subq	%r13, %rbp
	movabsq	$9223372036854775805, %rax      # imm = 0x7FFFFFFFFFFFFFFD
	cmpq	%rax, %rbp
	jae	.LBB34_16
# %bb.18:
	cmpq	%r13, %r12
	je	.LBB34_19
# %bb.20:
.Ltmp1201:                              # EH_LABEL
	movq	%rbp, %rdi
	callq	_Znwm@PLT
.Ltmp1202:                              # EH_LABEL
# %bb.21:
	movq	%rax, (%r15)
	movq	%rax, %r12
	addq	%rbp, %r12
	movq	%r12, 16(%r15)
	cmpq	$5, %rbp
	jb	.LBB34_23
# %bb.22:
	movq	%rax, %rdi
	movq	%r13, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
	jmp	.LBB34_25
.LBB34_19:
	movq	%rbp, 16(%r15)
	movq	%rbp, %r12
.LBB34_25:
	movq	%r12, 8(%r15)
	subq	%r14, %rbx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	addq	$40, %rsp
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
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB34_9:
	.cfi_def_cfa_offset 96
.Ltmp1206:                              # EH_LABEL
	leaq	.L.str.46(%rip), %rsi
	leaq	8(%rsp), %rdi
	leaq	7(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1207:                              # EH_LABEL
# %bb.10:
.Ltmp1209:                              # EH_LABEL
	leaq	8(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1210:                              # EH_LABEL
# %bb.11:
.LBB34_29:
	leaq	.L.str.40(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB34_16:
.Ltmp1203:                              # EH_LABEL
	leaq	.L.str.40(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1204:                              # EH_LABEL
# %bb.17:
.LBB34_23:
	cmpq	$4, %rbp
	jne	.LBB34_25
# %bb.24:
	movss	(%r13), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax)
	jmp	.LBB34_25
.LBB34_13:
.Ltmp1211:                              # EH_LABEL
	movq	%rax, %r15
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB34_27
# %bb.14:
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB34_27
.LBB34_12:
.Ltmp1208:                              # EH_LABEL
	movq	%rax, %r15
	jmp	.LBB34_27
.LBB34_30:
.Ltmp1205:                              # EH_LABEL
	movq	%rax, %r15
	jmp	.LBB34_27
.LBB34_26:
.Ltmp1200:                              # EH_LABEL
	movq	%rax, %r15
	testq	%r14, %r14
	je	.LBB34_28
.LBB34_27:
	subq	%r14, %rbx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
.LBB34_28:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end34:
	.size	_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_, .Lfunc_end34-_ZN12_GLOBAL__N_114download_guardINS_12GuardedFloatEEEDaRKT_
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table34:
.Lexception16:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end16-.Lcst_begin16
.Lcst_begin16:
	.uleb128 .Lfunc_begin16-.Lfunc_begin16  # >> Call Site 1 <<
	.uleb128 .Ltmp1196-.Lfunc_begin16       #   Call between .Lfunc_begin16 and .Ltmp1196
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1196-.Lfunc_begin16       # >> Call Site 2 <<
	.uleb128 .Ltmp1199-.Ltmp1196            #   Call between .Ltmp1196 and .Ltmp1199
	.uleb128 .Ltmp1200-.Lfunc_begin16       #     jumps to .Ltmp1200
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1201-.Lfunc_begin16       # >> Call Site 3 <<
	.uleb128 .Ltmp1202-.Ltmp1201            #   Call between .Ltmp1201 and .Ltmp1202
	.uleb128 .Ltmp1205-.Lfunc_begin16       #     jumps to .Ltmp1205
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1202-.Lfunc_begin16       # >> Call Site 4 <<
	.uleb128 .Ltmp1206-.Ltmp1202            #   Call between .Ltmp1202 and .Ltmp1206
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1206-.Lfunc_begin16       # >> Call Site 5 <<
	.uleb128 .Ltmp1207-.Ltmp1206            #   Call between .Ltmp1206 and .Ltmp1207
	.uleb128 .Ltmp1208-.Lfunc_begin16       #     jumps to .Ltmp1208
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1209-.Lfunc_begin16       # >> Call Site 6 <<
	.uleb128 .Ltmp1210-.Ltmp1209            #   Call between .Ltmp1209 and .Ltmp1210
	.uleb128 .Ltmp1211-.Lfunc_begin16       #     jumps to .Ltmp1211
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1210-.Lfunc_begin16       # >> Call Site 7 <<
	.uleb128 .Ltmp1203-.Ltmp1210            #   Call between .Ltmp1210 and .Ltmp1203
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1203-.Lfunc_begin16       # >> Call Site 8 <<
	.uleb128 .Ltmp1204-.Ltmp1203            #   Call between .Ltmp1203 and .Ltmp1204
	.uleb128 .Ltmp1205-.Lfunc_begin16       #     jumps to .Ltmp1205
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1204-.Lfunc_begin16       # >> Call Site 9 <<
	.uleb128 .Lfunc_end34-.Ltmp1204         #   Call between .Ltmp1204 and .Lfunc_end34
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end16:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev,"axG",@progbits,_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev,comdat
	.weak	_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev # -- Begin function _ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev
	.p2align	1
	.prefalign	4, .Lfunc_end35, nop
	.type	_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev,@function
_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev: # @_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	movq	48(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB35_2
# %bb.1:
	movq	64(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB35_2:
	movq	24(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB35_4
# %bb.3:
	movq	40(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB35_4:
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB35_5
# %bb.6:
	movq	16(%rbx), %rsi
	subq	%rdi, %rsi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB35_5:
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end35:
	.size	_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev, .Lfunc_end35-_ZNSt5arrayISt6vectorI12hip_bfloat16SaIS1_EELm3EED2Ev
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end36, nop    # -- Begin function _ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.type	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,@function
_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE: # @_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Lfunc_begin17:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception17
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
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 16(%rsp)                  # 8-byte Spill
	leaq	16(%rdi), %rbp
	movq	%rbp, (%rdi)
	movq	$0, 8(%rdi)
	movb	$0, 16(%rdi)
	movq	%rdx, 8(%rsp)                   # 8-byte Spill
	testq	%rdx, %rdx
	je	.LBB36_34
# %bb.1:
	movq	%rdi, %rbx
	xorl	%r12d, %r12d
	jmp	.LBB36_2
	.p2align	4
.LBB36_9:                               #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbx), %rax
.LBB36_10:                              #   in Loop: Header=BB36_2 Depth=1
	movb	%r15b, (%rax,%r13)
.LBB36_32:                              #   in Loop: Header=BB36_2 Depth=1
	movq	%r14, 8(%rbx)
	movq	(%rbx), %rax
	movb	$0, (%rax,%r14)
.LBB36_33:                              #   in Loop: Header=BB36_2 Depth=1
	incq	%r12
	cmpq	%r12, 8(%rsp)                   # 8-byte Folded Reload
	je	.LBB36_34
.LBB36_2:                               # =>This Inner Loop Header: Depth=1
	movq	16(%rsp), %rax                  # 8-byte Reload
	movzbl	(%rax,%r12), %r15d
	movzbl	%r15b, %eax
	cmpl	$33, %eax
	jg	.LBB36_11
# %bb.3:                                #   in Loop: Header=BB36_2 Depth=1
	cmpl	$10, %eax
	je	.LBB36_24
# %bb.4:                                #   in Loop: Header=BB36_2 Depth=1
	cmpl	$13, %eax
	je	.LBB36_33
	jmp	.LBB36_5
	.p2align	4
.LBB36_11:                              #   in Loop: Header=BB36_2 Depth=1
	cmpl	$92, %eax
	je	.LBB36_13
# %bb.12:                               #   in Loop: Header=BB36_2 Depth=1
	cmpl	$34, %eax
	jne	.LBB36_5
.LBB36_13:                              #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbx), %rax
	movl	$15, %ecx
	cmpq	%rbp, %rax
	je	.LBB36_15
# %bb.14:                               #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbp), %rcx
.LBB36_15:                              #   in Loop: Header=BB36_2 Depth=1
	movq	8(%rbx), %r13
	leaq	1(%r13), %r14
	cmpq	%rcx, %r14
	jbe	.LBB36_18
# %bb.16:                               #   in Loop: Header=BB36_2 Depth=1
.Ltmp1217:                              # EH_LABEL
	movl	$1, %r8d
	movq	%rbx, %rdi
	movq	%r13, %rsi
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp1218:                              # EH_LABEL
# %bb.17:                               #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbx), %rax
.LBB36_18:                              #   in Loop: Header=BB36_2 Depth=1
	movb	$92, (%rax,%r13)
	movq	%r14, 8(%rbx)
	movq	(%rbx), %rax
	movb	$0, 1(%rax,%r13)
.LBB36_5:                               #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbx), %rax
	movq	8(%rbx), %r13
	movl	$15, %ecx
	cmpq	%rbp, %rax
	je	.LBB36_7
# %bb.6:                                #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbp), %rcx
.LBB36_7:                               #   in Loop: Header=BB36_2 Depth=1
	leaq	1(%r13), %r14
	cmpq	%rcx, %r14
	jbe	.LBB36_10
# %bb.8:                                #   in Loop: Header=BB36_2 Depth=1
.Ltmp1219:                              # EH_LABEL
	movl	$1, %r8d
	movq	%rbx, %rdi
	movq	%r13, %rsi
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp1220:                              # EH_LABEL
	jmp	.LBB36_9
.LBB36_24:                              #   in Loop: Header=BB36_2 Depth=1
	movq	8(%rbx), %rsi
	movq	%rsi, %rax
	shrq	%rax
	movabsq	$4611686018427387903, %rcx      # imm = 0x3FFFFFFFFFFFFFFF
	cmpq	%rcx, %rax
	je	.LBB36_25
# %bb.27:                               #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbx), %rax
	movl	$15, %ecx
	cmpq	%rbp, %rax
	je	.LBB36_29
# %bb.28:                               #   in Loop: Header=BB36_2 Depth=1
	movq	(%rbp), %rcx
.LBB36_29:                              #   in Loop: Header=BB36_2 Depth=1
	leaq	2(%rsi), %r14
	cmpq	%rcx, %r14
	jbe	.LBB36_30
# %bb.31:                               #   in Loop: Header=BB36_2 Depth=1
.Ltmp1212:                              # EH_LABEL
	movl	$2, %r8d
	movq	%rbx, %rdi
	xorl	%edx, %edx
	leaq	.L.str.125(%rip), %rcx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp1213:                              # EH_LABEL
	jmp	.LBB36_32
.LBB36_30:                              #   in Loop: Header=BB36_2 Depth=1
	movw	$28252, (%rax,%rsi)             # imm = 0x6E5C
	jmp	.LBB36_32
.LBB36_34:
	addq	$24, %rsp
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
.LBB36_25:
	.cfi_def_cfa_offset 80
.Ltmp1214:                              # EH_LABEL
	leaq	.L.str.39(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1215:                              # EH_LABEL
# %bb.26:
.LBB36_20:
.Ltmp1216:                              # EH_LABEL
	jmp	.LBB36_21
.LBB36_19:
.Ltmp1221:                              # EH_LABEL
.LBB36_21:
	movq	%rax, %r14
	movq	(%rbx), %rdi
	cmpq	%rbp, %rdi
	je	.LBB36_23
# %bb.22:
	movq	(%rbp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB36_23:
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end36:
	.size	_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .Lfunc_end36-_ZN12_GLOBAL__N_111json_escapeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table36:
.Lexception17:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end17-.Lcst_begin17
.Lcst_begin17:
	.uleb128 .Ltmp1217-.Lfunc_begin17       # >> Call Site 1 <<
	.uleb128 .Ltmp1213-.Ltmp1217            #   Call between .Ltmp1217 and .Ltmp1213
	.uleb128 .Ltmp1221-.Lfunc_begin17       #     jumps to .Ltmp1221
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1214-.Lfunc_begin17       # >> Call Site 2 <<
	.uleb128 .Ltmp1215-.Ltmp1214            #   Call between .Ltmp1214 and .Ltmp1215
	.uleb128 .Ltmp1216-.Lfunc_begin17       #     jumps to .Ltmp1216
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1215-.Lfunc_begin17       # >> Call Site 3 <<
	.uleb128 .Lfunc_end36-.Ltmp1215         #   Call between .Ltmp1215 and .Lfunc_end36
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end17:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end37, nop    # -- Begin function _ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.type	_ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,@function
_ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE: # @_ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Lfunc_begin18:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception18
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
	subq	$4136, %rsp                     # imm = 0x1028
	.cfi_def_cfa_offset 4192
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rax
	movq	%rdi, %rbx
	leaq	.L.str.126(%rip), %rsi
	movq	%rax, %rdi
	callq	popen@PLT
	testq	%rax, %rax
	je	.LBB37_1
# %bb.7:
	movq	%rax, %r14
	leaq	40(%rsp), %r15
	movl	$4096, %edx                     # imm = 0x1000
	movq	%r15, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	16(%rbx), %r12
	movq	%r12, (%rbx)
	movq	$0, 8(%rbx)
	movb	$0, 16(%rbx)
	movq	%r15, %rdi
	movl	$4096, %esi                     # imm = 0x1000
	movq	%r14, %rdx
	callq	fgets@PLT
	testq	%rax, %rax
	je	.LBB37_21
# %bb.8:
	leaq	40(%rsp), %r15
	movabsq	$9223372036854775807, %r13      # imm = 0x7FFFFFFFFFFFFFFF
	jmp	.LBB37_9
	.p2align	4
.LBB37_19:                              #   in Loop: Header=BB37_9 Depth=1
.Ltmp1222:                              # EH_LABEL
	movq	%rbx, %rdi
	xorl	%edx, %edx
	movq	%r15, %rcx
	movq	%rax, %r8
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp1223:                              # EH_LABEL
.LBB37_20:                              #   in Loop: Header=BB37_9 Depth=1
	movq	%rbp, 8(%rbx)
	movq	(%rbx), %rax
	movb	$0, (%rax,%rbp)
	movq	%r15, %rdi
	movl	$4096, %esi                     # imm = 0x1000
	movq	%r14, %rdx
	callq	fgets@PLT
	testq	%rax, %rax
	je	.LBB37_21
.LBB37_9:                               # =>This Inner Loop Header: Depth=1
	movq	%r15, %rdi
	callq	strlen@PLT
	movq	8(%rbx), %rsi
	movq	%rsi, %rcx
	xorq	%r13, %rcx
	cmpq	%rax, %rcx
	jb	.LBB37_10
# %bb.12:                               #   in Loop: Header=BB37_9 Depth=1
	movq	(%rbx), %rdi
	movl	$15, %ecx
	cmpq	%r12, %rdi
	je	.LBB37_14
# %bb.13:                               #   in Loop: Header=BB37_9 Depth=1
	movq	(%r12), %rcx
.LBB37_14:                              #   in Loop: Header=BB37_9 Depth=1
	leaq	(%rsi,%rax), %rbp
	cmpq	%rcx, %rbp
	ja	.LBB37_19
# %bb.15:                               #   in Loop: Header=BB37_9 Depth=1
	testq	%rax, %rax
	je	.LBB37_20
# %bb.16:                               #   in Loop: Header=BB37_9 Depth=1
	addq	%rsi, %rdi
	cmpq	$1, %rax
	jne	.LBB37_18
# %bb.17:                               #   in Loop: Header=BB37_9 Depth=1
	movzbl	40(%rsp), %eax
	movb	%al, (%rdi)
	jmp	.LBB37_20
.LBB37_18:                              #   in Loop: Header=BB37_9 Depth=1
	movq	%r15, %rsi
	movq	%rax, %rdx
	callq	memcpy@PLT
	jmp	.LBB37_20
.LBB37_21:
	movq	%r14, %rdi
	callq	pclose@PLT
	testl	%eax, %eax
	je	.LBB37_22
# %bb.27:
.Ltmp1228:                              # EH_LABEL
	leaq	.L.str.128(%rip), %rsi
	leaq	8(%rsp), %rdi
	leaq	7(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1229:                              # EH_LABEL
# %bb.28:
.Ltmp1231:                              # EH_LABEL
	leaq	8(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1232:                              # EH_LABEL
# %bb.29:
	.p2align	4
.LBB37_25:                              #   in Loop: Header=BB37_22 Depth=1
	leaq	-1(%rax), %rdx
	movq	%rdx, 8(%rbx)
	movb	$0, -1(%rcx,%rax)
.LBB37_22:                              # =>This Inner Loop Header: Depth=1
	movq	8(%rbx), %rax
	testq	%rax, %rax
	je	.LBB37_26
# %bb.23:                               #   in Loop: Header=BB37_22 Depth=1
	movq	(%rbx), %rcx
	movzbl	-1(%rcx,%rax), %edx
	cmpl	$13, %edx
	je	.LBB37_25
# %bb.24:                               #   in Loop: Header=BB37_22 Depth=1
	cmpl	$10, %edx
	je	.LBB37_25
.LBB37_26:
	addq	$4136, %rsp                     # imm = 0x1028
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
.LBB37_10:
	.cfi_def_cfa_offset 4192
.Ltmp1225:                              # EH_LABEL
	leaq	.L.str.39(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1226:                              # EH_LABEL
# %bb.11:
.LBB37_1:
.Ltmp1234:                              # EH_LABEL
	leaq	.L.str.127(%rip), %rsi
	leaq	40(%rsp), %rdi
	leaq	8(%rsp), %rdx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp1235:                              # EH_LABEL
# %bb.2:
.Ltmp1237:                              # EH_LABEL
	leaq	40(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1238:                              # EH_LABEL
# %bb.3:
.LBB37_31:
.Ltmp1233:                              # EH_LABEL
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB37_35
# %bb.32:
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB37_35
.LBB37_30:
.Ltmp1230:                              # EH_LABEL
	jmp	.LBB37_34
.LBB37_5:
.Ltmp1239:                              # EH_LABEL
	movq	%rax, %r14
	movq	40(%rsp), %rdi
	leaq	56(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB37_38
# %bb.6:
	movq	56(%rsp), %rsi
	jmp	.LBB37_37
.LBB37_4:
.Ltmp1236:                              # EH_LABEL
	movq	%rax, %rdi
	callq	_Unwind_Resume@PLT
.LBB37_33:
.Ltmp1224:                              # EH_LABEL
	jmp	.LBB37_34
.LBB37_39:
.Ltmp1227:                              # EH_LABEL
.LBB37_34:
	movq	%rax, %r14
.LBB37_35:
	movq	(%rbx), %rdi
	cmpq	%r12, %rdi
	je	.LBB37_38
# %bb.36:
	movq	(%r12), %rsi
.LBB37_37:
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB37_38:
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end37:
	.size	_ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .Lfunc_end37-_ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table37:
.Lexception18:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end18-.Lcst_begin18
.Lcst_begin18:
	.uleb128 .Lfunc_begin18-.Lfunc_begin18  # >> Call Site 1 <<
	.uleb128 .Ltmp1222-.Lfunc_begin18       #   Call between .Lfunc_begin18 and .Ltmp1222
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1222-.Lfunc_begin18       # >> Call Site 2 <<
	.uleb128 .Ltmp1223-.Ltmp1222            #   Call between .Ltmp1222 and .Ltmp1223
	.uleb128 .Ltmp1224-.Lfunc_begin18       #     jumps to .Ltmp1224
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1223-.Lfunc_begin18       # >> Call Site 3 <<
	.uleb128 .Ltmp1228-.Ltmp1223            #   Call between .Ltmp1223 and .Ltmp1228
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1228-.Lfunc_begin18       # >> Call Site 4 <<
	.uleb128 .Ltmp1229-.Ltmp1228            #   Call between .Ltmp1228 and .Ltmp1229
	.uleb128 .Ltmp1230-.Lfunc_begin18       #     jumps to .Ltmp1230
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1231-.Lfunc_begin18       # >> Call Site 5 <<
	.uleb128 .Ltmp1232-.Ltmp1231            #   Call between .Ltmp1231 and .Ltmp1232
	.uleb128 .Ltmp1233-.Lfunc_begin18       #     jumps to .Ltmp1233
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1225-.Lfunc_begin18       # >> Call Site 6 <<
	.uleb128 .Ltmp1226-.Ltmp1225            #   Call between .Ltmp1225 and .Ltmp1226
	.uleb128 .Ltmp1227-.Lfunc_begin18       #     jumps to .Ltmp1227
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1234-.Lfunc_begin18       # >> Call Site 7 <<
	.uleb128 .Ltmp1235-.Ltmp1234            #   Call between .Ltmp1234 and .Ltmp1235
	.uleb128 .Ltmp1236-.Lfunc_begin18       #     jumps to .Ltmp1236
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1237-.Lfunc_begin18       # >> Call Site 8 <<
	.uleb128 .Ltmp1238-.Ltmp1237            #   Call between .Ltmp1237 and .Ltmp1238
	.uleb128 .Ltmp1239-.Lfunc_begin18       #     jumps to .Ltmp1239
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1238-.Lfunc_begin18       # >> Call Site 9 <<
	.uleb128 .Lfunc_end37-.Ltmp1238         #   Call between .Ltmp1238 and .Lfunc_end37
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end18:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end38, nop    # -- Begin function _ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.type	_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,@function
_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE: # @_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Lfunc_begin19:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception19
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
	subq	$184, %rsp
	.cfi_def_cfa_offset 240
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %r12
	movq	(%rsi), %rbp
	movq	%rsi, 104(%rsp)                 # 8-byte Spill
	movq	8(%rsi), %r14
	leaq	24(%rsp), %r13
	movq	%r13, 8(%rsp)
	movw	$39, 24(%rsp)
	movq	$1, 16(%rsp)
	testq	%r14, %r14
	je	.LBB38_17
# %bb.1:
	movq	%r12, 112(%rsp)                 # 8-byte Spill
	leaq	56(%rsp), %r15
	xorl	%ebx, %ebx
	jmp	.LBB38_3
	.p2align	4
.LBB38_2:                               #   in Loop: Header=BB38_3 Depth=1
	incq	%rbx
	cmpq	%rbx, %r14
	movq	%r12, %r13
	je	.LBB38_16
.LBB38_3:                               # =>This Inner Loop Header: Depth=1
	movzbl	(%rbp,%rbx), %eax
	movq	%r15, 40(%rsp)
	cmpb	$39, %al
	jne	.LBB38_5
# %bb.4:                                #   in Loop: Header=BB38_3 Depth=1
	movl	$656890919, 56(%rsp)            # imm = 0x27275C27
	movb	$0, 60(%rsp)
	movl	$4, %r8d
	jmp	.LBB38_6
	.p2align	4
.LBB38_5:                               #   in Loop: Header=BB38_3 Depth=1
	movb	%al, 56(%rsp)
	movb	$0, 57(%rsp)
	movl	$1, %r8d
.LBB38_6:                               #   in Loop: Header=BB38_3 Depth=1
	movq	%r8, 48(%rsp)
	movq	16(%rsp), %rsi
	movq	%rsi, %rcx
	movabsq	$9223372036854775807, %rdx      # imm = 0x7FFFFFFFFFFFFFFF
	xorq	%rdx, %rcx
	cmpq	%r8, %rcx
	jb	.LBB38_34
# %bb.7:                                #   in Loop: Header=BB38_3 Depth=1
	movq	8(%rsp), %rdi
	movl	$15, %ecx
	movq	%r13, %r12
	cmpq	%r13, %rdi
	je	.LBB38_9
# %bb.8:                                #   in Loop: Header=BB38_3 Depth=1
	movq	24(%rsp), %rcx
.LBB38_9:                               #   in Loop: Header=BB38_3 Depth=1
	leaq	(%rsi,%r8), %r13
	cmpq	%rcx, %r13
	jbe	.LBB38_11
# %bb.10:                               #   in Loop: Header=BB38_3 Depth=1
.Ltmp1240:                              # EH_LABEL
	leaq	8(%rsp), %rdi
	xorl	%edx, %edx
	movq	%r15, %rcx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.Ltmp1241:                              # EH_LABEL
	jmp	.LBB38_14
	.p2align	4
.LBB38_11:                              #   in Loop: Header=BB38_3 Depth=1
	addq	%rsi, %rdi
	cmpb	$39, %al
	jne	.LBB38_13
# %bb.12:                               #   in Loop: Header=BB38_3 Depth=1
	movq	%r15, %rsi
	movq	%r8, %rdx
	callq	memcpy@PLT
	jmp	.LBB38_14
	.p2align	4
.LBB38_13:                              #   in Loop: Header=BB38_3 Depth=1
	movb	%al, (%rdi)
.LBB38_14:                              #   in Loop: Header=BB38_3 Depth=1
	movq	%r13, 16(%rsp)
	movq	8(%rsp), %rax
	movb	$0, (%rax,%r13)
	movq	40(%rsp), %rdi
	cmpq	%r15, %rdi
	je	.LBB38_2
# %bb.15:                               #   in Loop: Header=BB38_3 Depth=1
	movq	56(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB38_2
.LBB38_16:
	movq	8(%rsp), %rsi
	movq	16(%rsp), %rdx
	movq	112(%rsp), %r12                 # 8-byte Reload
	jmp	.LBB38_18
.LBB38_17:
	movl	$1, %edx
	movq	%r13, %rsi
.LBB38_18:
.Ltmp1246:                              # EH_LABEL
	leaq	.L.str.131(%rip), %rcx
	leaq	120(%rsp), %rdi
	leaq	40(%rsp), %r9
	movl	$1, %r8d
	callq	_ZSt12__str_concatINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEET_PKNS6_10value_typeENS6_9size_typeES9_SA_RKNS6_14allocator_typeE
.Ltmp1247:                              # EH_LABEL
# %bb.19:
	movq	8(%rsp), %rdi
	cmpq	%r13, %rdi
	je	.LBB38_21
# %bb.20:
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB38_21:
.Ltmp1249:                              # EH_LABEL
	leaq	.L.str.129(%rip), %rcx
	leaq	120(%rsp), %rdi
	movl	$22, %r8d
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm
.Ltmp1250:                              # EH_LABEL
# %bb.22:
	movq	%rax, %rbx
	leaq	88(%rsp), %r15
	movq	%r15, 72(%rsp)
	movq	(%rax), %rcx
	addq	$16, %rax
	cmpq	%rax, %rcx
	je	.LBB38_24
# %bb.23:
	movq	%rcx, 72(%rsp)
	movq	16(%rbx), %rcx
	movq	%rcx, 88(%rsp)
	movq	8(%rbx), %r14
	jmp	.LBB38_25
.LBB38_24:
	movq	8(%rbx), %r14
	leaq	1(%r14), %rdx
	movq	%r15, %rdi
	movq	%rax, %rsi
	movq	%r12, %r13
	movq	%rax, %r12
	callq	memcpy@PLT
	movq	%r12, %rax
	movq	%r13, %r12
.LBB38_25:
	movq	%r14, 80(%rsp)
	movq	%rax, (%rbx)
	movq	$0, 8(%rbx)
	movb	$0, 16(%rbx)
	movq	72(%rsp), %rsi
.Ltmp1252:                              # EH_LABEL
	leaq	152(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_114command_outputERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1253:                              # EH_LABEL
# %bb.26:
	movq	72(%rsp), %rdi
	cmpq	%r15, %rdi
	je	.LBB38_28
# %bb.27:
	movq	88(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB38_28:
	movq	120(%rsp), %rdi
	leaq	136(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB38_30
# %bb.29:
	movq	136(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB38_30:
	cmpq	$65, 160(%rsp)
	jb	.LBB38_36
# %bb.31:
	movq	152(%rsp), %r15
	cmpb	$32, 64(%r15)
	jne	.LBB38_36
# %bb.32:
	leaq	16(%r12), %rax
	movq	%rax, (%r12)
.Ltmp1255:                              # EH_LABEL
	movl	$65, %edi
	callq	_Znwm@PLT
.Ltmp1256:                              # EH_LABEL
# %bb.33:
	movq	%rax, (%r12)
	movq	$64, 16(%r12)
	movups	(%r15), %xmm0
	movups	16(%r15), %xmm1
	movups	32(%r15), %xmm2
	movups	48(%r15), %xmm3
	movups	%xmm3, 48(%rax)
	movups	%xmm2, 32(%rax)
	movups	%xmm1, 16(%rax)
	movups	%xmm0, (%rax)
	movq	$64, 8(%r12)
	movb	$0, 64(%rax)
	movq	168(%rsp), %rsi
	incq	%rsi
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
	addq	$184, %rsp
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
.LBB38_34:
	.cfi_def_cfa_offset 240
.Ltmp1243:                              # EH_LABEL
	movq	%r13, %r12
	leaq	.L.str.39(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1244:                              # EH_LABEL
# %bb.35:
.LBB38_36:
.Ltmp1258:                              # EH_LABEL
	leaq	.L.str.130(%rip), %rsi
	leaq	8(%rsp), %rdi
	movq	104(%rsp), %rdx                 # 8-byte Reload
	callq	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_
.Ltmp1259:                              # EH_LABEL
# %bb.37:
.Ltmp1261:                              # EH_LABEL
	leaq	8(%rsp), %rdi
	callq	_ZN12_GLOBAL__N_14failERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1262:                              # EH_LABEL
# %bb.38:
.LBB38_39:
.Ltmp1257:                              # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB38_45
.LBB38_40:
.Ltmp1263:                              # EH_LABEL
	movq	%rax, %rbx
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB38_44
# %bb.41:
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB38_44
.LBB38_43:
.Ltmp1260:                              # EH_LABEL
	movq	%rax, %rbx
.LBB38_44:
	movq	152(%rsp), %r15
.LBB38_45:
	leaq	168(%rsp), %rax
	cmpq	%rax, %r15
	je	.LBB38_61
# %bb.46:
	movq	168(%rsp), %rsi
	incq	%rsi
	movq	%r15, %rdi
	jmp	.LBB38_60
.LBB38_47:
.Ltmp1254:                              # EH_LABEL
	movq	%rax, %rbx
	movq	72(%rsp), %rdi
	cmpq	%r15, %rdi
	je	.LBB38_50
# %bb.48:
	movq	88(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB38_50
.LBB38_49:
.Ltmp1251:                              # EH_LABEL
	movq	%rax, %rbx
.LBB38_50:
	movq	120(%rsp), %rdi
	leaq	136(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB38_61
# %bb.51:
	movq	136(%rsp), %rsi
	jmp	.LBB38_59
.LBB38_52:
.Ltmp1248:                              # EH_LABEL
	movq	%r13, %r12
	movq	%rax, %rbx
	jmp	.LBB38_57
.LBB38_53:
.Ltmp1242:                              # EH_LABEL
	jmp	.LBB38_55
.LBB38_54:
.Ltmp1245:                              # EH_LABEL
.LBB38_55:
	movq	%rax, %rbx
	movq	40(%rsp), %rdi
	cmpq	%r15, %rdi
	je	.LBB38_57
# %bb.56:
	movq	56(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB38_57:
	movq	8(%rsp), %rdi
	cmpq	%r12, %rdi
	je	.LBB38_61
# %bb.58:
	movq	24(%rsp), %rsi
.LBB38_59:
	incq	%rsi
.LBB38_60:
	callq	_ZdlPvm@PLT
.LBB38_61:
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end38:
	.size	_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .Lfunc_end38-_ZN12_GLOBAL__N_16sha256ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table38:
.Lexception19:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end19-.Lcst_begin19
.Lcst_begin19:
	.uleb128 .Ltmp1240-.Lfunc_begin19       # >> Call Site 1 <<
	.uleb128 .Ltmp1241-.Ltmp1240            #   Call between .Ltmp1240 and .Ltmp1241
	.uleb128 .Ltmp1242-.Lfunc_begin19       #     jumps to .Ltmp1242
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1241-.Lfunc_begin19       # >> Call Site 2 <<
	.uleb128 .Ltmp1246-.Ltmp1241            #   Call between .Ltmp1241 and .Ltmp1246
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1246-.Lfunc_begin19       # >> Call Site 3 <<
	.uleb128 .Ltmp1247-.Ltmp1246            #   Call between .Ltmp1246 and .Ltmp1247
	.uleb128 .Ltmp1248-.Lfunc_begin19       #     jumps to .Ltmp1248
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1249-.Lfunc_begin19       # >> Call Site 4 <<
	.uleb128 .Ltmp1250-.Ltmp1249            #   Call between .Ltmp1249 and .Ltmp1250
	.uleb128 .Ltmp1251-.Lfunc_begin19       #     jumps to .Ltmp1251
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1250-.Lfunc_begin19       # >> Call Site 5 <<
	.uleb128 .Ltmp1252-.Ltmp1250            #   Call between .Ltmp1250 and .Ltmp1252
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1252-.Lfunc_begin19       # >> Call Site 6 <<
	.uleb128 .Ltmp1253-.Ltmp1252            #   Call between .Ltmp1252 and .Ltmp1253
	.uleb128 .Ltmp1254-.Lfunc_begin19       #     jumps to .Ltmp1254
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1255-.Lfunc_begin19       # >> Call Site 7 <<
	.uleb128 .Ltmp1256-.Ltmp1255            #   Call between .Ltmp1255 and .Ltmp1256
	.uleb128 .Ltmp1257-.Lfunc_begin19       #     jumps to .Ltmp1257
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1243-.Lfunc_begin19       # >> Call Site 8 <<
	.uleb128 .Ltmp1244-.Ltmp1243            #   Call between .Ltmp1243 and .Ltmp1244
	.uleb128 .Ltmp1245-.Lfunc_begin19       #     jumps to .Ltmp1245
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1258-.Lfunc_begin19       # >> Call Site 9 <<
	.uleb128 .Ltmp1259-.Ltmp1258            #   Call between .Ltmp1258 and .Ltmp1259
	.uleb128 .Ltmp1260-.Lfunc_begin19       #     jumps to .Ltmp1260
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1261-.Lfunc_begin19       # >> Call Site 10 <<
	.uleb128 .Ltmp1262-.Ltmp1261            #   Call between .Ltmp1261 and .Ltmp1262
	.uleb128 .Ltmp1263-.Lfunc_begin19       #     jumps to .Ltmp1263
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1262-.Lfunc_begin19       # >> Call Site 11 <<
	.uleb128 .Lfunc_end38-.Ltmp1262         #   Call between .Ltmp1262 and .Lfunc_end38
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end19:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ # -- Begin function _ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.prefalign	4, .Lfunc_end39, nop
	.type	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,@function
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_: # @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
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
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbp
	subq	%rdi, %rbp
	sarq	$2, %rbp
	cmpq	$17, %rbp
	jl	.LBB39_40
# %bb.1:
	movq	%rdx, %r14
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	je	.LBB39_8
# %bb.2:
	movq	$-4, %r13
	subq	%rbx, %r13
	.p2align	4
.LBB39_3:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB39_33 Depth 2
                                        #       Child Loop BB39_34 Depth 3
                                        #       Child Loop BB39_36 Depth 3
	shrq	%rbp
	movss	4(%rbx), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%rbx,%rbp,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm2
	movss	-4(%rsi), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	jbe	.LBB39_27
# %bb.4:                                #   in Loop: Header=BB39_3 Depth=1
	ucomiss	%xmm2, %xmm0
	jbe	.LBB39_24
# %bb.5:                                #   in Loop: Header=BB39_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm2, (%rbx)
	movss	%xmm0, (%rbx,%rbp,4)
	jmp	.LBB39_32
	.p2align	4
.LBB39_27:                              #   in Loop: Header=BB39_3 Depth=1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB39_29
# %bb.28:                               #   in Loop: Header=BB39_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx)
	movss	%xmm0, 4(%rbx)
	jmp	.LBB39_32
	.p2align	4
.LBB39_24:                              #   in Loop: Header=BB39_3 Depth=1
	ucomiss	%xmm1, %xmm0
	movss	(%rbx), %xmm2                   # xmm2 = mem[0],zero,zero,zero
	jbe	.LBB39_26
# %bb.25:                               #   in Loop: Header=BB39_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm2, -4(%rsi)
	jmp	.LBB39_32
	.p2align	4
.LBB39_29:                              #   in Loop: Header=BB39_3 Depth=1
	ucomiss	%xmm2, %xmm0
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	jbe	.LBB39_31
# %bb.30:                               #   in Loop: Header=BB39_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm1, -4(%rsi)
	jmp	.LBB39_32
.LBB39_26:                              #   in Loop: Header=BB39_3 Depth=1
	movss	%xmm1, (%rbx)
	movss	%xmm2, 4(%rbx)
	jmp	.LBB39_32
.LBB39_31:                              #   in Loop: Header=BB39_3 Depth=1
	movss	%xmm2, (%rbx)
	movss	%xmm1, (%rbx,%rbp,4)
	.p2align	4
.LBB39_32:                              #   in Loop: Header=BB39_3 Depth=1
	decq	%r14
	leaq	4(%rbx), %r12
	movq	%rsi, %rax
	.p2align	4
.LBB39_33:                              #   Parent Loop BB39_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB39_34 Depth 3
                                        #       Child Loop BB39_36 Depth 3
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	leaq	(%r12,%r13), %rbp
	.p2align	4
.LBB39_34:                              #   Parent Loop BB39_3 Depth=1
                                        #     Parent Loop BB39_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	(%r12), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	addq	$4, %r12
	addq	$4, %rbp
	ucomiss	%xmm1, %xmm0
	ja	.LBB39_34
# %bb.35:                               #   in Loop: Header=BB39_33 Depth=2
	leaq	-4(%r12), %r15
	.p2align	4
.LBB39_36:                              #   Parent Loop BB39_3 Depth=1
                                        #     Parent Loop BB39_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	-4(%rax), %xmm2                 # xmm2 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm0, %xmm2
	ja	.LBB39_36
# %bb.37:                               #   in Loop: Header=BB39_33 Depth=2
	cmpq	%rax, %r15
	jae	.LBB39_39
# %bb.38:                               #   in Loop: Header=BB39_33 Depth=2
	movss	%xmm2, (%r15)
	movss	%xmm1, (%rax)
	jmp	.LBB39_33
	.p2align	4
.LBB39_39:                              #   in Loop: Header=BB39_3 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rdx
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	sarq	$2, %rbp
	cmpq	$16, %rbp
	jle	.LBB39_40
# %bb.6:                                #   in Loop: Header=BB39_3 Depth=1
	movq	%r15, %rsi
	testq	%r14, %r14
	jne	.LBB39_3
# %bb.7:
	addq	$-4, %r12
	movq	%r12, %rsi
.LBB39_8:
	leaq	7(%rsp), %rdx
	movq	%rbx, %rdi
	movq	%rsi, %r14
	callq	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	jmp	.LBB39_9
	.p2align	4
.LBB39_22:                              #   in Loop: Header=BB39_9 Depth=1
	xorl	%ecx, %ecx
.LBB39_23:                              #   in Loop: Header=BB39_9 Depth=1
	movss	%xmm0, (%rbx,%rcx,4)
	cmpq	$4, %rax
	jle	.LBB39_40
.LBB39_9:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB39_12 Depth 2
                                        #     Child Loop BB39_20 Depth 2
	movss	-4(%r14), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, -4(%r14)
	addq	$-4, %r14
	movq	%r14, %rax
	subq	%rbx, %rax
	movq	%rax, %rdx
	sarq	$2, %rdx
	cmpq	$3, %rdx
	jl	.LBB39_10
# %bb.11:                               #   in Loop: Header=BB39_9 Depth=1
	leaq	-1(%rdx), %rcx
	shrq	$63, %rcx
	leaq	(%rdx,%rcx), %rsi
	decq	%rsi
	sarq	%rsi
	xorl	%edi, %edi
	jmp	.LBB39_12
	.p2align	4
.LBB39_14:                              #   in Loop: Header=BB39_12 Depth=2
	leaq	2(,%rdi,2), %rcx
.LBB39_15:                              #   in Loop: Header=BB39_12 Depth=2
	movss	(%rbx,%rcx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rdi,4)
	movq	%rcx, %rdi
	cmpq	%rsi, %rcx
	jge	.LBB39_16
.LBB39_12:                              #   Parent Loop BB39_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rdi,%rdi), %rcx
	movss	4(%rbx,%rcx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rbx,%rcx,4), %xmm1
	jbe	.LBB39_14
# %bb.13:                               #   in Loop: Header=BB39_12 Depth=2
	leaq	1(,%rdi,2), %rcx
	jmp	.LBB39_15
	.p2align	4
.LBB39_10:                              #   in Loop: Header=BB39_9 Depth=1
	xorl	%ecx, %ecx
.LBB39_16:                              #   in Loop: Header=BB39_9 Depth=1
	testb	$4, %al
	jne	.LBB39_19
# %bb.17:                               #   in Loop: Header=BB39_9 Depth=1
	addq	$-2, %rdx
	sarq	%rdx
	cmpq	%rdx, %rcx
	jne	.LBB39_19
# %bb.18:                               #   in Loop: Header=BB39_9 Depth=1
	leaq	(%rcx,%rcx), %rdx
	movss	4(%rbx,%rdx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rcx,4)
	leaq	1(,%rcx,2), %rcx
	jmp	.LBB39_20
	.p2align	4
.LBB39_19:                              #   in Loop: Header=BB39_9 Depth=1
	testq	%rcx, %rcx
	je	.LBB39_22
	.p2align	4
.LBB39_20:                              #   Parent Loop BB39_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rcx), %rdx
	shrq	%rdx
	movss	(%rbx,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB39_23
# %bb.21:                               #   in Loop: Header=BB39_20 Depth=2
	movss	%xmm1, (%rbx,%rcx,4)
	movq	%rdx, %rcx
	testq	%rdx, %rdx
	jne	.LBB39_20
	jmp	.LBB39_22
.LBB39_40:
	addq	$8, %rsp
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
.Lfunc_end39:
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_, .Lfunc_end39-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_ # -- Begin function _ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.prefalign	4, .Lfunc_end40, nop
	.type	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,@function
_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_: # @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
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
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movq	%rsi, %rax
	subq	%rdi, %rax
	cmpq	$65, %rax
	jl	.LBB40_2
# %bb.1:
	leaq	4(%r14), %r15
	movl	$4, %r12d
	movq	%r15, %r13
	movq	%r14, %rbp
	jmp	.LBB40_18
.LBB40_2:
	cmpq	%rbx, %r14
	je	.LBB40_30
# %bb.3:
	leaq	4(%r14), %rax
	cmpq	%rbx, %rax
	je	.LBB40_30
# %bb.4:
	movq	%r14, %r15
	jmp	.LBB40_9
	.p2align	4
.LBB40_5:                               #   in Loop: Header=BB40_9 Depth=1
	movq	%r15, %rdx
	subq	%r14, %rdx
	movq	%rdx, %rax
	sarq	$2, %rax
	cmpq	$2, %rax
	jl	.LBB40_13
# %bb.6:                                #   in Loop: Header=BB40_9 Depth=1
	shlq	$2, %rax
	subq	%rax, %rdi
	addq	$8, %rdi
	movq	%r14, %rsi
	movss	%xmm1, 4(%rsp)                  # 4-byte Spill
	callq	memmove@PLT
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
.LBB40_7:                               #   in Loop: Header=BB40_9 Depth=1
	movq	%r14, %rax
.LBB40_8:                               #   in Loop: Header=BB40_9 Depth=1
	movss	%xmm1, (%rax)
	leaq	4(%r15), %rax
	cmpq	%rbx, %rax
	je	.LBB40_30
.LBB40_9:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB40_12 Depth 2
	movq	%r15, %rdi
	movq	%rax, %r15
	movss	4(%rdi), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	ja	.LBB40_5
# %bb.10:                               #   in Loop: Header=BB40_9 Depth=1
	movss	(%rdi), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%r15, %rax
	jbe	.LBB40_8
# %bb.11:                               #   in Loop: Header=BB40_9 Depth=1
	movq	%r15, %rax
	.p2align	4
.LBB40_12:                              #   Parent Loop BB40_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB40_12
	jmp	.LBB40_8
.LBB40_13:                              #   in Loop: Header=BB40_9 Depth=1
	movq	%r14, %rax
	cmpq	$4, %rdx
	jne	.LBB40_8
# %bb.14:                               #   in Loop: Header=BB40_9 Depth=1
	movss	%xmm0, 4(%rdi)
	jmp	.LBB40_7
.LBB40_15:                              #   in Loop: Header=BB40_18 Depth=1
	movss	%xmm0, 4(%rax)
	.p2align	4
.LBB40_16:                              #   in Loop: Header=BB40_18 Depth=1
	movq	%r14, %rax
.LBB40_17:                              #   in Loop: Header=BB40_18 Depth=1
	movss	%xmm1, (%rax)
	addq	$4, %r12
	addq	$4, %r13
	cmpq	$64, %r12
	je	.LBB40_24
.LBB40_18:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB40_23 Depth 2
	movq	%rbp, %rax
	leaq	(%r14,%r12), %rbp
	movss	(%r14,%r12), %xmm1              # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB40_21
# %bb.19:                               #   in Loop: Header=BB40_18 Depth=1
	cmpq	$5, %r12
	jb	.LBB40_15
# %bb.20:                               #   in Loop: Header=BB40_18 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rsi
	movq	%r12, %rdx
	movss	%xmm1, 4(%rsp)                  # 4-byte Spill
	callq	memmove@PLT
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jmp	.LBB40_16
	.p2align	4
.LBB40_21:                              #   in Loop: Header=BB40_18 Depth=1
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%rbp, %rax
	jbe	.LBB40_17
# %bb.22:                               #   in Loop: Header=BB40_18 Depth=1
	movq	%r13, %rax
	.p2align	4
.LBB40_23:                              #   Parent Loop BB40_18 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB40_23
	jmp	.LBB40_17
.LBB40_24:
	addq	$64, %r14
	jmp	.LBB40_26
	.p2align	4
.LBB40_25:                              #   in Loop: Header=BB40_26 Depth=1
	movss	%xmm0, (%rax)
	addq	$4, %r14
.LBB40_26:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB40_29 Depth 2
	cmpq	%rbx, %r14
	je	.LBB40_30
# %bb.27:                               #   in Loop: Header=BB40_26 Depth=1
	movss	-4(%r14), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm0, %xmm1
	movq	%r14, %rax
	jbe	.LBB40_25
# %bb.28:                               #   in Loop: Header=BB40_26 Depth=1
	movq	%r14, %rax
	.p2align	4
.LBB40_29:                              #   Parent Loop BB40_26 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm1, (%rax)
	movss	-8(%rax), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm0, %xmm1
	ja	.LBB40_29
	jmp	.LBB40_25
.LBB40_30:
	addq	$8, %rsp
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
.Lfunc_end40:
	.size	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_, .Lfunc_end40-_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,"axG",@progbits,_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,comdat
	.weak	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_ # -- Begin function _ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.prefalign	4, .Lfunc_end41, nop
	.type	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,@function
_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_: # @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_startproc
# %bb.0:
	subq	%rdi, %rsi
	movq	%rsi, %rax
	sarq	$2, %rax
	cmpq	$2, %rax
	jge	.LBB41_2
.LBB41_1:
	retq
.LBB41_2:
	leaq	-2(%rax), %rdx
	movq	%rdx, %rcx
	shrq	%rcx
	decq	%rax
	shrq	%rax
	testb	$4, %sil
	jne	.LBB41_20
# %bb.3:
	incq	%rdx
	movq	%rcx, %rsi
	jmp	.LBB41_6
	.p2align	4
.LBB41_4:                               #   in Loop: Header=BB41_6 Depth=1
	movq	%r8, %r9
.LBB41_5:                               #   in Loop: Header=BB41_6 Depth=1
	movss	%xmm0, (%rdi,%r9,4)
	subq	$1, %rsi
	jb	.LBB41_1
.LBB41_6:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB41_10 Depth 2
                                        #     Child Loop BB41_15 Depth 2
	movss	(%rdi,%rsi,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%rsi, %r8
	cmpq	%rax, %rsi
	jge	.LBB41_12
# %bb.7:                                #   in Loop: Header=BB41_6 Depth=1
	movq	%rsi, %r9
	jmp	.LBB41_10
	.p2align	4
.LBB41_8:                               #   in Loop: Header=BB41_10 Depth=2
	leaq	2(,%r9,2), %r8
.LBB41_9:                               #   in Loop: Header=BB41_10 Depth=2
	movss	(%rdi,%r8,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%r9,4)
	movq	%r8, %r9
	cmpq	%rax, %r8
	jge	.LBB41_12
.LBB41_10:                              #   Parent Loop BB41_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r9,%r9), %r8
	movss	4(%rdi,%r8,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rdi,%r8,4), %xmm1
	jbe	.LBB41_8
# %bb.11:                               #   in Loop: Header=BB41_10 Depth=2
	leaq	1(,%r9,2), %r8
	jmp	.LBB41_9
	.p2align	4
.LBB41_12:                              #   in Loop: Header=BB41_6 Depth=1
	cmpq	%rcx, %r8
	jne	.LBB41_14
# %bb.13:                               #   in Loop: Header=BB41_6 Depth=1
	movss	(%rdi,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%rcx,4)
	movq	%rdx, %r8
.LBB41_14:                              #   in Loop: Header=BB41_6 Depth=1
	cmpq	%rsi, %r8
	jle	.LBB41_4
	.p2align	4
.LBB41_15:                              #   Parent Loop BB41_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%r8), %r9
	shrq	$63, %r9
	addq	%r8, %r9
	decq	%r9
	sarq	%r9
	movss	(%rdi,%r9,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB41_4
# %bb.16:                               #   in Loop: Header=BB41_15 Depth=2
	movss	%xmm1, (%rdi,%r8,4)
	movq	%r9, %r8
	cmpq	%rsi, %r9
	jg	.LBB41_15
	jmp	.LBB41_5
	.p2align	4
.LBB41_18:                              #   in Loop: Header=BB41_20 Depth=1
	movq	%rdx, %rsi
.LBB41_19:                              #   in Loop: Header=BB41_20 Depth=1
	movss	%xmm0, (%rdi,%rsi,4)
	subq	$1, %rcx
	jb	.LBB41_1
.LBB41_20:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB41_24 Depth 2
                                        #     Child Loop BB41_27 Depth 2
	movss	(%rdi,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%rcx, %rsi
	cmpq	%rax, %rcx
	jge	.LBB41_19
# %bb.21:                               #   in Loop: Header=BB41_20 Depth=1
	movq	%rcx, %rsi
	jmp	.LBB41_24
	.p2align	4
.LBB41_22:                              #   in Loop: Header=BB41_24 Depth=2
	leaq	2(,%rsi,2), %rdx
.LBB41_23:                              #   in Loop: Header=BB41_24 Depth=2
	movss	(%rdi,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%rsi,4)
	movq	%rdx, %rsi
	cmpq	%rax, %rdx
	jge	.LBB41_26
.LBB41_24:                              #   Parent Loop BB41_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rsi,%rsi), %rdx
	movss	4(%rdi,%rdx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rdi,%rdx,4), %xmm1
	jbe	.LBB41_22
# %bb.25:                               #   in Loop: Header=BB41_24 Depth=2
	leaq	1(,%rsi,2), %rdx
	jmp	.LBB41_23
	.p2align	4
.LBB41_26:                              #   in Loop: Header=BB41_20 Depth=1
	cmpq	%rcx, %rdx
	jle	.LBB41_18
	.p2align	4
.LBB41_27:                              #   Parent Loop BB41_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rdx), %rsi
	shrq	$63, %rsi
	addq	%rdx, %rsi
	decq	%rsi
	sarq	%rsi
	movss	(%rdi,%rsi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB41_18
# %bb.28:                               #   in Loop: Header=BB41_27 Depth=2
	movss	%xmm1, (%rdi,%rdx,4)
	movq	%rsi, %rdx
	cmpq	%rcx, %rsi
	jg	.LBB41_27
	jmp	.LBB41_19
.Lfunc_end41:
	.size	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_, .Lfunc_end41-_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_,"axG",@progbits,_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_,comdat
	.weak	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_ # -- Begin function _ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_
	.prefalign	4, .Lfunc_end42, nop
	.type	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_,@function
_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_: # @_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
	movq	%rsi, %rdi
	callq	strlen@PLT
	movq	%r14, %rdi
	xorl	%esi, %esi
	xorl	%edx, %edx
	movq	%r15, %rcx
	movq	%rax, %r8
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm
	movq	%rax, %r14
	movq	%rax, %r15
	leaq	16(%rbx), %rdi
	movq	%rdi, (%rbx)
	movq	(%rax), %rax
	addq	$16, %r14
	cmpq	%r14, %rax
	je	.LBB42_1
# %bb.2:
	movq	%rax, (%rbx)
	movq	16(%r15), %rax
	movq	%rax, 16(%rbx)
	movq	8(%r15), %r12
	jmp	.LBB42_3
.LBB42_1:
	movq	8(%r15), %r12
	leaq	1(%r12), %rdx
	movq	%r14, %rsi
	callq	memcpy@PLT
.LBB42_3:
	movq	%r12, 8(%rbx)
	movq	%r14, (%r15)
	movq	$0, 8(%r15)
	movb	$0, 16(%r15)
	movq	%rbx, %rax
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end42:
	.size	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_, .Lfunc_end42-_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag,comdat
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag # -- Begin function _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag
	.p2align	1
	.prefalign	4, .Lfunc_end43, nop
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag,@function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag: # @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag
.Lfunc_begin20:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception20
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
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r8d, 12(%rsp)                  # 4-byte Spill
	movq	%rcx, %rbx
	movl	%edx, %r12d
	movq	%rsi, %r15
	movq	%rdi, 16(%rsp)                  # 8-byte Spill
	testq	%rsi, %rsi
	setne	%al
	cmpl	$-1, %edx
	sete	%dl
	andb	%al, %dl
	cmpb	$1, %dl
	jne	.LBB43_1
# %bb.2:
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_3
# %bb.4:
	movzbl	(%rax), %r13d
	jmp	.LBB43_5
.LBB43_1:
	movl	%r12d, %r13d
.LBB43_5:
	movq	16(%rsp), %rax                  # 8-byte Reload
	addq	$16, %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movl	12(%rsp), %edx                  # 4-byte Reload
	cmpl	$-1, %edx
	sete	%al
	testq	%rbx, %rbx
	setne	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB43_6
# %bb.7:
	movq	16(%rbx), %rax
	cmpq	24(%rbx), %rax
	jae	.LBB43_8
# %bb.13:
	cmpl	$-1, %r13d
	je	.LBB43_10
.LBB43_14:
	xorl	%ebp, %ebp
	movq	%rbx, %r13
	jmp	.LBB43_33
.LBB43_6:
	movl	%edx, %eax
.LBB43_9:
	cmpl	$-1, %r13d
	sete	%cl
	cmpl	$-1, %eax
	setne	%al
	xorb	%cl, %al
	jne	.LBB43_14
.LBB43_10:
	movl	12(%rsp), %r14d                 # 4-byte Reload
	movq	%rbx, %r13
	cmpl	$-1, %r12d
	sete	%al
	testq	%r15, %r15
	setne	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB43_16
# %bb.11:
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_12
# %bb.15:
	movzbl	(%rax), %r12d
.LBB43_16:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movb	%r12b, (%rax)
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_17
# %bb.18:
	incq	%rax
	movq	%rax, 16(%r15)
.LBB43_19:
	xorl	%ebp, %ebp
	.p2align	4
.LBB43_20:                              # =>This Inner Loop Header: Depth=1
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_22
# %bb.21:                               #   in Loop: Header=BB43_20 Depth=1
	xorl	%ebx, %ebx
.LBB43_24:                              #   in Loop: Header=BB43_20 Depth=1
	cmpl	$-1, %r14d
	sete	%al
	testq	%r13, %r13
	setne	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB43_25
# %bb.26:                               #   in Loop: Header=BB43_20 Depth=1
	movq	16(%r13), %rax
	cmpq	24(%r13), %rax
	jae	.LBB43_28
# %bb.27:                               #   in Loop: Header=BB43_20 Depth=1
	movzbl	(%rax), %eax
	jmp	.LBB43_30
	.p2align	4
.LBB43_25:                              #   in Loop: Header=BB43_20 Depth=1
	movl	%r14d, %eax
.LBB43_30:                              #   in Loop: Header=BB43_20 Depth=1
	cmpl	$-1, %eax
	setne	%al
	xorb	%al, %bl
	jne	.LBB43_32
# %bb.31:                               #   in Loop: Header=BB43_20 Depth=1
	leaq	1(%rbp), %rbx
	cmpq	$15, %rbx
	jae	.LBB43_32
# %bb.65:                               #   in Loop: Header=BB43_20 Depth=1
	testq	%r15, %r15
	je	.LBB43_66
# %bb.67:                               #   in Loop: Header=BB43_20 Depth=1
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_69
# %bb.68:                               #   in Loop: Header=BB43_20 Depth=1
	movzbl	(%rax), %eax
	jmp	.LBB43_71
	.p2align	4
.LBB43_66:                              #   in Loop: Header=BB43_20 Depth=1
	movb	$-1, %al
.LBB43_70:                              #   in Loop: Header=BB43_20 Depth=1
	xorl	%r15d, %r15d
.LBB43_71:                              #   in Loop: Header=BB43_20 Depth=1
	movq	16(%rsp), %rcx                  # 8-byte Reload
	movb	%al, 17(%rcx,%rbp)
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_73
# %bb.72:                               #   in Loop: Header=BB43_20 Depth=1
	incq	%rax
	movq	%rax, 16(%r15)
	movq	%rbx, %rbp
	jmp	.LBB43_20
.LBB43_22:                              #   in Loop: Header=BB43_20 Depth=1
	movq	(%r15), %rax
	movq	%r15, %rdi
	callq	*72(%rax)
	cmpl	$-1, %eax
	sete	%bl
	jne	.LBB43_24
# %bb.23:                               #   in Loop: Header=BB43_20 Depth=1
	xorl	%r15d, %r15d
	jmp	.LBB43_24
.LBB43_73:                              #   in Loop: Header=BB43_20 Depth=1
	movq	(%r15), %rax
	movq	%r15, %rdi
	callq	*80(%rax)
	movq	%rbx, %rbp
	jmp	.LBB43_20
.LBB43_69:                              #   in Loop: Header=BB43_20 Depth=1
	movq	(%r15), %rax
	movq	%r15, %rdi
	callq	*72(%rax)
	cmpl	$-1, %eax
	je	.LBB43_70
	jmp	.LBB43_71
.LBB43_28:                              #   in Loop: Header=BB43_20 Depth=1
	movq	(%r13), %rax
	movq	%r13, %rdi
	callq	*72(%rax)
	cmpl	$-1, %eax
	jne	.LBB43_30
# %bb.29:                               #   in Loop: Header=BB43_20 Depth=1
	xorl	%r13d, %r13d
	jmp	.LBB43_30
.LBB43_32:
	incq	%rbp
	movl	$-1, %r12d
.LBB43_33:
	movq	%r13, 24(%rsp)                  # 8-byte Spill
	testq	%r15, %r15
	setne	%al
	cmpl	$-1, %r12d
	sete	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB43_34
# %bb.35:
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_36
# %bb.38:
	movzbl	(%rax), %r13d
	jmp	.LBB43_39
.LBB43_34:
	movl	%r12d, %r13d
.LBB43_39:
	movl	12(%rsp), %edx                  # 4-byte Reload
	cmpl	$-1, %edx
	sete	%al
	movq	24(%rsp), %rsi                  # 8-byte Reload
	testq	%rsi, %rsi
	setne	%cl
	testb	%al, %cl
	je	.LBB43_40
# %bb.42:
	movq	16(%rsi), %rax
	cmpq	24(%rsi), %rax
	jae	.LBB43_43
.LBB43_47:
	cmpl	$-1, %r13d
	jne	.LBB43_41
.LBB43_48:
	cmpq	$15, %rbp
	jne	.LBB43_49
.LBB43_50:
.Ltmp1268:                              # EH_LABEL
	movl	$31, %edi
	callq	_Znwm@PLT
.Ltmp1269:                              # EH_LABEL
# %bb.51:
	movq	%rax, %rbx
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rdi
	movq	(%rdi), %rax
	movq	7(%rdi), %rcx
	movq	%rcx, 7(%rbx)
	movq	%rax, (%rbx)
	movq	32(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rdi
	je	.LBB43_53
# %bb.52:
	movq	(%rax), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB43_53:
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rbx, (%rax)
	movq	$30, 16(%rax)
	movl	$30, %r14d
	jmp	.LBB43_54
.LBB43_40:
	cmpl	$-1, %edx
	setne	%al
	cmpl	$-1, %r13d
	sete	%cl
	xorb	%al, %cl
	je	.LBB43_48
.LBB43_41:
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rbp, 8(%rax)
	movq	(%rax), %rax
	movb	$0, (%rax,%rbp)
	addq	$40, %rsp
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
.LBB43_3:
	.cfi_def_cfa_offset 96
	movq	(%r15), %rax
	movq	%r15, %rdi
	callq	*72(%rax)
	movl	%eax, %r13d
	xorl	%eax, %eax
	cmpl	$-1, %r13d
	cmoveq	%rax, %r15
	jmp	.LBB43_5
.LBB43_8:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	callq	*72(%rax)
	xorl	%ecx, %ecx
	cmpl	$-1, %eax
	cmoveq	%rcx, %rbx
	jmp	.LBB43_9
.LBB43_36:
	movq	(%r15), %rax
.Ltmp1264:                              # EH_LABEL
	movq	%r15, %rdi
	callq	*72(%rax)
.Ltmp1265:                              # EH_LABEL
# %bb.37:
	movl	%eax, %r13d
	xorl	%eax, %eax
	cmpl	$-1, %r13d
	cmoveq	%rax, %r15
	jmp	.LBB43_39
.LBB43_43:
	movq	24(%rsp), %rdi                  # 8-byte Reload
	movq	(%rdi), %rax
.Ltmp1266:                              # EH_LABEL
	callq	*72(%rax)
.Ltmp1267:                              # EH_LABEL
# %bb.44:
	cmpl	$-1, %eax
	jne	.LBB43_47
# %bb.45:
	cmpl	$-1, %r13d
	je	.LBB43_41
# %bb.46:
	movq	$0, 24(%rsp)                    # 8-byte Folded Spill
	cmpq	$15, %rbp
	je	.LBB43_50
.LBB43_49:
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rbx
	movl	$15, %r14d
.LBB43_54:
	cmpl	$-1, %r12d
	sete	%al
	testq	%r15, %r15
	setne	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB43_59
# %bb.55:
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_56
# %bb.58:
	movzbl	(%rax), %r12d
.LBB43_59:
	movb	%r12b, (%rbx,%rbp)
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_60
.LBB43_64:
	incq	%rax
	movq	%rax, 16(%r15)
.LBB43_61:
	incq	%rbp
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	movl	12(%rsp), %edx                  # 4-byte Reload
	jae	.LBB43_74
	.p2align	4
.LBB43_63:
	xorl	%ebx, %ebx
.LBB43_78:
	movq	24(%rsp), %rdi                  # 8-byte Reload
	cmpl	$-1, %edx
	sete	%al
	testq	%rdi, %rdi
	setne	%cl
	andb	%al, %cl
	cmpb	$1, %cl
	jne	.LBB43_85
# %bb.79:
	movq	16(%rdi), %rax
	cmpq	24(%rdi), %rax
	jae	.LBB43_80
.LBB43_84:
	testb	%bl, %bl
	je	.LBB43_41
.LBB43_86:
	cmpq	%r14, %rbp
	movq	%rdi, 24(%rsp)                  # 8-byte Spill
	jne	.LBB43_87
.LBB43_90:
	movq	%r14, %rax
	incq	%rax
	js	.LBB43_91
# %bb.93:
	cmpq	$-1, %r14
	je	.LBB43_94
# %bb.99:
	leaq	(%r14,%r14), %rcx
	movabsq	$9223372036854775807, %r13      # imm = 0x7FFFFFFFFFFFFFFF
	cmpq	%r13, %rcx
	cmovbq	%rcx, %r13
	cmpq	%rcx, %rax
	cmovaeq	%rax, %r13
	movq	%r13, %rdi
	incq	%rdi
	jns	.LBB43_95
	jmp	.LBB43_100
	.p2align	4
.LBB43_85:
	cmpl	$-1, %edx
	setne	%al
	xorb	%bl, %al
	je	.LBB43_86
	jmp	.LBB43_41
	.p2align	4
.LBB43_94:
	movl	$1, %edi
	xorl	%r13d, %r13d
.LBB43_95:
.Ltmp1283:                              # EH_LABEL
	callq	_Znwm@PLT
.Ltmp1284:                              # EH_LABEL
# %bb.96:
	movq	%rax, %r12
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rbx
	testq	%r14, %r14
	je	.LBB43_103
# %bb.97:
	cmpq	$1, %r14
	jne	.LBB43_102
# %bb.98:
	movzbl	(%rbx), %eax
	movb	%al, (%r12)
.LBB43_103:
	movq	32(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rbx
	je	.LBB43_105
.LBB43_104:
	movq	(%rax), %rsi
	incq	%rsi
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
.LBB43_105:
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%r12, (%rax)
	movq	%r13, 16(%rax)
	testq	%r15, %r15
	jne	.LBB43_110
.LBB43_89:
	movb	$-1, %al
.LBB43_114:
	xorl	%r15d, %r15d
	jmp	.LBB43_115
	.p2align	4
.LBB43_102:
	movq	%r12, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
	movq	32(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rbx
	jne	.LBB43_104
	jmp	.LBB43_105
.LBB43_74:
	movq	(%r15), %rax
.Ltmp1277:                              # EH_LABEL
	movq	%r15, %rdi
	callq	*72(%rax)
.Ltmp1278:                              # EH_LABEL
# %bb.75:
	cmpl	$-1, %eax
	sete	%bl
	jne	.LBB43_77
# %bb.76:
	xorl	%r15d, %r15d
.LBB43_77:
	movl	12(%rsp), %edx                  # 4-byte Reload
	jmp	.LBB43_78
.LBB43_80:
	movq	(%rdi), %rax
.Ltmp1279:                              # EH_LABEL
	callq	*72(%rax)
.Ltmp1280:                              # EH_LABEL
# %bb.81:
	cmpl	$-1, %eax
	movq	24(%rsp), %rdi                  # 8-byte Reload
	jne	.LBB43_84
# %bb.82:
	testb	%bl, %bl
	jne	.LBB43_41
# %bb.83:
	xorl	%edi, %edi
	cmpq	%r14, %rbp
	movq	%rdi, 24(%rsp)                  # 8-byte Spill
	je	.LBB43_90
	.p2align	4
.LBB43_87:
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %r12
	movq	%r14, %r13
	testq	%r15, %r15
	je	.LBB43_89
.LBB43_110:
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_112
# %bb.111:
	movzbl	(%rax), %eax
.LBB43_115:
	movb	%al, (%r12,%rbp)
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jae	.LBB43_117
# %bb.116:
	incq	%rax
	movq	%rax, 16(%r15)
.LBB43_118:
	incq	%rbp
	movq	%r13, %r14
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	movl	12(%rsp), %edx                  # 4-byte Reload
	jb	.LBB43_63
	jmp	.LBB43_74
.LBB43_117:
	movq	(%r15), %rax
.Ltmp1289:                              # EH_LABEL
	movq	%r15, %rdi
	callq	*80(%rax)
.Ltmp1290:                              # EH_LABEL
	jmp	.LBB43_118
.LBB43_112:
	movq	(%r15), %rax
.Ltmp1286:                              # EH_LABEL
	movq	%r15, %rdi
	callq	*72(%rax)
.Ltmp1287:                              # EH_LABEL
# %bb.113:
	cmpl	$-1, %eax
	je	.LBB43_114
	jmp	.LBB43_115
.LBB43_100:
.Ltmp1281:                              # EH_LABEL
	callq	_ZSt17__throw_bad_allocv@PLT
.Ltmp1282:                              # EH_LABEL
# %bb.101:
.LBB43_17:
	movq	(%r15), %rax
	movq	%r15, %rdi
	callq	*80(%rax)
	jmp	.LBB43_19
.LBB43_12:
	movq	(%r15), %rax
	movq	%r15, %rdi
	callq	*72(%rax)
	movl	%eax, %r12d
	xorl	%eax, %eax
	cmpl	$-1, %r12d
	cmoveq	%rax, %r15
	jmp	.LBB43_16
.LBB43_56:
	movq	(%r15), %rax
.Ltmp1271:                              # EH_LABEL
	movq	%r15, %rdi
	callq	*72(%rax)
.Ltmp1272:                              # EH_LABEL
# %bb.57:
	movl	%eax, %r12d
	xorl	%eax, %eax
	cmpl	$-1, %r12d
	cmoveq	%rax, %r15
	movb	%r12b, (%rbx,%rbp)
	movq	16(%r15), %rax
	cmpq	24(%r15), %rax
	jb	.LBB43_64
.LBB43_60:
	movq	(%r15), %rax
.Ltmp1274:                              # EH_LABEL
	movq	%r15, %rdi
	callq	*80(%rax)
.Ltmp1275:                              # EH_LABEL
	jmp	.LBB43_61
.LBB43_91:
.Ltmp1292:                              # EH_LABEL
	leaq	.L.str.34(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp1293:                              # EH_LABEL
# %bb.92:
.LBB43_120:
.Ltmp1273:                              # EH_LABEL
	jmp	.LBB43_122
.LBB43_106:
.Ltmp1276:                              # EH_LABEL
	jmp	.LBB43_122
.LBB43_119:
.Ltmp1288:                              # EH_LABEL
	jmp	.LBB43_122
.LBB43_121:
.Ltmp1291:                              # EH_LABEL
	jmp	.LBB43_122
.LBB43_108:
.Ltmp1270:                              # EH_LABEL
	jmp	.LBB43_122
.LBB43_107:
.Ltmp1285:                              # EH_LABEL
	jmp	.LBB43_122
.LBB43_109:
.Ltmp1294:                              # EH_LABEL
.LBB43_122:
	movq	%rax, %r14
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rdi
	movq	32(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rdi
	je	.LBB43_124
# %bb.123:
	movq	(%rax), %rsi
	incq	%rsi
	callq	_ZdlPvm@PLT
.LBB43_124:
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end43:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag, .Lfunc_end43-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag
	.cfi_endproc
	.section	.gcc_except_table._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag,"aG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructISt19istreambuf_iteratorIcS2_EEEvT_S8_St18input_iterator_tag,comdat
	.p2align	2, 0x0
GCC_except_table43:
.Lexception20:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end20-.Lcst_begin20
.Lcst_begin20:
	.uleb128 .Lfunc_begin20-.Lfunc_begin20  # >> Call Site 1 <<
	.uleb128 .Ltmp1268-.Lfunc_begin20       #   Call between .Lfunc_begin20 and .Ltmp1268
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1268-.Lfunc_begin20       # >> Call Site 2 <<
	.uleb128 .Ltmp1269-.Ltmp1268            #   Call between .Ltmp1268 and .Ltmp1269
	.uleb128 .Ltmp1270-.Lfunc_begin20       #     jumps to .Ltmp1270
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1269-.Lfunc_begin20       # >> Call Site 3 <<
	.uleb128 .Ltmp1264-.Ltmp1269            #   Call between .Ltmp1269 and .Ltmp1264
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1264-.Lfunc_begin20       # >> Call Site 4 <<
	.uleb128 .Ltmp1267-.Ltmp1264            #   Call between .Ltmp1264 and .Ltmp1267
	.uleb128 .Ltmp1276-.Lfunc_begin20       #     jumps to .Ltmp1276
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1283-.Lfunc_begin20       # >> Call Site 5 <<
	.uleb128 .Ltmp1284-.Ltmp1283            #   Call between .Ltmp1283 and .Ltmp1284
	.uleb128 .Ltmp1285-.Lfunc_begin20       #     jumps to .Ltmp1285
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1284-.Lfunc_begin20       # >> Call Site 6 <<
	.uleb128 .Ltmp1277-.Ltmp1284            #   Call between .Ltmp1284 and .Ltmp1277
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1277-.Lfunc_begin20       # >> Call Site 7 <<
	.uleb128 .Ltmp1290-.Ltmp1277            #   Call between .Ltmp1277 and .Ltmp1290
	.uleb128 .Ltmp1291-.Lfunc_begin20       #     jumps to .Ltmp1291
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1286-.Lfunc_begin20       # >> Call Site 8 <<
	.uleb128 .Ltmp1287-.Ltmp1286            #   Call between .Ltmp1286 and .Ltmp1287
	.uleb128 .Ltmp1288-.Lfunc_begin20       #     jumps to .Ltmp1288
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1281-.Lfunc_begin20       # >> Call Site 9 <<
	.uleb128 .Ltmp1282-.Ltmp1281            #   Call between .Ltmp1281 and .Ltmp1282
	.uleb128 .Ltmp1294-.Lfunc_begin20       #     jumps to .Ltmp1294
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1282-.Lfunc_begin20       # >> Call Site 10 <<
	.uleb128 .Ltmp1271-.Ltmp1282            #   Call between .Ltmp1282 and .Ltmp1271
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1271-.Lfunc_begin20       # >> Call Site 11 <<
	.uleb128 .Ltmp1272-.Ltmp1271            #   Call between .Ltmp1271 and .Ltmp1272
	.uleb128 .Ltmp1273-.Lfunc_begin20       #     jumps to .Ltmp1273
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1274-.Lfunc_begin20       # >> Call Site 12 <<
	.uleb128 .Ltmp1275-.Ltmp1274            #   Call between .Ltmp1274 and .Ltmp1275
	.uleb128 .Ltmp1276-.Lfunc_begin20       #     jumps to .Ltmp1276
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1292-.Lfunc_begin20       # >> Call Site 13 <<
	.uleb128 .Ltmp1293-.Ltmp1292            #   Call between .Ltmp1292 and .Ltmp1293
	.uleb128 .Ltmp1294-.Lfunc_begin20       #     jumps to .Ltmp1294
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1293-.Lfunc_begin20       # >> Call Site 14 <<
	.uleb128 .Lfunc_end43-.Ltmp1293         #   Call between .Ltmp1293 and .Lfunc_end43
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end20:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end44, nop    # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_57a29556a1d631fd(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB44_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_57a29556a1d631fd(%rip)
.LBB44_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	leaq	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm(%rip), %rsi
	leaq	.L__unnamed_5(%rip), %rcx
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
.Lfunc_end44:
	.size	__hip_module_ctor, .Lfunc_end44-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end45, nop    # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_57a29556a1d631fd(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB45_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_57a29556a1d631fd(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB45_2:
	retq
.Lfunc_end45:
	.size	__hip_module_dtor, .Lfunc_end45-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	.L.str.3,@object                # @.str.3
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.3:
	.asciz	"usage: "
	.size	.L.str.3, 8

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	" --output REPORT.json --assembly FILE.s --static-receipt FILE.txt\n"
	.size	.L.str.4, 67

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"hipGetDeviceCount(&count)"
	.size	.L.str.5, 26

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"/ssdpool2nvme/local_llm/ninfer-amd-r9700/tools/r9700/bf16_gdn_control_t1_qual.hip"
	.size	.L.str.6, 82

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"qualifier requires device 0"
	.size	.L.str.7, 28

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"hipSetDevice(0)"
	.size	.L.str.8, 16

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"hipGetDeviceProperties(&properties, 0)"
	.size	.L.str.9, 39

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"qualifier requires R9700 gfx1201 at device 0"
	.size	.L.str.12, 45

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0)"
	.size	.L.str.13, 56

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"0000:"
	.size	.L.str.14, 6

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"/sys/bus/pci/devices/"
	.size	.L.str.15, 22

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"/power_dpm_force_performance_level"
	.size	.L.str.16, 35

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	"performance admission requires device-0 power profile auto"
	.size	.L.str.18, 59

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"hipRuntimeGetVersion(&runtime_version)"
	.size	.L.str.19, 39

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"launch_serial(serial_args)"
	.size	.L.str.20, 27

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	"launch_combined(combined_args)"
	.size	.L.str.21, 31

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"launch_fused(fused_args)"
	.size	.L.str.22, 25

	.type	.L.str.23,@object               # @.str.23
.L.str.23:
	.asciz	"hipDeviceSynchronize()"
	.size	.L.str.23, 23

	.type	.L.str.24,@object               # @.str.24
.L.str.24:
	.asciz	"null input was not rejected"
	.size	.L.str.24, 28

	.type	.L.str.25,@object               # @.str.25
.L.str.25:
	.asciz	"output alias was not rejected"
	.size	.L.str.25, 30

	.type	.L.str.26,@object               # @.str.26
.L.str.26:
	.asciz	"input/output alias was not rejected"
	.size	.L.str.26, 36

	.type	.L.str.27,@object               # @.str.27
.L.str.27:
	.asciz	"hipMemset(eviction.get(), 0xa5, eviction.bytes())"
	.size	.L.str.27, 50

	.type	.L.str.28,@object               # @.str.28
.L.str.28:
	.asciz	"hipMemset(sink.get(), 0, sink.bytes())"
	.size	.L.str.28, 39

	.type	.L__const.main.orders,@object   # @__const.main.orders
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.L__const.main.orders:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	1                               # 0x1
	.size	.L__const.main.orders, 72

	.type	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm,@object # @_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm
	.section	.data.rel.ro,"aw",@progbits
	.p2align	3, 0x0
_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm:
	.quad	_ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm
	.size	_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm, 8

	.type	.L.str.29,@object               # @.str.29
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.29:
	.asciz	"hipGetLastError()"
	.size	.L.str.29, 18

	.type	.L.str.30,@object               # @.str.30
.L.str.30:
	.asciz	"power profile changed during timing"
	.size	.L.str.30, 36

	.type	.L.str.31,@object               # @.str.31
.L.str.31:
	.asciz	"PASS wrote "
	.size	.L.str.31, 12

	.type	.L.str.32,@object               # @.str.32
.L.str.32:
	.asciz	"FAIL: "
	.size	.L.str.32, 7

	.type	.L.str.33,@object               # @.str.33
.L.str.33:
	.asciz	"basic_string: construction from null is not valid"
	.size	.L.str.33, 50

	.type	.L.str.34,@object               # @.str.34
.L.str.34:
	.asciz	"basic_string::_M_create"
	.size	.L.str.34, 24

	.type	.L.str.35,@object               # @.str.35
.L.str.35:
	.asciz	": "
	.size	.L.str.35, 3

	.type	.L.str.37,@object               # @.str.37
.L.str.37:
	.asciz	"basic_string::_M_replace"
	.size	.L.str.37, 25

	.type	.L.str.39,@object               # @.str.39
.L.str.39:
	.asciz	"basic_string::append"
	.size	.L.str.39, 21

	.type	.L.str.40,@object               # @.str.40
.L.str.40:
	.asciz	"cannot create std::vector larger than max_size()"
	.size	.L.str.40, 49

	.type	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_,@object # @_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.section	.data.rel.ro,"aw",@progbits
	.p2align	3, 0x0
_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_:
	.quad	_ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.size	_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_, 8

	.type	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_,@object # @_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.p2align	3, 0x0
_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_:
	.quad	_ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.size	_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_, 8

	.type	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_,@object # @_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.p2align	3, 0x0
_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_:
	.quad	_ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.size	_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_, 8

	.type	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_,@object # @_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.p2align	3, 0x0
_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_:
	.quad	_ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.size	_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_, 8

	.type	.L.str.41,@object               # @.str.41
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.41:
	.asciz	"candidate differs bitwise from serial BF16/control boundaries"
	.size	.L.str.41, 62

	.type	.L.str.42,@object               # @.str.42
.L.str.42:
	.asciz	"projection output exceeds FP64 tolerance"
	.size	.L.str.42, 41

	.type	.L.str.43,@object               # @.str.43
.L.str.43:
	.asciz	"control output exceeds FP64 tolerance"
	.size	.L.str.43, 38

	.type	.L.str.44,@object               # @.str.44
.L.str.44:
	.asciz	"hipMemcpy(host.data(), output.storage.get(), output.storage.bytes(), hipMemcpyDeviceToHost)"
	.size	.L.str.44, 92

	.type	.L.str.45,@object               # @.str.45
.L.str.45:
	.asciz	"BF16 canary changed"
	.size	.L.str.45, 20

	.type	.L.str.46,@object               # @.str.46
.L.str.46:
	.asciz	"FP32 canary changed"
	.size	.L.str.46, 20

	.type	.L.str.47,@object               # @.str.47
.L.str.47:
	.asciz	"vector::_M_realloc_insert"
	.size	.L.str.47, 26

	.type	.L.str.48,@object               # @.str.48
.L.str.48:
	.asciz	"{\n  \"schema\": \"ninfer.r9700-bf16-gdn-control-t1-qualification.v1\",\n"
	.size	.L.str.48, 68

	.type	.L.str.49,@object               # @.str.49
.L.str.49:
	.asciz	"  \"status\": \"pass\",\n  \"power_profile_before\": \""
	.size	.L.str.49, 48

	.type	.L.str.50,@object               # @.str.50
.L.str.50:
	.asciz	"\",\n"
	.size	.L.str.50, 4

	.type	.L.str.51,@object               # @.str.51
.L.str.51:
	.asciz	"  \"platform\": {\"device_name\": \""
	.size	.L.str.51, 32

	.type	.L.str.52,@object               # @.str.52
.L.str.52:
	.asciz	"\", \"gcn_arch\": \""
	.size	.L.str.52, 17

	.type	.L.str.53,@object               # @.str.53
.L.str.53:
	.asciz	"\", \"pci_domain\": "
	.size	.L.str.53, 18

	.type	.L.str.54,@object               # @.str.54
.L.str.54:
	.asciz	", \"pci_bus\": "
	.size	.L.str.54, 14

	.type	.L.str.55,@object               # @.str.55
.L.str.55:
	.asciz	", \"pci_device\": "
	.size	.L.str.55, 17

	.type	.L.str.56,@object               # @.str.56
.L.str.56:
	.asciz	", \"hip_runtime_version\": "
	.size	.L.str.56, 26

	.type	.L.str.57,@object               # @.str.57
.L.str.57:
	.asciz	", \"rocm_hipcc_version\": \""
	.size	.L.str.57, 26

	.type	.L.str.58,@object               # @.str.58
.L.str.58:
	.asciz	"/opt/rocm/bin/hipcc --version 2>&1"
	.size	.L.str.58, 35

	.type	.L.str.59,@object               # @.str.59
.L.str.59:
	.asciz	"\"},\n"
	.size	.L.str.59, 5

	.type	.L.str.60,@object               # @.str.60
.L.str.60:
	.asciz	"  \"identities\": {\"executable_path\": \""
	.size	.L.str.60, 38

	.type	.L.str.61,@object               # @.str.61
.L.str.61:
	.asciz	"\", \"executable_sha256\": \""
	.size	.L.str.61, 26

	.type	.L.str.62,@object               # @.str.62
.L.str.62:
	.asciz	"\", \"source_path\": \"tools/r9700/bf16_gdn_control_t1_qual.hip\", \"source_sha256\": \""
	.size	.L.str.62, 81

	.type	.L.str.63,@object               # @.str.63
.L.str.63:
	.asciz	"tools/r9700/bf16_gdn_control_t1_qual.hip"
	.size	.L.str.63, 41

	.type	.L.str.64,@object               # @.str.64
.L.str.64:
	.asciz	"\", \"checker_path\": \"tools/r9700/check_bf16_gdn_control_t1_static.py\", \"checker_sha256\": \""
	.size	.L.str.64, 90

	.type	.L.str.65,@object               # @.str.65
.L.str.65:
	.asciz	"tools/r9700/check_bf16_gdn_control_t1_static.py"
	.size	.L.str.65, 48

	.type	.L.str.66,@object               # @.str.66
.L.str.66:
	.asciz	"\", \"assembly_path\": \""
	.size	.L.str.66, 22

	.type	.L.str.67,@object               # @.str.67
.L.str.67:
	.asciz	"\", \"assembly_sha256\": \""
	.size	.L.str.67, 24

	.type	.L.str.68,@object               # @.str.68
.L.str.68:
	.asciz	"\", \"static_receipt_path\": \""
	.size	.L.str.68, 28

	.type	.L.str.69,@object               # @.str.69
.L.str.69:
	.asciz	"\", \"static_receipt_sha256\": \""
	.size	.L.str.69, 30

	.type	.L.str.70,@object               # @.str.70
.L.str.70:
	.asciz	"\", \"static_receipt\": \""
	.size	.L.str.70, 23

	.type	.L.str.71,@object               # @.str.71
.L.str.71:
	.asciz	"  \"contract\": {\"tokens\": 1, \"heads\": 48, \"columns\": 5120, \"layers_per_token\": 48, \"weight_format\": \"BF16\", \"intermediate_boundary\": \"round_each_projection_once_to_BF16_before_control\"},\n"
	.size	.L.str.71, 187

	.type	.L.str.72,@object               # @.str.72
.L.str.72:
	.asciz	"  \"layer0\": {\"classification\": \"launch_and_cross_op_materialization\", "
	.size	.L.str.72, 71

	.type	.L.str.73,@object               # @.str.73
.L.str.73:
	.asciz	"\"accounting_scope\": \"logical_Op_boundary_not_physical_CTA_transactions\", "
	.size	.L.str.73, 74

	.type	.L.str.74,@object               # @.str.74
.L.str.74:
	.asciz	"\"terms_bytes\": {\"two_weights\": "
	.size	.L.str.74, 32

	.type	.L.str.75,@object               # @.str.75
.L.str.75:
	.asciz	", \"one_input\": "
	.size	.L.str.75, 16

	.type	.L.str.76,@object               # @.str.76
.L.str.76:
	.asciz	", \"a_b_intermediates\": "
	.size	.L.str.76, 24

	.type	.L.str.77,@object               # @.str.77
.L.str.77:
	.asciz	", \"control_parameters\": "
	.size	.L.str.77, 25

	.type	.L.str.78,@object               # @.str.78
.L.str.78:
	.asciz	", \"g_beta_results\": "
	.size	.L.str.78, 21

	.type	.L.str.79,@object               # @.str.79
.L.str.79:
	.asciz	"}, \"incumbent_formula\": \"weights + 2*input + projection_writes + control_reads + parameters + results\", "
	.size	.L.str.79, 105

	.type	.L.str.80,@object               # @.str.80
.L.str.80:
	.asciz	"\"incumbent_bytes\": "
	.size	.L.str.80, 20

	.type	.L.str.81,@object               # @.str.81
.L.str.81:
	.asciz	", \"combined_formula\": \"weights + input + projection_writes + control_reads + parameters + results\", "
	.size	.L.str.81, 101

	.type	.L.str.82,@object               # @.str.82
.L.str.82:
	.asciz	"\"combined_bytes\": "
	.size	.L.str.82, 19

	.type	.L.str.83,@object               # @.str.83
.L.str.83:
	.asciz	", \"fused_formula\": \"weights + input + projection_writes + parameters + results\", "
	.size	.L.str.83, 82

	.type	.L.str.84,@object               # @.str.84
.L.str.84:
	.asciz	"\"fused_bytes\": "
	.size	.L.str.84, 16

	.type	.L.str.85,@object               # @.str.85
.L.str.85:
	.asciz	", \"fused_avoidable_bytes\": 10432},\n"
	.size	.L.str.85, 36

	.type	.L.str.86,@object               # @.str.86
.L.str.86:
	.asciz	"  \"correctness\": {\"projection_fp64_max_abs\": "
	.size	.L.str.86, 46

	.type	.L.str.87,@object               # @.str.87
.L.str.87:
	.asciz	", \"control_fp64_max_abs\": "
	.size	.L.str.87, 27

	.type	.L.str.88,@object               # @.str.88
.L.str.88:
	.asciz	", \"candidate_serial_bit_exact\": true, \"canaries\": \"pass\", \"malformed_and_alias_rejection\": \"pass\"},\n"
	.size	.L.str.88, 101

	.type	.L.str.89,@object               # @.str.89
.L.str.89:
	.asciz	"  \"timing_ms_per_layer\": {\"serial_median\": "
	.size	.L.str.89, 44

	.type	.L.str.90,@object               # @.str.90
.L.str.90:
	.asciz	", \"combined_median\": "
	.size	.L.str.90, 22

	.type	.L.str.91,@object               # @.str.91
.L.str.91:
	.asciz	", \"fused_median\": "
	.size	.L.str.91, 19

	.type	.L.str.92,@object               # @.str.92
.L.str.92:
	.asciz	"},\n"
	.size	.L.str.92, 4

	.type	.L.str.93,@object               # @.str.93
.L.str.93:
	.asciz	"  \"whole_token_bound\": {\"best_possible_observed_saving_ms\": "
	.size	.L.str.93, 61

	.type	.L.str.94,@object               # @.str.94
.L.str.94:
	.asciz	", \"material_threshold_ms\": 0.200000000, "
	.size	.L.str.94, 41

	.type	.L.str.95,@object               # @.str.95
.L.str.95:
	.asciz	"\"admitted\": "
	.size	.L.str.95, 13

	.type	.L.str.96,@object               # @.str.96
.L.str.96:
	.asciz	"true"
	.size	.L.str.96, 5

	.type	.L.str.97,@object               # @.str.97
.L.str.97:
	.asciz	"false"
	.size	.L.str.97, 6

	.type	.L.str.98,@object               # @.str.98
.L.str.98:
	.asciz	"  \"schedule\": {\"trials\": 12, \"orders\": [[0,1,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[0,2,1]], "
	.size	.L.str.98, 91

	.type	.L.str.99,@object               # @.str.99
.L.str.99:
	.asciz	"\"route_names\": [\"serial\",\"combined\",\"fused\"], \"weight_copies\": 3, \"samples_per_route_copy\": 4, "
	.size	.L.str.99, 96

	.type	.L.str.100,@object              # @.str.100
.L.str.100:
	.asciz	"\"eviction_bytes_before_each_event\": 100663296},\n"
	.size	.L.str.100, 49

	.type	.L.str.101,@object              # @.str.101
.L.str.101:
	.asciz	"  \"samples_ms\": {\n"
	.size	.L.str.101, 19

	.type	.L.str.102,@object              # @.str.102
.L.str.102:
	.asciz	"serial"
	.size	.L.str.102, 7

	.type	.L.str.103,@object              # @.str.103
.L.str.103:
	.asciz	"combined"
	.size	.L.str.103, 9

	.type	.L.str.104,@object              # @.str.104
.L.str.104:
	.asciz	"fused"
	.size	.L.str.104, 6

	.type	.L__const._ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE.names,@object # @__const._ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE.names
	.section	.data.rel.ro,"aw",@progbits
	.p2align	4, 0x0
.L__const._ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE.names:
	.quad	.L.str.102
	.quad	.L.str.103
	.quad	.L.str.104
	.size	.L__const._ZN12_GLOBAL__N_110write_jsonERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_S7_S7_S7_RK20hipDeviceProp_tR0600iRKNS_11CorrectnessERKSt5arrayISt6vectorIfSaIfEELm3EERKSF_INS_11TrialRecordESaISL_EE.names, 24

	.type	.L.str.105,@object              # @.str.105
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.105:
	.asciz	"    \""
	.size	.L.str.105, 6

	.type	.L.str.106,@object              # @.str.106
.L.str.106:
	.asciz	"\": ["
	.size	.L.str.106, 5

	.type	.L.str.107,@object              # @.str.107
.L.str.107:
	.asciz	", "
	.size	.L.str.107, 3

	.type	.L.str.108,@object              # @.str.108
.L.str.108:
	.asciz	"]"
	.size	.L.str.108, 2

	.type	.L.str.109,@object              # @.str.109
.L.str.109:
	.asciz	"\n"
	.size	.L.str.109, 2

	.type	.L.str.110,@object              # @.str.110
.L.str.110:
	.asciz	",\n"
	.size	.L.str.110, 3

	.type	.L.str.111,@object              # @.str.111
.L.str.111:
	.asciz	"  },\n  \"per_copy_medians_ms\": {\n"
	.size	.L.str.111, 33

	.type	.L.str.112,@object              # @.str.112
.L.str.112:
	.asciz	"route/copy schedule is not balanced"
	.size	.L.str.112, 36

	.type	.L.str.113,@object              # @.str.113
.L.str.113:
	.asciz	"  },\n  \"trial_records\": [\n"
	.size	.L.str.113, 27

	.type	.L.str.114,@object              # @.str.114
.L.str.114:
	.asciz	"    {\"trial\": "
	.size	.L.str.114, 15

	.type	.L.str.115,@object              # @.str.115
.L.str.115:
	.asciz	", \"order\": "
	.size	.L.str.115, 12

	.type	.L.str.116,@object              # @.str.116
.L.str.116:
	.asciz	", \"position\": "
	.size	.L.str.116, 15

	.type	.L.str.117,@object              # @.str.117
.L.str.117:
	.asciz	", \"route\": "
	.size	.L.str.117, 12

	.type	.L.str.118,@object              # @.str.118
.L.str.118:
	.asciz	", \"copy\": "
	.size	.L.str.118, 11

	.type	.L.str.119,@object              # @.str.119
.L.str.119:
	.asciz	", \"milliseconds\": "
	.size	.L.str.119, 19

	.type	.L.str.120,@object              # @.str.120
.L.str.120:
	.asciz	"}"
	.size	.L.str.120, 2

	.type	.L.str.121,@object              # @.str.121
.L.str.121:
	.asciz	"  ],\n  \"power_profile_after\": \""
	.size	.L.str.121, 32

	.type	.L.str.122,@object              # @.str.122
.L.str.122:
	.asciz	"\"\n}\n"
	.size	.L.str.122, 5

	.type	.L.str.123,@object              # @.str.123
.L.str.123:
	.asciz	"/proc/self/exe"
	.size	.L.str.123, 15

	.type	.L.str.124,@object              # @.str.124
.L.str.124:
	.asciz	"cannot resolve executable path"
	.size	.L.str.124, 31

	.type	.L.str.125,@object              # @.str.125
.L.str.125:
	.asciz	"\\n"
	.size	.L.str.125, 3

	.type	.L.str.126,@object              # @.str.126
.L.str.126:
	.asciz	"r"
	.size	.L.str.126, 2

	.type	.L.str.127,@object              # @.str.127
.L.str.127:
	.asciz	"cannot execute identity command"
	.size	.L.str.127, 32

	.type	.L.str.128,@object              # @.str.128
.L.str.128:
	.asciz	"identity command failed"
	.size	.L.str.128, 24

	.type	.L.str.129,@object              # @.str.129
.L.str.129:
	.asciz	"/usr/bin/sha256sum -- "
	.size	.L.str.129, 23

	.type	.L.str.130,@object              # @.str.130
.L.str.130:
	.asciz	"malformed sha256 for "
	.size	.L.str.130, 22

	.type	.L.str.131,@object              # @.str.131
.L.str.131:
	.asciz	"'"
	.size	.L.str.131, 2

	.type	.L.str.135,@object              # @.str.135
.L.str.135:
	.asciz	"cannot read "
	.size	.L.str.135, 13

	.type	.L.str.136,@object              # @.str.136
.L.str.136:
	.asciz	"create-only report open failed: "
	.size	.L.str.136, 33

	.type	.L.str.137,@object              # @.str.137
.L.str.137:
	.asciz	"report write failed: "
	.size	.L.str.137, 22

	.type	.L.str.138,@object              # @.str.138
.L.str.138:
	.asciz	"report close failed"
	.size	.L.str.138, 20

	.type	.L.str.139,@object              # @.str.139
.L.str.139:
	.asciz	"hipMalloc(&data_, count * sizeof(T))"
	.size	.L.str.139, 37

	.type	.L.str.140,@object              # @.str.140
.L.str.140:
	.asciz	"hipMemcpy(destination.get(), source.data(), source.size() * sizeof(T), hipMemcpyHostToDevice)"
	.size	.L.str.140, 94

	.type	.L.str.141,@object              # @.str.141
.L.str.141:
	.asciz	"hipEventRecord(begin.get())"
	.size	.L.str.141, 28

	.type	.L.str.142,@object              # @.str.142
.L.str.142:
	.asciz	"launch()"
	.size	.L.str.142, 9

	.type	.L.str.143,@object              # @.str.143
.L.str.143:
	.asciz	"hipEventRecord(end.get())"
	.size	.L.str.143, 26

	.type	.L.str.144,@object              # @.str.144
.L.str.144:
	.asciz	"hipEventSynchronize(end.get())"
	.size	.L.str.144, 31

	.type	.L.str.145,@object              # @.str.145
.L.str.145:
	.asciz	"hipEventElapsedTime(&ms, begin.get(), end.get())"
	.size	.L.str.145, 49

	.type	.L.str.146,@object              # @.str.146
.L.str.146:
	.asciz	"hipEventCreate(&value_)"
	.size	.L.str.146, 24

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_"
	.size	.L__unnamed_1, 71

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_"
	.size	.L__unnamed_2, 75

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_"
	.size	.L__unnamed_3, 76

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_"
	.size	.L__unnamed_4, 92

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_ZN12_GLOBAL__N_115eviction_kernelEPKjPjm"
	.size	.L__unnamed_5, 42

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_57a29556a1d631fd
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_57a29556a1d631fd,@object # @__hip_gpubin_handle_57a29556a1d631fd
	.local	__hip_gpubin_handle_57a29556a1d631fd
	.comm	__hip_gpubin_handle_57a29556a1d631fd,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_57a29556a1d631fd,@object # @__hip_cuid_57a29556a1d631fd
	.bss
	.globl	__hip_cuid_57a29556a1d631fd
__hip_cuid_57a29556a1d631fd:
	.byte	0                               # 0x0
	.size	__hip_cuid_57a29556a1d631fd, 1

	.data
	.p2align	3, 0x0
.L_ZTISt9exception.DW.stub:
	.quad	_ZTISt9exception
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
	.addrsig_sym _ZN12_GLOBAL__N_130__device_stub__eviction_kernelEPKjPjm
	.addrsig_sym _ZN12_GLOBAL__N_142__device_stub__incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.addrsig_sym _ZN12_GLOBAL__N_139__device_stub__incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.addrsig_sym _ZN12_GLOBAL__N_141__device_stub__combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.addrsig_sym _ZN12_GLOBAL__N_146__device_stub__fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZTISt9exception
	.addrsig_sym _ZSt4cerr
	.addrsig_sym _ZN12_GLOBAL__N_115eviction_kernelEPKjPjm
	.addrsig_sym _ZSt4cout
	.addrsig_sym _ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE
	.addrsig_sym _ZTVSt15basic_streambufIcSt11char_traitsIcEE
	.addrsig_sym _ZTISt13runtime_error
	.addrsig_sym _ZN12_GLOBAL__N_127incumbent_projection_kernelEPK12hip_bfloat16S2_PS0_
	.addrsig_sym _ZN12_GLOBAL__N_124incumbent_control_kernelEPK12hip_bfloat16S2_PKfS4_PfS5_
	.addrsig_sym _ZN12_GLOBAL__N_126combined_projection_kernelEPK12hip_bfloat16S2_S2_PS0_S3_
	.addrsig_sym _ZN12_GLOBAL__N_131fused_projection_control_kernelEPK12hip_bfloat16S2_S2_PKfS4_PS0_S5_PfS6_
	.addrsig_sym __hip_fatbin_57a29556a1d631fd
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_57a29556a1d631fd

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
