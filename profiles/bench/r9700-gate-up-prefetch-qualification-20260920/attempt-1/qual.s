	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	a8q4_gate_up_prefetch_qualification_kernel ; -- Begin function a8q4_gate_up_prefetch_qualification_kernel
	.globl	a8q4_gate_up_prefetch_qualification_kernel
	.p2align	8
	.type	a8q4_gate_up_prefetch_qualification_kernel,@function
a8q4_gate_up_prefetch_qualification_kernel: ; @a8q4_gate_up_prefetch_qualification_kernel
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_lshl_or_b32 v1, ttmp9, 8, v0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e32 0x8800, v1
	s_cbranch_execz .LBB0_10
; %bb.1:
	s_load_b256 s[8:15], s[0:1], 0x0
	v_mov_b16_e32 v3.h, 0x7fc1
	s_wait_kmcnt 0x0
	s_load_b32 s14, s[14:15], 0x0
	s_load_b256 s[0:7], s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_mov_b32 s7, 0
	s_cmp_lg_u32 s14, 0
	s_cbranch_scc1 .LBB0_9
; %bb.2:
	v_lshrrev_b32_e32 v11, 4, v1
	v_and_b32_e32 v12, 15, v0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v0, 0
	s_mov_b32 s6, s7
	v_mul_lo_u32 v2, 0x1400, v11
	v_mul_lo_u32 v13, 0x50, v11
	v_lshlrev_b32_e32 v14, 1, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v7, v2, v12
	v_lshlrev_b64_e32 v[2:3], 3, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s0, v2
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	s_clause 0x3
	global_load_b64 v[9:10], v[2:3], off
	global_load_b64 v[6:7], v[2:3], off offset:128
	global_load_b64 v[4:5], v[2:3], off offset:256
	global_load_b64 v[2:3], v[2:3], off offset:384
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v23, s6, v13
	s_lshl_b64 s[14:15], s[6:7], 1
	s_wait_loadcnt 0x2
	v_mov_b32_e32 v18, v7
	s_add_nc_u64 s[14:15], s[12:13], s[14:15]
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v17, v6 :: v_dual_mov_b32 v20, v5
	v_lshl_or_b32 v15, v23, 5, v14
	global_load_d16_b16 v24, v8, s[14:15]
	global_load_d16_b16 v25, v15, s[2:3]
	v_dual_mov_b32 v16, v10 :: v_dual_mov_b32 v15, v9
	s_wait_loadcnt 0x2
	v_dual_mov_b32 v19, v4 :: v_dual_mov_b32 v22, v3
	v_mov_b32_e32 v21, v2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v24, v24, v25, neg(0) op_sel_hi:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_lshl_or_b32 v2, v23, 6, v12
	s_add_co_i32 s33, s6, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v7, 64, v2
	v_lshlrev_b64_e32 v[2:3], 3, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	s_clause 0x3
	global_load_b64 v[9:10], v[2:3], off
	global_load_b64 v[6:7], v[2:3], off offset:128
	global_load_b64 v[4:5], v[2:3], off offset:256
	global_load_b64 v[2:3], v[2:3], off offset:384
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s6, s6, 3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[14:15], s[6:7], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[16:17], s[8:9], s[14:15]
	s_add_nc_u64 s[14:15], s[10:11], s[14:15]
	s_load_b256 s[16:23], s[16:17], 0x0
	s_load_b256 s[24:31], s[14:15], 0x0
	s_wait_kmcnt 0x0
	v_dot8_i32_iu4 v23, s16, v15, 0 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s24, v15, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v23, s17, v16, v23 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s25, v16, v15 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v16, s18, v17, v23 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s26, v17, v15 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v16, s19, v18, v16 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s27, v18, v15 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v16, s20, v19, v16 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s28, v19, v15 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v16, s21, v20, v16 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s29, v20, v15 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v16, s22, v21, v16 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s30, v21, v15 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v16, s23, v22, v16 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v15, s31, v22, v15 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v15, v15, 4, v16
	v_cvt_f32_i32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v0, v15, v24
	; sched_barrier mask(0x00000000)
	s_cmp_eq_u32 s33, 0x4f
	s_mov_b32 s6, s33
	s_cbranch_scc0 .LBB0_3
; %bb.4:
	v_mul_lo_u32 v8, 0x500, v11
	v_mov_b32_e32 v13, 0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v8, v12
	v_lshlrev_b64_e32 v[11:12], 1, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, s2, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s3, v12, vcc_lo
	global_load_d16_b16 v8, v13, s[12:13] offset:158
	global_load_d16_b16 v11, v[11:12], off offset:2528
	s_load_b256 s[12:19], s[8:9], 0x9e0
	s_load_b256 s[20:27], s[10:11], 0x9e0
	s_wait_loadcnt 0x5
	s_wait_kmcnt 0x0
	v_dot8_i32_iu4 v12, s12, v9, 0 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s20, v9, 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v12, s13, v10, v12 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v9, s21, v10, v9 neg_lo:[1,1,0]
	s_wait_loadcnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v10, s14, v6, v12 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v6, s22, v6, v9 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v9, s15, v7, v10 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v6, s23, v7, v6 neg_lo:[1,1,0]
	s_wait_loadcnt 0x3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v7, s16, v4, v9 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v4, s24, v4, v6 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v6, s17, v5, v7 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v4, s25, v5, v4 neg_lo:[1,1,0]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v5, s18, v2, v6 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v2, s26, v2, v4 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot8_i32_iu4 v4, s19, v3, v5 neg_lo:[0,1,0]
	v_dot8_i32_iu4 v2, s27, v3, v2 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v2, v2, 4, v4
	v_cvt_f32_i32_e32 v2, v2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v3, v8, v11, neg(0) op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v0, v2, v3
                                        ; implicit-def: $vgpr3
	v_and_b32_e32 v2, 0x7f800000, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v2
	s_xor_b32 s0, exec_lo, s0
; %bb.5:
	v_bfe_u32 v2, v0, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v3, v0, v2, 0x7fff
                                        ; implicit-def: $vgpr0
; %bb.6:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.7:
	v_or_b32_e32 v3, 0x10000, v0
	v_and_b32_e32 v2, 0xffff, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v3, v0, vcc_lo
; %bb.8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.LBB0_9:
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[1:2]
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v3, off
.LBB0_10:
	s_endpgm
.Lfunc_end0:
	.size	a8q4_gate_up_prefetch_qualification_kernel, .Lfunc_end0-a8q4_gate_up_prefetch_qualification_kernel
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel a8q4_gate_up_prefetch_qualification_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
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
		.amdhsa_next_free_vgpr 26
		.amdhsa_next_free_sgpr 34
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-a8q4_gate_up_prefetch_qualification_kernel)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .La8q4_gate_up_prefetch_qualification_kernel.num_vgpr, 26
	.set .La8q4_gate_up_prefetch_qualification_kernel.num_agpr, 0
	.set .La8q4_gate_up_prefetch_qualification_kernel.numbered_sgpr, 34
	.set .La8q4_gate_up_prefetch_qualification_kernel.num_named_barrier, 0
	.set .La8q4_gate_up_prefetch_qualification_kernel.private_seg_size, 0
	.set .La8q4_gate_up_prefetch_qualification_kernel.uses_vcc, 1
	.set .La8q4_gate_up_prefetch_qualification_kernel.uses_flat_scratch, 0
	.set .La8q4_gate_up_prefetch_qualification_kernel.has_dyn_sized_stack, 0
	.set .La8q4_gate_up_prefetch_qualification_kernel.has_recursion, 0
	.set .La8q4_gate_up_prefetch_qualification_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1112
; TotalNumSgprs: 36
; NumVgprs: 26
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 36
; NumVGPRsForWavesPerEU: 26
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_412d2160bddfabe3,@object ; @__hip_cuid_412d2160bddfabe3
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_412d2160bddfabe3
__hip_cuid_412d2160bddfabe3:
	.byte	0                               ; 0x0
	.size	__hip_cuid_412d2160bddfabe3, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_412d2160bddfabe3
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
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           a8q4_gate_up_prefetch_qualification_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         a8q4_gate_up_prefetch_qualification_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     26
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
