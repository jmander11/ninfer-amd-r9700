	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16,"axG",@progbits,_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16,comdat
	.globl	_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16 ; -- Begin function _ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16
	.p2align	8
	.type	_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16,@function
_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16: ; @_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshrrev_b32_e32 v41, 3, v0
	s_lshl_b32 s12, ttmp9, 7
	s_lshl_b32 s13, ttmp9, 4
	s_load_b64 s[2:3], s[0:1], 0x20
	s_and_b32 s1, s12, 0x780
	s_and_b32 s12, s13, 0xffffff00
	v_or_b32_e32 v1, s1, v41
	v_lshrrev_b32_e32 v2, 1, v0
	v_or_b32_e32 v5, s12, v41
	v_dual_mov_b32 v25, 0 :: v_dual_lshlrev_b32 v38, 2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_u32_u24_e32 v7, 0x1400, v1
	v_and_b32_e32 v39, 0x70, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[1:2], null, 0x1400, v5, 0
	v_dual_mov_b32 v29, v25 :: v_dual_and_b32 v8, 28, v38
	v_bfe_u32 v4, v0, 4, 1
	v_dual_mov_b32 v27, v25 :: v_dual_and_b32 v6, 15, v0
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, s0, s4, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s5, 0, s0
	v_or_b32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, vcc_lo, v5, v8
	v_add_co_ci_u32_e64 v34, null, 0, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s8, v1
	v_lshrrev_b32_e32 v3, 4, v0
	v_dual_mov_b32 v31, v25 :: v_dual_lshlrev_b32 v40, 3, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s9, v2, vcc_lo
	v_mov_b32_e32 v26, v25
	v_or_b32_e32 v9, v39, v6
	v_add_co_u32 v35, vcc_lo, 0xa0000, v1
	v_lshl_or_b32 v42, v41, 5, v8
	v_dual_mov_b32 v17, v25 :: v_dual_lshlrev_b32 v44, 11, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v43, v9, 5, v40
	v_and_or_b32 v37, v3, 48, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, 0, v2, vcc_lo
	v_dual_mov_b32 v28, v25 :: v_dual_lshlrev_b32 v45, 8, v8
	v_dual_mov_b32 v30, v25 :: v_dual_mov_b32 v19, v25
	v_dual_mov_b32 v32, v25 :: v_dual_mov_b32 v21, v25
	v_dual_mov_b32 v18, v25 :: v_dual_mov_b32 v23, v25
	v_dual_mov_b32 v20, v25 :: v_dual_mov_b32 v9, v25
	v_dual_mov_b32 v22, v25 :: v_dual_mov_b32 v11, v25
	v_dual_mov_b32 v24, v25 :: v_dual_mov_b32 v13, v25
	v_dual_mov_b32 v10, v25 :: v_dual_mov_b32 v15, v25
	v_dual_mov_b32 v12, v25 :: v_dual_mov_b32 v1, v25
	v_dual_mov_b32 v14, v25 :: v_dual_mov_b32 v3, v25
	v_dual_mov_b32 v16, v25 :: v_dual_mov_b32 v5, v25
	v_dual_mov_b32 v2, v25 :: v_dual_mov_b32 v7, v25
	v_mov_b32_e32 v4, v25
	v_mov_b32_e32 v6, v25
	v_mov_b32_e32 v8, v25
	s_mov_b32 s4, 0
	s_movk_i32 s5, 0xa0
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b32 v46, v[35:36], off offset:-655360
	global_load_b32 v47, v[35:36], off
	global_load_b32 v48, v[33:34], off
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s0, s4, 12
	s_lshl_b32 s8, s4, 13
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v50, s0, v42
	v_or_b32_e32 v49, s8, v44
	v_or3_b32 v51, s8, v45, v41
	v_or_b32_e32 v52, s0, v43
	v_add_co_u32 v33, vcc_lo, v33, 32
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v53, v49, v37
	v_add_co_u32 v35, s0, v35, 32
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, 0, v36, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, 0, v34, vcc_lo
	s_add_co_i32 s5, s5, -1
	s_xor_b32 s4, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, 0
	s_wait_loadcnt 0x2
	v_lshrrev_b32_e32 v49, 24, v46
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v54, 24, v47
	s_wait_loadcnt 0x0
	ds_store_b32 v50, v48
	v_lshrrev_b32_e32 v48, 8, v46
	v_lshrrev_b32_e32 v50, 8, v47
	ds_store_b8 v51, v46 offset:8192
	ds_store_b8 v51, v47 offset:8320
	ds_store_b8 v51, v48 offset:8448
	ds_store_b8_d16_hi v51, v46 offset:8704
	ds_store_b8_d16_hi v51, v47 offset:8832
	ds_store_b8 v51, v49 offset:8960
	ds_store_b8 v51, v50 offset:8576
	ds_store_b8 v51, v54 offset:9088
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_u8 v50, v53 offset:8704
	ds_load_u8 v51, v53 offset:8960
	ds_load_u8 v54, v53 offset:9728
	ds_load_u8 v55, v53 offset:9984
	ds_load_2addr_b64 v[46:49], v52 offset1:2
	ds_load_u8 v52, v53 offset:12800
	ds_load_u8 v56, v53 offset:13056
	ds_load_u8 v57, v53 offset:13824
	ds_load_u8 v58, v53 offset:14080
	ds_load_u8 v59, v53 offset:13312
	ds_load_u8 v60, v53 offset:13568
	ds_load_u8 v61, v53 offset:13376
	ds_load_u8 v62, v53 offset:13632
	ds_load_u8 v63, v53 offset:12288
	ds_load_u8 v64, v53 offset:12544
	ds_load_u8 v65, v53 offset:12352
	ds_load_u8 v66, v53 offset:12608
	ds_load_u8 v67, v53 offset:12416
	ds_load_u8 v68, v53 offset:12672
	ds_load_u8 v69, v53 offset:8192
	ds_load_u8 v70, v53 offset:8448
	ds_load_u8 v71, v53 offset:8256
	ds_load_u8 v72, v53 offset:8512
	ds_load_u8 v73, v53 offset:8320
	ds_load_u8 v74, v53 offset:8576
	ds_load_u8 v75, v53 offset:8640
	ds_load_u8 v76, v53 offset:8384
	ds_load_u8 v77, v53 offset:9216
	ds_load_u8 v78, v53 offset:9472
	ds_load_u8 v79, v53 offset:9280
	ds_load_u8 v80, v53 offset:9536
	ds_load_u8 v81, v53 offset:9344
	ds_load_u8 v82, v53 offset:9600
	ds_load_u8 v83, v53 offset:9664
	ds_load_u8 v84, v53 offset:9408
	ds_load_u8 v85, v53 offset:8768
	ds_load_u8 v86, v53 offset:9024
	ds_load_u8 v87, v53 offset:8832
	ds_load_u8 v88, v53 offset:9088
	ds_load_u8 v89, v53 offset:9152
	ds_load_u8 v90, v53 offset:8896
	ds_load_u8 v91, v53 offset:9792
	ds_load_u8 v92, v53 offset:10048
	ds_load_u8 v93, v53 offset:9856
	ds_load_u8 v94, v53 offset:10112
	ds_load_u8 v95, v53 offset:10176
	s_wait_dscnt 0x19
	v_lshl_or_b32 v69, v70, 8, v69
	v_lshlrev_b32_e32 v50, 16, v50
	v_lshlrev_b32_e32 v51, 24, v51
	s_wait_dscnt 0x11
	v_lshl_or_b32 v77, v78, 8, v77
	ds_load_u8 v70, v53 offset:12736
	v_lshl_or_b32 v63, v64, 8, v63
	ds_load_u8 v64, v53 offset:13504
	v_or3_b32 v50, v69, v50, v51
	v_lshlrev_b32_e32 v51, 16, v54
	v_lshlrev_b32_e32 v54, 24, v55
	v_lshl_or_b32 v59, v60, 8, v59
	ds_load_u8 v60, v53 offset:12864
	ds_load_u8 v69, v53 offset:9920
	v_lshl_or_b32 v71, v72, 8, v71
	v_or3_b32 v51, v77, v51, v54
	ds_load_u8 v54, v53 offset:13440
	ds_load_u8 v55, v53 offset:13696
	ds_load_u8 v77, v53 offset:13760
	ds_load_u8 v72, v53 offset:13888
	s_wait_dscnt 0x17
	v_lshl_or_b32 v79, v80, 8, v79
	ds_load_u8 v80, v53 offset:13120
	v_lshl_or_b32 v65, v66, 8, v65
	ds_load_u8 v66, v53 offset:14144
	v_lshl_or_b32 v61, v62, 8, v61
	ds_load_u8 v62, v53 offset:12928
	v_lshl_or_b32 v73, v74, 8, v73
	s_wait_dscnt 0x18
	v_lshl_or_b32 v81, v82, 8, v81
	ds_load_u8 v82, v53 offset:13184
	v_lshl_or_b32 v67, v68, 8, v67
	ds_load_u8 v68, v53 offset:14208
	ds_load_u8 v74, v53 offset:13952
	v_lshl_or_b32 v75, v75, 8, v76
	s_wait_dscnt 0x19
	v_lshl_or_b32 v76, v83, 8, v84
	v_lshlrev_b32_e32 v52, 16, v52
	v_lshlrev_b32_e32 v56, 24, v56
	v_lshlrev_b32_e32 v57, 16, v57
	s_wait_dscnt 0x8
	v_lshl_or_b32 v96, v55, 8, v54
	ds_load_u8 v54, v53 offset:13248
	ds_load_u8 v55, v53 offset:12992
	ds_load_u8 v78, v53 offset:12480
	ds_load_u8 v83, v53 offset:14272
	ds_load_u8 v53, v53 offset:14016
	v_lshlrev_b32_e32 v58, 24, v58
	s_wait_dscnt 0xc
	v_lshl_or_b32 v77, v77, 8, v64
	v_lshlrev_b32_e32 v64, 16, v85
	v_lshlrev_b32_e32 v84, 16, v91
	v_lshlrev_b32_e32 v85, 24, v92
	v_lshlrev_b32_e32 v60, 16, v60
	s_wait_dscnt 0xa
	v_lshlrev_b32_e32 v80, 24, v80
	v_lshlrev_b32_e32 v88, 24, v88
	v_lshlrev_b32_e32 v91, 24, v94
	s_wait_dscnt 0x8
	v_lshlrev_b32_e32 v62, 16, v62
	s_wait_dscnt 0x7
	v_lshlrev_b32_e32 v82, 24, v82
	v_lshlrev_b32_e32 v90, 16, v90
	v_lshlrev_b32_e32 v69, 16, v69
	v_lshlrev_b32_e32 v89, 24, v89
	v_lshlrev_b32_e32 v92, 24, v95
	v_lshlrev_b32_e32 v72, 16, v72
	v_lshlrev_b32_e32 v66, 24, v66
	s_wait_dscnt 0x5
	v_lshlrev_b32_e32 v74, 16, v74
	s_wait_dscnt 0x2
	v_lshl_or_b32 v70, v70, 8, v78
	v_lshlrev_b32_e32 v78, 24, v86
	v_lshlrev_b32_e32 v86, 16, v87
	v_lshlrev_b32_e32 v87, 16, v93
	v_lshlrev_b32_e32 v68, 24, v68
	v_lshlrev_b32_e32 v93, 16, v55
	s_wait_dscnt 0x0
	v_lshlrev_b32_e32 v94, 16, v53
	v_lshlrev_b32_e32 v95, 24, v54
	v_lshlrev_b32_e32 v83, 24, v83
	v_or3_b32 v52, v63, v52, v56
	v_or3_b32 v53, v59, v57, v58
	v_or3_b32 v54, v71, v64, v78
	v_or3_b32 v55, v79, v84, v85
	v_or3_b32 v56, v65, v60, v80
	v_or3_b32 v58, v73, v86, v88
	v_or3_b32 v59, v81, v87, v91
	v_or3_b32 v60, v67, v62, v82
	v_or3_b32 v62, v75, v90, v89
	v_or3_b32 v63, v76, v69, v92
	v_or3_b32 v57, v61, v72, v66
	v_or3_b32 v61, v96, v74, v68
	v_or3_b32 v64, v70, v93, v95
	v_or3_b32 v65, v77, v94, v83
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[46:47], v[50:51], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[46:47], v[54:55], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[46:47], v[58:59], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[46:47], v[62:63], v[1:8]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[48:49], v[52:53], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[48:49], v[56:57], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[48:49], v[60:61], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[48:49], v[64:65], v[1:8]
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB0_1
; %bb.2:
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e32 0x80, v0
	s_cbranch_execz .LBB0_4
; %bb.3:
	v_or_b32_e32 v33, s1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 2, v33
	global_load_b32 v33, v33, s[6:7]
	s_wait_loadcnt 0x0
	ds_store_b32 v38, v33
.LBB0_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB0_6
; %bb.5:
	v_or_b32_e32 v33, s12, v0
	v_mov_b32_e32 v34, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, vcc_lo, s10, v33
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, s11, v34, vcc_lo
	global_load_b32 v0, v[33:34], off
	s_wait_loadcnt 0x0
	ds_store_b32 v38, v0 offset:512
.LBB0_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v35, v39, v40
	v_lshlrev_b32_e32 v0, 2, v37
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr39
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v36, 2, v35
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v33, v36
	ds_load_b32 v38, v0 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v25, v25, v33
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v25, v38, v25
	v_and_b32_e32 v34, 0x7f800000, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v34
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.7:
	v_bfe_u32 v34, v25, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v39, v25, v34, 0x7fff
                                        ; implicit-def: $vgpr25
; %bb.8:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.9:
	v_or_b32_e32 v39, 0x10000, v25
	v_and_b32_e32 v34, 0xffff, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v34
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v39, v39, v25, vcc_lo
; %bb.10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v34, v36 offset:4
	s_mov_b32 s13, 0
	v_or_b32_e32 v25, s1, v35
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[12:13], 1
	v_lshlrev_b32_e32 v37, 1, v37
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]
	v_mul_u32_u24_e32 v25, 0x11000, v25
	s_wait_dscnt 0x0
	v_mul_f32_e32 v26, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v35, v38, v26
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v26, s0, s0, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v37, null, s1, 0, s0
	v_and_b32_e32 v40, 0x7f800000, v35
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v25, vcc_lo, v26, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, 0, v37, vcc_lo
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr37
	global_store_d16_hi_b16 v[25:26], v39, off
	v_cmpx_ne_u32_e32 0x7f800000, v40
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.11:
	v_bfe_u32 v37, v35, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v37, v35, v37, 0x7fff
                                        ; implicit-def: $vgpr35
; %bb.12:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.13:
	v_and_b32_e32 v37, 0xffff, v35
	v_or_b32_e32 v39, 0x10000, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v39, v35, vcc_lo
; %bb.14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v35, v36 offset:8
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v37, off offset:69632
                                        ; implicit-def: $vgpr37
	s_wait_dscnt 0x0
	v_mul_f32_e32 v27, v27, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v27, v38, v27
	v_and_b32_e32 v39, 0x7f800000, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v39
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.15:
	v_bfe_u32 v37, v27, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v37, v27, v37, 0x7fff
                                        ; implicit-def: $vgpr27
; %bb.16:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.17:
	v_and_b32_e32 v37, 0xffff, v27
	v_or_b32_e32 v39, 0x10000, v27
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v39, v27, vcc_lo
; %bb.18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v27, v36 offset:12
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v37, off offset:139264
                                        ; implicit-def: $vgpr37
	s_wait_dscnt 0x0
	v_mul_f32_e32 v28, v28, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v28, v38, v28
	v_and_b32_e32 v39, 0x7f800000, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v39
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.19:
	v_bfe_u32 v37, v28, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v37, v28, v37, 0x7fff
                                        ; implicit-def: $vgpr28
; %bb.20:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.21:
	v_and_b32_e32 v37, 0xffff, v28
	v_or_b32_e32 v39, 0x10000, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v39, v28, vcc_lo
; %bb.22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v28, v36 offset:16
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v37, off offset:208896
                                        ; implicit-def: $vgpr37
	s_wait_dscnt 0x0
	v_mul_f32_e32 v29, v29, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v29, v38, v29
	v_and_b32_e32 v39, 0x7f800000, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v39
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.23:
	v_bfe_u32 v37, v29, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v37, v29, v37, 0x7fff
                                        ; implicit-def: $vgpr29
; %bb.24:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.25:
	v_and_b32_e32 v37, 0xffff, v29
	v_or_b32_e32 v39, 0x10000, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v39, v29, vcc_lo
; %bb.26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v29, v36 offset:20
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v37, off offset:278528
                                        ; implicit-def: $vgpr37
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v30, v38, v30
	v_and_b32_e32 v39, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v39
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.27:
	v_bfe_u32 v37, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v37, v30, v37, 0x7fff
                                        ; implicit-def: $vgpr30
; %bb.28:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.29:
	v_and_b32_e32 v37, 0xffff, v30
	v_or_b32_e32 v39, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v39, v30, vcc_lo
; %bb.30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v30, v36 offset:24
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v37, off offset:348160
                                        ; implicit-def: $vgpr37
	s_wait_dscnt 0x0
	v_mul_f32_e32 v31, v31, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v31, v38, v31
	v_and_b32_e32 v39, 0x7f800000, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ne_u32_e32 0x7f800000, v39
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.31:
	v_bfe_u32 v37, v31, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v37, v31, v37, 0x7fff
                                        ; implicit-def: $vgpr31
; %bb.32:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.33:
	v_and_b32_e32 v37, 0xffff, v31
	v_or_b32_e32 v39, 0x10000, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v39, v31, vcc_lo
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v31, v36 offset:28
	global_store_d16_hi_b16 v[25:26], v37, off offset:417792
	s_wait_dscnt 0x0
	v_mul_f32_e32 v32, v32, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v32, v38, v32
	v_and_b32_e32 v36, 0x7f800000, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v36
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.35:
	v_bfe_u32 v36, v32, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v36, v32, v36, 0x7fff
                                        ; implicit-def: $vgpr32
; %bb.36:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.37:
	v_and_b32_e32 v36, 0xffff, v32
	v_or_b32_e32 v37, 0x10000, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v36
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v36, v37, v32, vcc_lo
; %bb.38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v32, v0 offset:768
	v_mul_f32_e32 v17, v17, v33
	global_store_d16_hi_b16 v[25:26], v36, off offset:487424
	s_wait_dscnt 0x0
	v_mul_f32_e32 v37, v32, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v17, 0x7f800000, v37
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v17
                                        ; implicit-def: $vgpr17
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.39:
	v_bfe_u32 v17, v37, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v37, v17, 0x7fff
                                        ; implicit-def: $vgpr37
; %bb.40:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.41:
	v_and_b32_e32 v17, 0xffff, v37
	v_or_b32_e32 v36, 0x10000, v37
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v36, v37, vcc_lo
; %bb.42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v18, v18, v34
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v17, off offset:128
                                        ; implicit-def: $vgpr17
	v_mul_f32_e32 v18, v32, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v36, 0x7f800000, v18
	v_cmpx_ne_u32_e32 0x7f800000, v36
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.43:
	v_bfe_u32 v17, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v18, v17, 0x7fff
                                        ; implicit-def: $vgpr18
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.45:
	v_and_b32_e32 v17, 0xffff, v18
	v_or_b32_e32 v36, 0x10000, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v36, v18, vcc_lo
; %bb.46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v18, v19, v35
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v17, off offset:69760
                                        ; implicit-def: $vgpr17
	v_mul_f32_e32 v18, v32, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v19, 0x7f800000, v18
	v_cmpx_ne_u32_e32 0x7f800000, v19
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.47:
	v_bfe_u32 v17, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v18, v17, 0x7fff
                                        ; implicit-def: $vgpr18
; %bb.48:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.49:
	v_and_b32_e32 v17, 0xffff, v18
	v_or_b32_e32 v19, 0x10000, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v19, v18, vcc_lo
; %bb.50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v18, v20, v27
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v17, off offset:139392
                                        ; implicit-def: $vgpr17
	v_mul_f32_e32 v18, v32, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v19, 0x7f800000, v18
	v_cmpx_ne_u32_e32 0x7f800000, v19
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.51:
	v_bfe_u32 v17, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v18, v17, 0x7fff
                                        ; implicit-def: $vgpr18
; %bb.52:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.53:
	v_and_b32_e32 v17, 0xffff, v18
	v_or_b32_e32 v19, 0x10000, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v19, v18, vcc_lo
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v18, v21, v28
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v17, off offset:209024
                                        ; implicit-def: $vgpr17
	v_mul_f32_e32 v18, v32, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v19, 0x7f800000, v18
	v_cmpx_ne_u32_e32 0x7f800000, v19
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.55:
	v_bfe_u32 v17, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v18, v17, 0x7fff
                                        ; implicit-def: $vgpr18
; %bb.56:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.57:
	v_and_b32_e32 v17, 0xffff, v18
	v_or_b32_e32 v19, 0x10000, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v19, v18, vcc_lo
; %bb.58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v18, v22, v29
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v17, off offset:278656
                                        ; implicit-def: $vgpr17
	v_mul_f32_e32 v18, v32, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v19, 0x7f800000, v18
	v_cmpx_ne_u32_e32 0x7f800000, v19
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.59:
	v_bfe_u32 v17, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v18, v17, 0x7fff
                                        ; implicit-def: $vgpr18
; %bb.60:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.61:
	v_and_b32_e32 v17, 0xffff, v18
	v_or_b32_e32 v19, 0x10000, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v19, v18, vcc_lo
; %bb.62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v18, v23, v30
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v17, off offset:348288
                                        ; implicit-def: $vgpr17
	v_mul_f32_e32 v18, v32, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v19, 0x7f800000, v18
	v_cmpx_ne_u32_e32 0x7f800000, v19
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.63:
	v_bfe_u32 v17, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v18, v17, 0x7fff
                                        ; implicit-def: $vgpr18
; %bb.64:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.65:
	v_and_b32_e32 v17, 0xffff, v18
	v_or_b32_e32 v19, 0x10000, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v19, v18, vcc_lo
; %bb.66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v18, v24, v31
	global_store_d16_hi_b16 v[25:26], v17, off offset:417920
	v_mul_f32_e32 v19, v32, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v18, 0x7f800000, v19
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.67:
	v_bfe_u32 v17, v19, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v18, v19, v17, 0x7fff
                                        ; implicit-def: $vgpr19
; %bb.68:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.69:
	v_or_b32_e32 v18, 0x10000, v19
	v_and_b32_e32 v17, 0xffff, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
; %bb.70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v17, v0 offset:1024
	v_mul_f32_e32 v9, v9, v33
	global_store_d16_hi_b16 v[25:26], v18, off offset:487552
	s_wait_dscnt 0x0
	v_mul_f32_e32 v19, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v9, 0x7f800000, v19
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v9
                                        ; implicit-def: $vgpr9
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.71:
	v_bfe_u32 v9, v19, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v19, v9, 0x7fff
                                        ; implicit-def: $vgpr19
; %bb.72:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.73:
	v_and_b32_e32 v9, 0xffff, v19
	v_or_b32_e32 v18, 0x10000, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v18, v19, vcc_lo
; %bb.74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v10, v10, v34
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v9, off offset:256
                                        ; implicit-def: $vgpr9
	v_mul_f32_e32 v10, v17, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v18, 0x7f800000, v10
	v_cmpx_ne_u32_e32 0x7f800000, v18
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.75:
	v_bfe_u32 v9, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v10, v9, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.76:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.77:
	v_and_b32_e32 v9, 0xffff, v10
	v_or_b32_e32 v18, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v18, v10, vcc_lo
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v10, v11, v35
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v9, off offset:69888
                                        ; implicit-def: $vgpr9
	v_mul_f32_e32 v10, v17, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 0x7f800000, v10
	v_cmpx_ne_u32_e32 0x7f800000, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.79:
	v_bfe_u32 v9, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v10, v9, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.80:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.81:
	v_and_b32_e32 v9, 0xffff, v10
	v_or_b32_e32 v11, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo
; %bb.82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v10, v12, v27
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v9, off offset:139520
                                        ; implicit-def: $vgpr9
	v_mul_f32_e32 v10, v17, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 0x7f800000, v10
	v_cmpx_ne_u32_e32 0x7f800000, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.83:
	v_bfe_u32 v9, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v10, v9, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.84:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.85:
	v_and_b32_e32 v9, 0xffff, v10
	v_or_b32_e32 v11, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo
; %bb.86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v10, v13, v28
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v9, off offset:209152
                                        ; implicit-def: $vgpr9
	v_mul_f32_e32 v10, v17, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 0x7f800000, v10
	v_cmpx_ne_u32_e32 0x7f800000, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.87:
	v_bfe_u32 v9, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v10, v9, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.88:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.89:
	v_and_b32_e32 v9, 0xffff, v10
	v_or_b32_e32 v11, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo
; %bb.90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v10, v14, v29
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v9, off offset:278784
                                        ; implicit-def: $vgpr9
	v_mul_f32_e32 v10, v17, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 0x7f800000, v10
	v_cmpx_ne_u32_e32 0x7f800000, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.91:
	v_bfe_u32 v9, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v10, v9, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.92:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.93:
	v_and_b32_e32 v9, 0xffff, v10
	v_or_b32_e32 v11, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo
; %bb.94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v10, v15, v30
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v9, off offset:348416
                                        ; implicit-def: $vgpr9
	v_mul_f32_e32 v10, v17, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 0x7f800000, v10
	v_cmpx_ne_u32_e32 0x7f800000, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.95:
	v_bfe_u32 v9, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v10, v9, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.97:
	v_and_b32_e32 v9, 0xffff, v10
	v_or_b32_e32 v11, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo
; %bb.98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v10, v16, v31
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v9, off offset:418048
                                        ; implicit-def: $vgpr9
	v_mul_f32_e32 v10, v17, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 0x7f800000, v10
	v_cmpx_ne_u32_e32 0x7f800000, v11
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.99:
	v_bfe_u32 v9, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v10, v9, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.100:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.101:
	v_and_b32_e32 v9, 0xffff, v10
	v_or_b32_e32 v11, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v11, v10, vcc_lo
; %bb.102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	ds_load_b32 v0, v0 offset:1280
	v_mul_f32_e32 v1, v1, v33
	global_store_d16_hi_b16 v[25:26], v9, off offset:487680
	s_wait_dscnt 0x0
	v_mul_f32_e32 v10, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0x7f800000, v10
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.103:
	v_bfe_u32 v1, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v10, v1, 0x7fff
                                        ; implicit-def: $vgpr10
; %bb.104:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.105:
	v_and_b32_e32 v1, 0xffff, v10
	v_or_b32_e32 v9, 0x10000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v9, v10, vcc_lo
; %bb.106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v2, v2, v34
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v1, off offset:384
                                        ; implicit-def: $vgpr1
	v_mul_f32_e32 v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v9, 0x7f800000, v2
	v_cmpx_ne_u32_e32 0x7f800000, v9
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.107:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.108:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.109:
	v_and_b32_e32 v1, 0xffff, v2
	v_or_b32_e32 v9, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v9, v2, vcc_lo
; %bb.110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v2, v3, v35
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v1, off offset:70016
                                        ; implicit-def: $vgpr1
	v_mul_f32_e32 v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7f800000, v2
	v_cmpx_ne_u32_e32 0x7f800000, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.111:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.112:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.113:
	v_and_b32_e32 v1, 0xffff, v2
	v_or_b32_e32 v3, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v3, v2, vcc_lo
; %bb.114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v2, v4, v27
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v1, off offset:139648
                                        ; implicit-def: $vgpr1
	v_mul_f32_e32 v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7f800000, v2
	v_cmpx_ne_u32_e32 0x7f800000, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.115:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.116:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.117:
	v_and_b32_e32 v1, 0xffff, v2
	v_or_b32_e32 v3, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v3, v2, vcc_lo
; %bb.118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v2, v5, v28
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v1, off offset:209280
                                        ; implicit-def: $vgpr1
	v_mul_f32_e32 v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7f800000, v2
	v_cmpx_ne_u32_e32 0x7f800000, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.119:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.120:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.121:
	v_and_b32_e32 v1, 0xffff, v2
	v_or_b32_e32 v3, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v3, v2, vcc_lo
; %bb.122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v2, v6, v29
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v1, off offset:278912
                                        ; implicit-def: $vgpr1
	v_mul_f32_e32 v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7f800000, v2
	v_cmpx_ne_u32_e32 0x7f800000, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.123:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.124:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.125:
	v_and_b32_e32 v1, 0xffff, v2
	v_or_b32_e32 v3, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v3, v2, vcc_lo
; %bb.126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v2, v7, v30
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v1, off offset:348544
                                        ; implicit-def: $vgpr1
	v_mul_f32_e32 v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0x7f800000, v2
	v_cmpx_ne_u32_e32 0x7f800000, v3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
; %bb.127:
	v_bfe_u32 v1, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v2, v1, 0x7fff
                                        ; implicit-def: $vgpr2
; %bb.128:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
; %bb.129:
	v_and_b32_e32 v1, 0xffff, v2
	v_or_b32_e32 v3, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v3, v2, vcc_lo
; %bb.130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mul_f32_e32 v2, v8, v31
	s_mov_b32 s0, exec_lo
	global_store_d16_hi_b16 v[25:26], v1, off offset:418176
                                        ; implicit-def: $vgpr1
	v_mul_f32_e32 v0, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0x7f800000, v0
	v_cmpx_ne_u32_e32 0x7f800000, v2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s0
	s_cbranch_execnz .LBB0_133
; %bb.131:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execnz .LBB0_134
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	global_store_d16_hi_b16 v[25:26], v1, off offset:487808
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB0_133:
	v_bfe_u32 v1, v0, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v0, v1, 0x7fff
                                        ; implicit-def: $vgpr0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB0_132
.LBB0_134:
	v_and_b32_e32 v1, 0xffff, v0
	v_or_b32_e32 v2, 0x10000, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v2, v0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	global_store_d16_hi_b16 v[25:26], v1, off offset:487808
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16, .Lfunc_end0-_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16
		.amdhsa_group_segment_fixed_size 24576
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 97
		.amdhsa_next_free_sgpr 14
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16,"axG",@progbits,_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16,comdat
                                        ; -- End function
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.num_vgpr, 97
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.num_agpr, 0
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.numbered_sgpr, 14
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.num_named_barrier, 0
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.private_seg_size, 0
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.uses_vcc, 1
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.uses_flat_scratch, 0
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.has_dyn_sized_stack, 0
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.has_recursion, 0
	.set .L_ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6192
; TotalNumSgprs: 16
; NumVgprs: 97
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 24576 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 97
; Occupancy: 12
; WaveLimiterHint : 1
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
	.type	__hip_cuid_bf841fd633a8b1a6,@object ; @__hip_cuid_bf841fd633a8b1a6
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_bf841fd633a8b1a6
__hip_cuid_bf841fd633a8b1a6:
	.byte	0                               ; 0x0
	.size	__hip_cuid_bf841fd633a8b1a6, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_bf841fd633a8b1a6
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 24576
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _ZN6ninfer3ops5r97006linear12_GLOBAL__N_158fp8_gate_up_m128n256_w32_k32_pingpong_qualification_kernelEPKhPKfS5_S7_P12hip_bfloat16.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     97
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
