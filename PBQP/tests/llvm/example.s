	.file	"example.c"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function high_register_pressure
.LCPI0_0:
	.quad	0x4004000000000000              # double 2.5
	.quad	0x4003333333333333              # double 2.3999999999999999
.LCPI0_1:
	.quad	0x400599999999999a              # double 2.7000000000000002
	.quad	0x4004cccccccccccd              # double 2.6000000000000001
.LCPI0_2:
	.quad	0x4007333333333333              # double 2.8999999999999999
	.quad	0x4006666666666666              # double 2.7999999999999998
.LCPI0_3:
	.quad	0x4008cccccccccccd              # double 3.1000000000000001
	.quad	0x4008000000000000              # double 3
.LCPI0_4:
	.quad	0x400a666666666666              # double 3.2999999999999998
	.quad	0x400999999999999a              # double 3.2000000000000002
.LCPI0_5:
	.quad	0x400c000000000000              # double 3.5
	.quad	0x400b333333333333              # double 3.3999999999999999
.LCPI0_6:
	.quad	0x400d99999999999a              # double 3.7000000000000002
	.quad	0x400ccccccccccccd              # double 3.6000000000000001
.LCPI0_7:
	.quad	0x400f333333333333              # double 3.8999999999999999
	.quad	0x400e666666666666              # double 3.7999999999999998
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI0_8:
	.quad	0x3ff0000000000000              # double 1
.LCPI0_9:
	.quad	0x3ff199999999999a              # double 1.1000000000000001
.LCPI0_10:
	.quad	0x3ff3333333333333              # double 1.2
.LCPI0_11:
	.quad	0x3ff4cccccccccccd              # double 1.3
.LCPI0_12:
	.quad	0x3ff6666666666666              # double 1.3999999999999999
.LCPI0_13:
	.quad	0x3ff8000000000000              # double 1.5
.LCPI0_14:
	.quad	0x3ff999999999999a              # double 1.6000000000000001
.LCPI0_15:
	.quad	0x3ffb333333333333              # double 1.7
.LCPI0_16:
	.quad	0x3ffccccccccccccd              # double 1.8
.LCPI0_17:
	.quad	0x3ffe666666666666              # double 1.8999999999999999
.LCPI0_18:
	.quad	0x4000000000000000              # double 2
.LCPI0_19:
	.quad	0x4000cccccccccccd              # double 2.1000000000000001
.LCPI0_20:
	.quad	0x400199999999999a              # double 2.2000000000000002
.LCPI0_21:
	.quad	0x4002666666666666              # double 2.2999999999999998
	.text
	.globl	high_register_pressure
	.p2align	4
	.type	high_register_pressure,@function
high_register_pressure:                 # @high_register_pressure
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movapd	.LCPI0_0(%rip), %xmm1           # xmm1 = [2.5E+0,2.3999999999999999E+0]
	movapd	.LCPI0_1(%rip), %xmm2           # xmm2 = [2.7000000000000002E+0,2.6000000000000001E+0]
	movapd	.LCPI0_2(%rip), %xmm3           # xmm3 = [2.8999999999999999E+0,2.7999999999999998E+0]
	movapd	.LCPI0_3(%rip), %xmm4           # xmm4 = [3.1000000000000001E+0,3.0E+0]
	movapd	.LCPI0_4(%rip), %xmm5           # xmm5 = [3.2999999999999998E+0,3.2000000000000002E+0]
	movapd	.LCPI0_5(%rip), %xmm6           # xmm6 = [3.5E+0,3.3999999999999999E+0]
	movapd	.LCPI0_6(%rip), %xmm14          # xmm14 = [3.7000000000000002E+0,3.6000000000000001E+0]
	movapd	.LCPI0_7(%rip), %xmm15          # xmm15 = [3.8999999999999999E+0,3.7999999999999998E+0]
	movsd	.LCPI0_8(%rip), %xmm7           # xmm7 = [1.0E+0,0.0E+0]
	movsd	.LCPI0_9(%rip), %xmm8           # xmm8 = [1.1000000000000001E+0,0.0E+0]
	movsd	.LCPI0_10(%rip), %xmm9          # xmm9 = [1.2E+0,0.0E+0]
	movsd	.LCPI0_11(%rip), %xmm10         # xmm10 = [1.3E+0,0.0E+0]
	movsd	.LCPI0_12(%rip), %xmm11         # xmm11 = [1.3999999999999999E+0,0.0E+0]
	movsd	.LCPI0_13(%rip), %xmm0          # xmm0 = [1.5E+0,0.0E+0]
	movsd	%xmm0, -128(%rsp)               # 8-byte Spill
	movsd	.LCPI0_14(%rip), %xmm0          # xmm0 = [1.6000000000000001E+0,0.0E+0]
	movsd	%xmm0, -120(%rsp)               # 8-byte Spill
	movsd	.LCPI0_15(%rip), %xmm0          # xmm0 = [1.7E+0,0.0E+0]
	movsd	%xmm0, -112(%rsp)               # 8-byte Spill
	movl	$100, %eax
	movsd	.LCPI0_16(%rip), %xmm12         # xmm12 = [1.8E+0,0.0E+0]
	movsd	.LCPI0_17(%rip), %xmm13         # xmm13 = [1.8999999999999999E+0,0.0E+0]
	movsd	.LCPI0_18(%rip), %xmm0          # xmm0 = [2.0E+0,0.0E+0]
	movsd	%xmm0, -104(%rsp)               # 8-byte Spill
	movsd	.LCPI0_19(%rip), %xmm0          # xmm0 = [2.1000000000000001E+0,0.0E+0]
	movsd	%xmm0, -96(%rsp)                # 8-byte Spill
	movsd	.LCPI0_20(%rip), %xmm0          # xmm0 = [2.2000000000000002E+0,0.0E+0]
	movsd	%xmm0, -88(%rsp)                # 8-byte Spill
	movsd	.LCPI0_21(%rip), %xmm0          # xmm0 = [2.2999999999999998E+0,0.0E+0]
	movapd	%xmm0, -80(%rsp)                # 16-byte Spill
	.p2align	4
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	movapd	%xmm2, -64(%rsp)                # 16-byte Spill
	movapd	%xmm3, -48(%rsp)                # 16-byte Spill
	movapd	%xmm4, -32(%rsp)                # 16-byte Spill
	movapd	%xmm5, -16(%rsp)                # 16-byte Spill
	movapd	%xmm6, %xmm2
	movapd	%xmm14, %xmm3
	movapd	%xmm15, %xmm4
	mulsd	%xmm8, %xmm7
	addsd	%xmm9, %xmm7
	mulsd	%xmm9, %xmm8
	addsd	%xmm10, %xmm8
	mulsd	%xmm10, %xmm9
	addsd	%xmm11, %xmm9
	mulsd	%xmm11, %xmm10
	addsd	-128(%rsp), %xmm10              # 8-byte Folded Reload
	mulsd	-128(%rsp), %xmm11              # 8-byte Folded Reload
	addsd	-120(%rsp), %xmm11              # 8-byte Folded Reload
	movsd	-128(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	-120(%rsp), %xmm0               # 8-byte Folded Reload
	movsd	%xmm0, -128(%rsp)               # 8-byte Spill
	movsd	-128(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addsd	-112(%rsp), %xmm0               # 8-byte Folded Reload
	movsd	%xmm0, -128(%rsp)               # 8-byte Spill
	movsd	-120(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	-112(%rsp), %xmm0               # 8-byte Folded Reload
	movsd	%xmm0, -120(%rsp)               # 8-byte Spill
	movsd	-120(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addsd	%xmm12, %xmm0
	movsd	%xmm0, -120(%rsp)               # 8-byte Spill
	movsd	-112(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	%xmm12, %xmm0
	movsd	%xmm0, -112(%rsp)               # 8-byte Spill
	movsd	-112(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addsd	%xmm13, %xmm0
	movsd	%xmm0, -112(%rsp)               # 8-byte Spill
	mulsd	%xmm13, %xmm12
	addsd	-104(%rsp), %xmm12              # 8-byte Folded Reload
	mulsd	-104(%rsp), %xmm13              # 8-byte Folded Reload
	addsd	-96(%rsp), %xmm13               # 8-byte Folded Reload
	movsd	-104(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	-96(%rsp), %xmm0                # 8-byte Folded Reload
	movsd	%xmm0, -104(%rsp)               # 8-byte Spill
	movsd	-104(%rsp), %xmm0               # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addsd	-88(%rsp), %xmm0                # 8-byte Folded Reload
	movsd	%xmm0, -104(%rsp)               # 8-byte Spill
	movsd	-96(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	-88(%rsp), %xmm0                # 8-byte Folded Reload
	movsd	%xmm0, -96(%rsp)                # 8-byte Spill
	movsd	-96(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addsd	-80(%rsp), %xmm0                # 16-byte Folded Reload
	movsd	%xmm0, -96(%rsp)                # 8-byte Spill
	movapd	%xmm1, %xmm5
	unpckhpd	%xmm1, %xmm5                    # xmm5 = xmm5[1],xmm1[1]
	movsd	-88(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	-80(%rsp), %xmm0                # 16-byte Folded Reload
	movsd	%xmm0, -88(%rsp)                # 8-byte Spill
	movsd	-88(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addsd	%xmm5, %xmm0
	movsd	%xmm0, -88(%rsp)                # 8-byte Spill
	movapd	-80(%rsp), %xmm0                # 16-byte Reload
	mulsd	%xmm5, %xmm0
	movapd	%xmm0, -80(%rsp)                # 16-byte Spill
	movapd	-80(%rsp), %xmm0                # 16-byte Reload
	addsd	%xmm1, %xmm0
	movapd	%xmm0, -80(%rsp)                # 16-byte Spill
	movapd	%xmm7, %xmm15
	unpcklpd	%xmm4, %xmm15                   # xmm15 = xmm15[0],xmm4[0]
	movapd	%xmm8, %xmm0
	unpcklpd	%xmm7, %xmm0                    # xmm0 = xmm0[0],xmm7[0]
	mulpd	%xmm4, %xmm15
	addpd	%xmm0, %xmm15
	movapd	%xmm4, %xmm14
	shufpd	$1, %xmm3, %xmm14               # xmm14 = xmm14[1],xmm3[0]
	mulpd	%xmm3, %xmm14
	addpd	%xmm4, %xmm14
	movapd	%xmm3, %xmm6
	shufpd	$1, %xmm2, %xmm6                # xmm6 = xmm6[1],xmm2[0]
	mulpd	%xmm2, %xmm6
	addpd	%xmm3, %xmm6
	movapd	%xmm2, %xmm5
	shufpd	$1, -16(%rsp), %xmm5            # 16-byte Folded Reload
                                        # xmm5 = xmm5[1],mem[0]
	mulpd	-16(%rsp), %xmm5                # 16-byte Folded Reload
	addpd	%xmm2, %xmm5
	movapd	-16(%rsp), %xmm4                # 16-byte Reload
	shufpd	$1, -32(%rsp), %xmm4            # 16-byte Folded Reload
                                        # xmm4 = xmm4[1],mem[0]
	mulpd	-32(%rsp), %xmm4                # 16-byte Folded Reload
	addpd	-16(%rsp), %xmm4                # 16-byte Folded Reload
	movapd	-32(%rsp), %xmm3                # 16-byte Reload
	shufpd	$1, -48(%rsp), %xmm3            # 16-byte Folded Reload
                                        # xmm3 = xmm3[1],mem[0]
	mulpd	-48(%rsp), %xmm3                # 16-byte Folded Reload
	addpd	-32(%rsp), %xmm3                # 16-byte Folded Reload
	movapd	-48(%rsp), %xmm2                # 16-byte Reload
	shufpd	$1, -64(%rsp), %xmm2            # 16-byte Folded Reload
                                        # xmm2 = xmm2[1],mem[0]
	mulpd	-64(%rsp), %xmm2                # 16-byte Folded Reload
	addpd	-48(%rsp), %xmm2                # 16-byte Folded Reload
	movapd	-64(%rsp), %xmm0                # 16-byte Reload
	shufpd	$1, %xmm1, %xmm0                # xmm0 = xmm0[1],xmm1[0]
	mulpd	%xmm0, %xmm1
	addpd	-64(%rsp), %xmm1                # 16-byte Folded Reload
	decl	%eax
	jne	.LBB0_1
# %bb.2:
	addsd	%xmm8, %xmm7
	addsd	%xmm9, %xmm7
	addsd	%xmm10, %xmm7
	addsd	%xmm11, %xmm7
	addsd	-128(%rsp), %xmm7               # 8-byte Folded Reload
	addsd	-120(%rsp), %xmm7               # 8-byte Folded Reload
	addsd	-112(%rsp), %xmm7               # 8-byte Folded Reload
	addsd	%xmm12, %xmm7
	addsd	%xmm13, %xmm7
	addsd	-104(%rsp), %xmm7               # 8-byte Folded Reload
	addsd	-96(%rsp), %xmm7                # 8-byte Folded Reload
	addsd	-88(%rsp), %xmm7                # 8-byte Folded Reload
	addsd	-80(%rsp), %xmm7                # 16-byte Folded Reload
	movapd	%xmm1, %xmm8
	unpckhpd	%xmm1, %xmm8                    # xmm8 = xmm8[1],xmm1[1]
	addsd	%xmm7, %xmm8
	addsd	%xmm1, %xmm8
	movapd	%xmm2, %xmm0
	unpckhpd	%xmm2, %xmm0                    # xmm0 = xmm0[1],xmm2[1]
	addsd	%xmm8, %xmm0
	addsd	%xmm2, %xmm0
	movapd	%xmm3, %xmm1
	unpckhpd	%xmm3, %xmm1                    # xmm1 = xmm1[1],xmm3[1]
	addsd	%xmm0, %xmm1
	addsd	%xmm3, %xmm1
	movapd	%xmm4, %xmm0
	unpckhpd	%xmm4, %xmm0                    # xmm0 = xmm0[1],xmm4[1]
	addsd	%xmm1, %xmm0
	addsd	%xmm4, %xmm0
	movapd	%xmm5, %xmm1
	unpckhpd	%xmm5, %xmm1                    # xmm1 = xmm1[1],xmm5[1]
	addsd	%xmm0, %xmm1
	addsd	%xmm5, %xmm1
	movapd	%xmm6, %xmm0
	unpckhpd	%xmm6, %xmm0                    # xmm0 = xmm0[1],xmm6[1]
	addsd	%xmm1, %xmm0
	addsd	%xmm6, %xmm0
	movapd	%xmm14, %xmm1
	unpckhpd	%xmm14, %xmm1                   # xmm1 = xmm1[1],xmm14[1]
	addsd	%xmm0, %xmm1
	addsd	%xmm14, %xmm1
	movapd	%xmm15, %xmm0
	unpckhpd	%xmm15, %xmm0                   # xmm0 = xmm0[1],xmm15[1]
	addsd	%xmm1, %xmm0
	addsd	%xmm15, %xmm0
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	high_register_pressure, .Lfunc_end0-high_register_pressure
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	high_register_pressure
	cvttsd2si	%xmm0, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
