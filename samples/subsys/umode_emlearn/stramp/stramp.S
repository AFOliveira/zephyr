/* Minimal S-mode trampoline at FW_WORKER_SMODE_ENTRY (0x8000C00000).
 *
 * Role: bridge between stock MachineMinion and Zephyr-U.
 *
 *   Stock MM (shire 0, non-master) does:
 *     csrw mepc, FW_WORKER_SMODE_ENTRY
 *     mret    (MPP=S)
 *     -> arrives here at 0x8000C00000 with priv=S
 *
 *   We install stvec to catch U-mode ecalls, flip MPP->U via sstatus.SPP=0,
 *   set sepc to Zephyr's _start, and sret into U-mode.  sscratch=0 at reset
 *   distinguishes cold entry from re-entry via trap.
 *
 *   On trap: only ecall-from-U (scause=8) is allowed.  SYSCALL_RETURN_FROM_KERNEL
 *   (a0=8) halts.  Other ids are treated as stubs: return a0=0, bump sepc+=4, sret.
 */

.section .text.init, "ax", @progbits
.globl _stramp_start
_stramp_start:
    csrr  t0, sscratch
    beqz  t0, stramp_cold

    /* Trap path: scause==8 (ecall-from-U) expected. */
    csrr  t0, scause
    li    t1, 8
    bne   t0, t1, stramp_halt

    /* a0 == SYSCALL_RETURN_FROM_KERNEL (8) -> halt. */
    li    t1, 8
    beq   a0, t1, stramp_halt

    /* Any other syscall: stub with return value 0, advance past the ecall. */
    li    a0, 0
    csrr  t0, sepc
    addi  t0, t0, 4
    csrw  sepc, t0
    sret

stramp_cold:
    /* Mark initialised. */
    li    t0, 1
    csrw  sscratch, t0

    /* stvec = this entry (page-aligned by link). */
    la    t0, _stramp_start
    csrw  stvec, t0

    /* sstatus.SPP = 0 so sret drops to U-mode. */
    li    t0, 0x100
    csrc  sstatus, t0

    /* sepc = Zephyr _start. */
    li    t0, ZEPHYR_ENTRY
    csrw  sepc, t0
    sret

stramp_halt:
1:  wfi
    j 1b
