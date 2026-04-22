/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Erbium MPROT plumbing for the U-mode demo slice.
 *
 * MACHINE_NEIGH.mprot register (from et-platform hal esr.h):
 *   address  = 0x80100038  (ESR base 0x80000000 + Machine_neigh base
 *                           0x00100000 + register offset 0x38)
 *   fields:
 *     [3:0]  smode_size
 *     [7:4]  mmode_size
 *     [8]    mprot_en     (0 = MPROT disabled / allow-all)
 *   reset    = 0
 *
 * Reset value has mprot_en=0 which means MPROT enforcement is off -- i.e.
 * U-mode has full access to the memory map.  That is exactly what we need
 * for the minimal single-hart round-trip, so the "allow all" implementation
 * is simply "leave the reset value alone".
 *
 * Keeping these as real functions (rather than no-ops inlined at call sites)
 * documents the intent and gives us a future seam for enabling MPROT once
 * the demo grows into something more realistic.
 *
 * TODO: hard-coded shire=0, neigh=0 per the single-minion assumption.
 */

#include <stdint.h>
#include <zephyr/sys/sys_io.h>

#define ERBIUM_MACHINE_NEIGH_MPROT_ADDR 0x80100038UL

#define ERBIUM_MPROT_EN_BIT             (1U << 8)

void erbium_mprot_allow_all(void)
{
	/* Explicitly clear mprot_en so the register state is deterministic
	 * even if some earlier boot stage poked it.  Upper bits are RES0
	 * per the reset mask 0x1FF. */
	volatile uint64_t *mprot = (volatile uint64_t *)ERBIUM_MACHINE_NEIGH_MPROT_ADDR;

	*mprot = 0;
}

void erbium_mprot_lock_m(void)
{
	/*
	 * Placeholder: a real M-mode lock would configure mmode_size /
	 * smode_size to partition the address space and then set mprot_en.
	 * For the demo we intentionally leave MPROT open.
	 */
}
