# Copyright (c) 2025 AIFoundry
# SPDX-License-Identifier: Apache-2.0

"""Runner for the ET system emulator (sys_emu)."""

import argparse
import os

from runners.core import RunnerCaps, ZephyrBinaryRunner

try:
    from elftools.elf.elffile import ELFFile
    ELFTOOLS_MISSING = False
except ImportError:  # pragma: no cover
    ELFTOOLS_MISSING = True


def _int_arg(value: str) -> int:
    return int(value, 0)


def _elf_entry(elf_path: str) -> int:
    if ELFTOOLS_MISSING:
        raise RuntimeError(
            'pyelftools is missing; pass --sp-reset-pc or set --bootrom')

    with open(elf_path, 'rb') as elf:
        return ELFFile(elf).header['e_entry']


class EtEmuBinaryRunner(ZephyrBinaryRunner):
    """Runner for the ET System Emulator (sys_emu)."""

    def __init__(self, cfg, sys_emu, bootrom, no_bootrom, sp_reset_pc,
                 shires, max_cycles, enable_minions, uart0_tx_file,
                 uart1_tx_file, uart0_rx_file, uart1_rx_file, extra_args):
        super().__init__(cfg)
        self.sys_emu = sys_emu
        self.bootrom = bootrom
        self.no_bootrom = no_bootrom
        self.sp_reset_pc = sp_reset_pc
        self.shires = shires
        self.max_cycles = max_cycles
        self.enable_minions = enable_minions
        self.uart0_tx_file = uart0_tx_file
        self.uart1_tx_file = uart1_tx_file
        self.uart0_rx_file = uart0_rx_file
        self.uart1_rx_file = uart1_rx_file
        self.extra_args = extra_args or []

    @classmethod
    def name(cls):
        return 'etemu'

    @classmethod
    def capabilities(cls):
        return RunnerCaps(commands={'simulate'})

    @classmethod
    def do_add_parser(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--sys-emu',
            help='path to sys_emu binary (defaults to $SYS_EMU or sys_emu in PATH)'
        )
        parser.add_argument(
            '--bootrom',
            help='path to BootromTrampolineToBL2.elf (optional)'
        )
        parser.add_argument(
            '--no-bootrom',
            action='store_true',
            help='do not load bootrom; use --sp-reset-pc or ELF entry'
        )
        parser.add_argument(
            '--sp-reset-pc',
            type=_int_arg,
            help='override SP reset PC (hex ok; required if no bootrom and pyelftools missing)'
        )
        parser.add_argument(
            '--shires',
            default='0x400000000',
            help='shires mask (default: 0x400000000)'
        )
        parser.add_argument(
            '--max-cycles',
            type=int,
            help='stop execution after this many cycles'
        )
        parser.add_argument(
            '--enable-minions',
            action='store_true',
            help='do not pass -mins_dis (enable minions)'
        )
        parser.add_argument(
            '--uart0-tx-file',
            help='path for SPIO UART0 TX output (default: stdout)'
        )
        parser.add_argument(
            '--uart1-tx-file',
            help='path for SPIO UART1 TX output (default: stdout)'
        )
        parser.add_argument(
            '--uart0-rx-file',
            help='path for SPIO UART0 RX input (default: stdin)'
        )
        parser.add_argument(
            '--uart1-rx-file',
            help='path for SPIO UART1 RX input (default: stdin)'
        )
        parser.add_argument(
            '--sys-emu-arg',
            action='append',
            metavar='ARG',
            help='extra argument passed to sys_emu (repeatable)'
        )

    @classmethod
    def do_create(cls, cfg, args: argparse.Namespace):
        sys_emu = args.sys_emu or os.environ.get('SYS_EMU') or 'sys_emu'
        bootrom = args.bootrom or os.environ.get('BOOTROM_TRAMPOLINE_TO_BL2_ELF')

        return EtEmuBinaryRunner(
            cfg,
            sys_emu=sys_emu,
            bootrom=bootrom,
            no_bootrom=args.no_bootrom,
            sp_reset_pc=args.sp_reset_pc,
            shires=args.shires,
            max_cycles=args.max_cycles,
            enable_minions=args.enable_minions,
            uart0_tx_file=args.uart0_tx_file,
            uart1_tx_file=args.uart1_tx_file,
            uart0_rx_file=args.uart0_rx_file,
            uart1_rx_file=args.uart1_rx_file,
            extra_args=args.sys_emu_arg,
        )

    def do_run(self, command, **kwargs):
        if command != 'simulate':
            raise AssertionError

        if self.cfg.elf_file is None:
            raise RuntimeError('missing ELF file in RunnerConfig')

        sys_emu = self.require(self.sys_emu)
        cmd = [sys_emu]

        use_bootrom = (not self.no_bootrom) and bool(self.bootrom)
        if use_bootrom and not os.path.exists(self.bootrom):
            raise RuntimeError(f'bootrom not found: {self.bootrom}')
        if use_bootrom:
            cmd += ['-elf_load', self.bootrom]
        cmd += ['-elf_load', self.cfg.elf_file]

        sp_reset_pc = self.sp_reset_pc
        if sp_reset_pc is not None:
            cmd += ['-sp_reset_pc', f'{sp_reset_pc:#x}']
        elif not use_bootrom:
            sp_reset_pc = _elf_entry(self.cfg.elf_file)
            cmd += ['-sp_reset_pc', f'{sp_reset_pc:#x}']

        cmd += ['-shires', self.shires]

        if not self.enable_minions:
            cmd.append('-mins_dis')

        if self.max_cycles is not None:
            cmd += ['-max_cycles', str(self.max_cycles)]

        if self.uart0_tx_file:
            cmd += ['-spio_uart0_tx_file', self.uart0_tx_file]
        if self.uart1_tx_file:
            cmd += ['-spio_uart1_tx_file', self.uart1_tx_file]
        if self.uart0_rx_file:
            cmd += ['-spio_uart0_rx_file', self.uart0_rx_file]
        if self.uart1_rx_file:
            cmd += ['-spio_uart1_rx_file', self.uart1_rx_file]

        cmd += self.extra_args

        self.check_call(cmd)
