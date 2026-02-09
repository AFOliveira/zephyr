# Copyright (c) 2026 AIFoundry
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
        raise RuntimeError('pyelftools is missing; pass --sp-reset-pc or --reset-pc')

    with open(elf_path, 'rb') as elf:
        return ELFFile(elf).header['e_entry']


class EtEmuBinaryRunner(ZephyrBinaryRunner):
    """Runner for the ET System Emulator (sys_emu)."""

    def __init__(self, cfg, sys_emu, sp_reset_pc, reset_pc, reset_target,
                 shires, max_cycles, enable_minions, extra_args):
        super().__init__(cfg)
        self.sys_emu = sys_emu
        self.sp_reset_pc = sp_reset_pc
        self.reset_pc = reset_pc
        self.reset_target = reset_target
        self.shires = shires
        self.max_cycles = max_cycles
        self.enable_minions = enable_minions
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
            '--sp-reset-pc',
            type=_int_arg,
            help='override SP reset PC (hex ok)'
        )
        parser.add_argument(
            '--reset-pc',
            type=_int_arg,
            help='override reset PC (hex ok)'
        )
        parser.add_argument(
            '--reset-target',
            choices=('sp', 'main'),
            default='sp',
            help='target for auto reset PC selection (default: sp)'
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
            '--sys-emu-arg',
            action='append',
            metavar='ARG',
            help='extra argument passed to sys_emu (repeatable)'
        )

    @classmethod
    def do_create(cls, cfg, args: argparse.Namespace):
        sys_emu = args.sys_emu or os.environ.get('SYS_EMU') or 'sys_emu'

        return EtEmuBinaryRunner(
            cfg,
            sys_emu=sys_emu,
            sp_reset_pc=args.sp_reset_pc,
            reset_pc=args.reset_pc,
            reset_target=args.reset_target,
            shires=args.shires,
            max_cycles=args.max_cycles,
            enable_minions=args.enable_minions,
            extra_args=args.sys_emu_arg,
        )

    def do_run(self, command, **kwargs):
        if command != 'simulate':
            raise AssertionError

        if self.cfg.elf_file is None:
            raise RuntimeError('missing ELF file in RunnerConfig')

        sys_emu = self.require(self.sys_emu)
        cmd = [sys_emu]

        cmd += ['-elf_load', self.cfg.elf_file]

        if self.sp_reset_pc is not None and self.reset_pc is not None:
            raise RuntimeError('use only one of --sp-reset-pc or --reset-pc')

        if self.sp_reset_pc is not None:
            cmd += ['-sp_reset_pc', f'{self.sp_reset_pc:#x}']
        elif self.reset_pc is not None:
            cmd += ['-reset_pc', f'{self.reset_pc:#x}']
        else:
            entry = _elf_entry(self.cfg.elf_file)
            if self.reset_target == 'sp':
                cmd += ['-sp_reset_pc', f'{entry:#x}']
            else:
                cmd += ['-reset_pc', f'{entry:#x}']

        cmd += ['-shires', self.shires]

        if not self.enable_minions:
            cmd.append('-mins_dis')

        if self.max_cycles is not None:
            cmd += ['-max_cycles', str(self.max_cycles)]

        cmd += self.extra_args

        self.check_call(cmd)
