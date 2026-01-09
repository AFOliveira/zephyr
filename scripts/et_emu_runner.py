#!/usr/bin/env python3
# Copyright (c) 2025 AIFoundry
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper script for sys_emu that monitors output and exits on test completion

import subprocess
import sys
import os
import re
import signal

# Patterns that indicate test completion
PASS_PATTERNS = [
    r'PROJECT EXECUTION SUCCESSFUL',
    r'RunID:.*PASSED',
    r'PASS - .* in \d+\.\d+ seconds',
]

FAIL_PATTERNS = [
    r'PROJECT EXECUTION FAILED',
    r'RunID:.*FAILED',
    r'FAIL - .* in \d+\.\d+ seconds',
    r'ASSERTION FAIL',
    r'\*\*\* Booting Zephyr.*\n.*FAIL',
]

# Pattern for simple samples (like hello_world) that just print and should exit
SIMPLE_DONE_PATTERNS = [
    r'Hello World!',
]

def main():
    if len(sys.argv) < 2:
        print("Usage: et_emu_runner.py <sys_emu_path> [args...]", file=sys.stderr)
        sys.exit(1)

    sysemu_path = sys.argv[1]
    args = sys.argv[2:]

    # Start sys_emu process
    proc = subprocess.Popen(
        [sysemu_path] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    result = None

    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)

            # Check for pass patterns
            for pattern in PASS_PATTERNS:
                if re.search(pattern, line):
                    result = 0
                    break

            # Check for fail patterns
            for pattern in FAIL_PATTERNS:
                if re.search(pattern, line):
                    result = 1
                    break

            # Check for simple completion (like hello_world)
            # Give it a moment to print more output
            for pattern in SIMPLE_DONE_PATTERNS:
                if re.search(pattern, line):
                    # Wait briefly then exit success
                    import time
                    time.sleep(0.1)
                    result = 0
                    break

            if result is not None:
                break

    except KeyboardInterrupt:
        result = 130

    finally:
        # Terminate the emulator
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    sys.exit(result if result is not None else proc.returncode)

if __name__ == '__main__':
    main()
