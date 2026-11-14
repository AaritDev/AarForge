#!/usr/bin/env python3
"""Capture a command's combined output as timestamped JSONL records."""
from __future__ import annotations

import codecs
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 5:
        print("[ERROR] Invalid log-capture arguments.", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    run_code, stage = sys.argv[2:4]
    command = sys.argv[5:] if sys.argv[4] == "--" else sys.argv[4:]
    if not command:
        print("[ERROR] No command was provided for log capture.", file=sys.stderr)
        return 2

    try:
        log_stream = log_path.open("a", encoding="utf-8")
    except OSError as exc:
        print(f"[ERROR] Could not open JSONL log {log_path}: {exc}", file=sys.stderr)
        return 2
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        log_stream.close()
        print(f"[ERROR] Could not start logged command: {exc}", file=sys.stderr)
        return 2

    def record(message: str) -> None:
        if not message:
            return
        upper = message.upper()
        if re.search(r"\b(ERROR|FAIL|FAILURE|FAILED|EXCEPTION|TRACEBACK)\b", upper):
            level = "ERROR"
        elif re.search(r"\b(WARNING|WARN)\b", upper):
            level = "WARN"
        else:
            level = "INFO"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_code": run_code,
            "stage": stage,
            "level": level,
            "message": message,
        }
        log_stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log_stream.flush()

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    try:
        assert process.stdout is not None
        while chunk := os.read(process.stdout.fileno(), 8192):
            pending += decoder.decode(chunk)
            while True:
                separators = [index for index in (pending.find("\n"), pending.find("\r")) if index >= 0]
                if not separators:
                    break
                index = min(separators)
                separator = pending[index]
                record(pending[:index])
                pending = pending[index + 1:]
                if separator == "\r" and pending.startswith("\n"):
                    pending = pending[1:]
        pending += decoder.decode(b"", final=True)
        record(pending)
        return_code = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait()
        record("Command interrupted by user.")
        return 130
    finally:
        if process.stdout is not None:
            process.stdout.close()
        log_stream.close()

    return return_code if return_code >= 0 else 128 - return_code


if __name__ == "__main__":
    raise SystemExit(main())
