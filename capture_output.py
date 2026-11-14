#!/usr/bin/env python3
"""Capture a command's combined output as timestamped JSONL records."""
from __future__ import annotations

import codecs
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
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
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
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

    def is_live_line(message: str) -> bool:
        text = message.strip()
        if not text:
            return False
        lower = text.lower()

        excluded = (
            "unsloth:",
            "unsloth zoo",
            "tokenizing",
            "filter:",
            "building labels",
            "warmup_ratio",
            "num examples =",
            "batch size per device",
            "gradient accumulation steps",
            "xformers",
            "torch.compile",
            "cuda usable",
            "cuda built",
        )
        if any(token in lower for token in excluded):
            return False

        if text.startswith("[INFO]") or text.startswith("[WARN]") or text.startswith("[ERROR]"):
            if any(token in lower for token in ("loading model", "loading tokenizer", "training for", "assistant-only loss labels verified", "saving", "export", "selected", "main model", "merging", "initializing")):
                return True
            if "unsloth" in lower or "xformers" in lower or "torch" in lower or "transformers" in lower or "cuda" in lower:
                return False
            return True

        if any(token in lower for token in ("epoch", "step", "loss", "grad", "learning rate", "lr", "saving", "progress", "export", "main model", "loading model", "training", "merging")):
            return True
        if re.search(r"\b\d+%\s*\|", text) or re.search(r"\b\d+/?\d+\s*(?:epoch|step)s?\b", lower):
            return True
        return False

    progress_line_active = False
    heartbeat_interval = 30 if stage in {"training", "gguf-export"} else None
    last_visible_output = time.monotonic()

    def finish_progress_line() -> None:
        nonlocal progress_line_active
        if progress_line_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            progress_line_active = False

    def emit_live(message: str) -> None:
        nonlocal last_visible_output, progress_line_active
        text = message.strip()
        if not text:
            return
        if (
            stage == "training"
            and text.startswith("{")
            and (
                re.search(r"['\"]train_runtime['\"]\s*:", text)
                or all(re.search(rf"['\"]{key}['\"]\s*:", text) for key in ("loss", "grad_norm", "learning_rate", "epoch"))
            )
        ):
            return
        if not is_live_line(text):
            return
        lower = text.lower()
        is_progress = bool(re.search(r"\b\d+%\s*\|", text)) or "it/s" in lower or "s/it" in lower or "epoch" in lower and "%" in text
        if is_progress:
            if sys.stdout.isatty():
                sys.stdout.write(f"\033[2K\r{text}")
                progress_line_active = True
            else:
                sys.stdout.write(f"{text}\n")
            sys.stdout.flush()
            last_visible_output = time.monotonic()
            return
        finish_progress_line()
        sys.stdout.write(f"{text}\n")
        sys.stdout.flush()
        last_visible_output = time.monotonic()

    def emit_heartbeat() -> None:
        nonlocal last_visible_output
        finish_progress_line()
        notice = f"[NOTICE] Still working on {stage.replace('-', ' ')}; detailed output is being saved to the run log."
        record(notice)
        sys.stdout.write(f"{notice}\n")
        sys.stdout.flush()
        last_visible_output = time.monotonic()

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    try:
        assert process.stdout is not None
        while True:
            timeout = None
            if heartbeat_interval is not None:
                timeout = max(0.0, heartbeat_interval - (time.monotonic() - last_visible_output))
            readable, _, _ = select.select([process.stdout.fileno()], [], [], timeout)
            if not readable:
                emit_heartbeat()
                continue
            chunk = os.read(process.stdout.fileno(), 8192)
            if not chunk:
                break
            pending += decoder.decode(chunk)
            while True:
                separators = [index for index in (pending.find("\n"), pending.find("\r")) if index >= 0]
                if not separators:
                    break
                index = min(separators)
                separator = pending[index]
                line = pending[:index]
                record(line)
                emit_live(line)
                pending = pending[index + 1:]
                if separator == "\r" and pending.startswith("\n"):
                    pending = pending[1:]
            if heartbeat_interval is not None and time.monotonic() - last_visible_output >= heartbeat_interval:
                emit_heartbeat()
        pending += decoder.decode(b"", final=True)
        if pending:
            record(pending)
            emit_live(pending)
        finish_progress_line()
        return_code = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait()
        finish_progress_line()
        record("Command interrupted by user.")
        return 130
    finally:
        if process.stdout is not None:
            process.stdout.close()
        log_stream.close()

    return return_code if return_code >= 0 else 128 - return_code


if __name__ == "__main__":
    raise SystemExit(main())
