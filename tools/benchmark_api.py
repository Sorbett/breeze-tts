"""Measure Breeze streaming API first-audio latency and real-time factor."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

SAMPLE_RATE = 24000
SAMPLE_BYTES = 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860/v1/audio/speech")
    parser.add_argument("--text", default="你好，这是 Breeze CUDA 流式性能测试。")
    parser.add_argument("--instruction", default="自然、清晰地说话。")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text-file", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, help="write the final run as raw PCM16")
    args = parser.parse_args()

    if (args.ref_audio is None) != (args.ref_text_file is None):
        parser.error("--ref-audio and --ref-text-file must be provided together")
    ref_bytes = args.ref_audio.read_bytes() if args.ref_audio else None
    ref_text = args.ref_text_file.read_text(encoding="utf-8").strip() if args.ref_text_file else ""

    with httpx.Client(timeout=120.0) as client:
        for run in range(1, args.runs + 1):
            fields = {
                "text": (None, args.text),
                "instruction": (None, args.instruction),
                "cfg_scale": (None, str(args.cfg_scale)),
                "seed": (None, "42"),
            }
            if ref_bytes is not None:
                fields["ref_audio"] = (args.ref_audio.name, ref_bytes)
                fields["ref_text"] = (None, ref_text)
            started = time.perf_counter()
            first_audio = None
            pcm = bytearray()
            with client.stream("POST", args.url, files=fields) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if chunk and first_audio is None:
                        first_audio = time.perf_counter()
                    pcm.extend(chunk)
            finished = time.perf_counter()
            total_bytes = len(pcm)
            audio_s = total_bytes / (SAMPLE_RATE * SAMPLE_BYTES)
            elapsed_s = finished - started
            print(json.dumps({
                "run": run,
                "ttfa_ms": round(((first_audio or finished) - started) * 1000, 1),
                "elapsed_s": round(elapsed_s, 3),
                "audio_s": round(audio_s, 3),
                "rtf": round(elapsed_s / audio_s, 3) if audio_s else None,
                "bytes": total_bytes,
            }))
            if args.output and run == args.runs:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(pcm)


if __name__ == "__main__":
    main()
