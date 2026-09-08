from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest
from fastapi import HTTPException

from breeze_infer.api import (
    DEFAULT_CFG_SCALE,
    _acquire_request_slot,
    _iter_pcm_chunks,
    _iter_seeded_audio_chunks,
    _pcm16,
    _request_lock,
    app,
    speech,
)
from breeze_infer.api import (
    MAX_NEW_TOKENS as API_MAX_NEW_TOKENS,
)
from breeze_infer.api import (
    MAX_SEQ_LEN as API_MAX_SEQ_LEN,
)
from infer import MAX_NEW_TOKENS as CLI_MAX_NEW_TOKENS
from infer import MAX_SEQ_LEN as CLI_MAX_SEQ_LEN
from models.fast_streaming import FastStreamingChunk


def test_api_exposes_only_health_and_streaming_speech() -> None:
    paths = {route.path for route in app.routes if route.path.startswith("/")}

    assert "/health" in paths
    assert "/v1/audio/speech" in paths
    assert "/api/ref-audio-codes" not in paths


def test_speech_request_parameters_are_minimal() -> None:
    assert list(inspect.signature(speech).parameters) == [
        "text",
        "instruction",
        "cfg_scale",
        "ref_audio",
        "ref_text",
        "seed",
    ]


def test_api_cfg_defaults_to_one() -> None:
    cfg_parameter = inspect.signature(speech).parameters["cfg_scale"]

    assert DEFAULT_CFG_SCALE == 1.0
    assert cfg_parameter.default.default == 1.0


def test_api_instruction_defaults_to_none() -> None:
    instruction_parameter = inspect.signature(speech).parameters["instruction"]

    assert instruction_parameter.default.default is None


def test_cli_and_api_support_1500_generated_tokens() -> None:
    assert CLI_MAX_NEW_TOKENS == API_MAX_NEW_TOKENS == 1500
    assert CLI_MAX_SEQ_LEN == API_MAX_SEQ_LEN == 2048


def test_pcm16_clips_and_encodes_little_endian() -> None:
    encoded = _pcm16(np.array([-2.0, 0.0, 2.0], dtype=np.float32))

    assert np.frombuffer(encoded, dtype="<i2").tolist() == [-32767, 0, 32767]


def _audio_chunk(value: float, *, final: bool = False) -> FastStreamingChunk:
    return FastStreamingChunk(
        audio=np.array([value], dtype=np.float32),
        sample_rate=24000,
        codec_frames=1,
        is_final=final,
    )


def test_pcm_stream_forwards_first_frame_then_coalesces_later_frames() -> None:
    output = list(
        _iter_pcm_chunks(
            iter([_audio_chunk(0.1), _audio_chunk(0.2), _audio_chunk(0.3, final=True)]),
            subsequent_frames=2,
        )
    )

    assert [len(part) for part in output] == [2, 4]


def test_pcm_stream_flushes_incomplete_final_group() -> None:
    output = list(
        _iter_pcm_chunks(
            iter([_audio_chunk(0.1), _audio_chunk(0.2, final=True)]),
            subsequent_frames=4,
        )
    )

    assert [len(part) for part in output] == [2, 2]


def test_request_slot_waits_instead_of_returning_conflict() -> None:
    assert _request_lock.acquire(blocking=False)

    async def exercise() -> None:
        asyncio.get_running_loop().call_later(0.01, _request_lock.release)
        await _acquire_request_slot(timeout=1.0)

    try:
        asyncio.run(exercise())
        assert _request_lock.locked()
    finally:
        if _request_lock.locked():
            _request_lock.release()


def test_request_slot_timeout_is_service_unavailable() -> None:
    assert _request_lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as error:
            asyncio.run(_acquire_request_slot(timeout=0))
        assert error.value.status_code == 503
    finally:
        _request_lock.release()


def test_streaming_reseeds_immediately_before_model_sampling(monkeypatch) -> None:
    events = []

    class Runtime:
        def iter_audio_chunks(
            self, inputs, *, request_id, seed, token_observer
        ):
            events.append(("sample", inputs, request_id, seed, token_observer))
            yield "chunk"

    monkeypatch.setattr(
        "breeze_infer.api.set_all_seeds",
        lambda seed: events.append(("seed", seed)),
    )

    chunks = list(
        _iter_seeded_audio_chunks(
            Runtime(), {"input_ids": "prepared"}, request_id="request-1", seed=43
        )
    )

    assert chunks == ["chunk"]
    assert events == [
        ("seed", 43),
        ("sample", {"input_ids": "prepared"}, "request-1", 43, None),
    ]
