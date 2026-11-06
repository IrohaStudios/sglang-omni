# SPDX-License-Identifier: Apache-2.0
"""Iroha-TTS-V1 stage factories: S2-Pro's preprocessing and engine with the Iroha controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from sglang_omni.models.fishaudio_s2_pro import stages as fish_stages
from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.iroha_tts.controls import IrohaConfig, prepare_line
from sglang_omni.models.iroha_tts.payload_types import IrohaState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.utils.checkpoint import resolve_checkpoint


def with_text(payload: StagePayload, inputs: dict, text: str) -> StagePayload:
    return replace(
        payload, request=replace(payload.request, inputs={**inputs, "text": text})
    )


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 8,
):
    """S2-Pro's preprocessing, run on the line the controls rewrite.

    The controls come in stage_params={"preprocessing": {"emotion", "intensity", "guidance"}}. A guided request runs
    S2-Pro's preprocessing a second time on the line without the guided tags; its references hit the encode cache.
    """
    scheduler = fish_stages.create_preprocessing_executor(
        model_path, device=device, gpu_id=gpu_id, max_concurrency=max_concurrency
    )
    cfg = IrohaConfig.load(resolve_checkpoint(model_path))
    preprocess_s2pro = scheduler.fn

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}
        else:
            pass
        params = payload.request.params or {}
        controls = (params.get("stage_params") or {}).get("preprocessing") or {}
        line = prepare_line(
            cfg,
            inputs.get("text", ""),
            controls.get("emotion"),
            controls.get("intensity"),
            controls.get("guidance"),
        )
        state = IrohaState.from_dict(
            preprocess_s2pro(with_text(payload, inputs, line.text)).data
        )
        state.steer_emotion = line.emotion
        state.steer_alpha = line.alpha
        state.guidance = line.guidance
        if line.uncond_text is not None:
            uncond = S2ProState.from_dict(
                preprocess_s2pro(with_text(payload, inputs, line.uncond_text)).data
            )
            state.uncond_input_ids = uncond.input_ids
            state.uncond_vq_mask_tokens = uncond.vq_mask_tokens
        else:
            pass
        return store_state(payload, state)

    scheduler.fn = preprocess
    return scheduler


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_new_tokens: int = 2048,
    top_k: int = 30,
    ras_window: int = 16,
    enable_async_decode: bool = False,
    server_args_overrides: Mapping[str, object] | None = None,
):
    """S2-Pro's engine factory, building IrohaEngineBuilder."""
    del top_k
    from sglang_omni.models.iroha_tts.engine_builder import IrohaEngineBuilder

    return IrohaEngineBuilder(
        max_new_tokens=max_new_tokens,
        ras_window=ras_window,
        enable_async_decode=enable_async_decode,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )
