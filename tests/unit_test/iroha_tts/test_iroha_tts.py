# SPDX-License-Identifier: Apache-2.0
"""Iroha-TTS-V1: request rules ported from emotts, the steering formula, and the pipeline wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.iroha_tts import controls
from sglang_omni.models.iroha_tts.config import IrohaTTSPipelineConfig
from sglang_omni.models.iroha_tts.model_runner import (
    MAX_GUIDED_REQUESTS,
    IrohaModelRunner,
)
from sglang_omni.models.iroha_tts.sglang_model import steer_last_tokens
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

LEVELS = {"weakest": -0.06, "weak": -0.03, "default": 0.0, "strong": 0.045}
CFG = controls.IrohaConfig(
    intensity={"levels": {"joy": LEVELS, "sadness": LEVELS}},
    guidance={"whisper": 1.5},
    vectors=None,
)


def test_emotion_and_intensity_resolve_like_emotts() -> None:
    line = controls.prepare_line(CFG, " やっと会えたね！", "joy", "strong", None)
    assert (line.text, line.emotion, line.alpha, line.guidance) == (
        "[happy]やっと会えたね！",
        "joy",
        0.045,
        0.0,
    )
    assert controls.prepare_line(CFG, "x", "joy", -0.015, None).alpha == -0.015
    neutral = controls.prepare_line(CFG, "x", "neutral", "default", None)
    assert (neutral.text, neutral.emotion, neutral.alpha) == ("x", None, 0.0)


@pytest.mark.parametrize(
    "text,emotion,intensity",
    [
        ("x", None, "strong"),  # no whole-line emotion
        (
            "[happy]x",
            "joy",
            "strong",
        ),  # inline emotion tag wins, so no explicit emotion is left
        ("x[laugh]", "joy", "strong"),  # events cannot be steered
        ("x", "joy", 0.1),  # outside [weakest, strong]
        ("x", "joy", True),
        ("x", "anger", "strong"),  # no levels configured for this emotion
    ],
)
def test_unsupported_intensity_is_rejected(
    text: str, emotion: str | None, intensity: object
) -> None:
    with pytest.raises(ValueError):
        controls.prepare_line(CFG, text, emotion, intensity, None)


def test_guidance_defaults_to_whisper_only() -> None:
    whisper = controls.prepare_line(CFG, "[whisper]ここだけの話", None, None, None)
    assert (whisper.guidance, whisper.uncond_text) == (1.5, "ここだけの話")
    assert controls.prepare_line(CFG, "[laugh]ふふ", None, None, None).guidance == 0.0
    laugh = controls.prepare_line(CFG, "[laugh]ふふ", "joy", None, 2.0)
    assert (laugh.guidance, laugh.uncond_text) == (2.0, "[happy]ふふ")
    assert controls.prepare_line(CFG, "[whisper]x", None, None, 0).guidance == 0.0


def test_steering_matches_emotts_renorm_on_last_tokens() -> None:
    torch.manual_seed(0)
    model = SimpleNamespace(
        steer_vectors=torch.randn(3, 1, 8),
        steer_index=torch.tensor([1, 2, 0]),
        steer_alpha=torch.tensor([0.5, -0.2, 0.0]),
    )
    hidden, residual = torch.randn(7, 8), torch.randn(7, 8)
    last = torch.tensor([2, 4, 6])  # three sequences of an extend batch
    before = hidden + residual
    hidden, residual = steer_last_tokens(
        model, hidden.clone(), residual.clone(), last, 0
    )
    after = hidden + residual
    for row, token in enumerate(last.tolist()[:2]):
        h = before[token]
        moved = (
            h + model.steer_alpha[row] * model.steer_vectors[model.steer_index[row], 0]
        )
        expected = moved * (h.norm() / moved.norm())  # emotts steer.renorm
        assert torch.allclose(after[token], expected, atol=1e-5)
    untouched = [0, 1, 3, 5, 6]  # not a last token, or alpha 0
    assert torch.equal(after[untouched], before[untouched])


def test_guided_requests_past_the_cap_run_unguided() -> None:
    runner = object.__new__(IrohaModelRunner)
    runner.model = SimpleNamespace(cfg_weight=torch.ones(2))
    full = {f"r{i}": None for i in range(MAX_GUIDED_REQUESTS)}
    runner.uncond = SimpleNamespace(states=full)
    data = SimpleNamespace(req=SimpleNamespace(rid="new"), guidance=3.0)
    runner.prefill_uncond(1, data)
    assert data.guidance == 0.0
    assert runner.model.cfg_weight.tolist() == [1.0, 0.0]


def test_guided_rows_keep_async_decode_off() -> None:
    runner = object.__new__(IrohaModelRunner)
    runner.async_enabled = True
    runner.model = SimpleNamespace(
        async_decode_state_enabled=True,
        vq_codes=SimpleNamespace(device=torch.device("cuda")),
    )
    runner.uncond = SimpleNamespace(states={"guided": None})
    runner.last_decode_request_ids = ("guided",)
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="guided")])
    assert not runner.lookahead_eligible(batch)


def test_pipeline_swaps_preprocessing_and_engine_only() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name("IrohaTTSPipelineConfig")
        is IrohaTTSPipelineConfig
    )
    stages = {
        s.name: s.factory_path for s in IrohaTTSPipelineConfig(model_path="x").stages
    }
    assert stages == {
        "preprocessing": "sglang_omni.models.iroha_tts.stages.create_preprocessing_executor",
        "tts_engine": "sglang_omni.models.iroha_tts.stages.create_sglang_tts_engine_executor",
        "vocoder": "sglang_omni.models.fishaudio_s2_pro.stages.create_vocoder_executor",
    }
