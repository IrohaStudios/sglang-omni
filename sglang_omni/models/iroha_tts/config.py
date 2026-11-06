# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Iroha-TTS-V1: S2-Pro's pipeline with Iroha's preprocessing and engine."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import StageConfig
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig

PACKAGE = "sglang_omni.models.iroha_tts"
FACTORIES = {
    "preprocessing": f"{PACKAGE}.stages.create_preprocessing_executor",
    "tts_engine": f"{PACKAGE}.stages.create_sglang_tts_engine_executor",
}


def iroha_stages() -> list[StageConfig]:
    """S2-Pro's stages with Iroha's preprocessing and engine; the vocoder stays S2-Pro's."""
    return [
        stage.model_copy(
            update={"factory_path": FACTORIES.get(stage.name, stage.factory_path)},
            deep=True,
        )
        for stage in S2ProPipelineConfig.model_fields["stages"].default
    ]


class IrohaTTSPipelineConfig(S2ProPipelineConfig):
    # No checkpoint names this architecture: examples/configs/iroha_tts.yaml picks the pipeline by
    # class name, and without it a merged checkpoint is served as plain S2-Pro.
    architecture: ClassVar[str] = "IrohaTTS"

    stages: list[StageConfig] = Field(default_factory=iroha_stages)


EntryClass = IrohaTTSPipelineConfig
