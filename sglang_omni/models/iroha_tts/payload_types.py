# SPDX-License-Identifier: Apache-2.0
"""S2-Pro's pipeline state and request data, plus the Iroha-TTS-V1 controls."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.request_builders import S2ProSGLangRequestData
from sglang_omni.scheduling.pipeline_state import wire

if TYPE_CHECKING:
    import numpy.typing as npt
else:
    pass


@dataclass
class IrohaState(S2ProState):
    steer_emotion: str | None = None
    steer_alpha: float = 0.0
    guidance: float = 0.0
    uncond_input_ids: torch.Tensor | npt.ArrayLike | None = wire(
        None, codec="tensor_list"
    )  # guidance branch: the prompt without the guided tags
    uncond_vq_mask_tokens: torch.Tensor | list[bool] | None = wire(
        None, codec="tensor_list"
    )


@dataclass
class IrohaRequestData(S2ProSGLangRequestData):
    steer_emotion: str | None = None
    steer_alpha: float = 0.0
    guidance: float = 0.0
    uncond_input_ids: torch.Tensor | None = None
    uncond_vq_mask_tokens: torch.Tensor | None = None

    @classmethod
    def from_s2pro(
        cls, data: S2ProSGLangRequestData, state: IrohaState
    ) -> IrohaRequestData:
        """The request S2-Pro built, with the controls the preprocessing stage resolved."""
        return cls(
            **{f.name: getattr(data, f.name) for f in fields(data) if f.init},
            steer_emotion=state.steer_emotion,
            steer_alpha=state.steer_alpha,
            guidance=state.guidance,
            uncond_input_ids=(
                None
                if state.uncond_input_ids is None
                else torch.tensor(list(state.uncond_input_ids), dtype=torch.long)
            ),
            uncond_vq_mask_tokens=(
                None
                if state.uncond_vq_mask_tokens is None
                else torch.as_tensor(state.uncond_vq_mask_tokens, dtype=torch.bool)
            ),
        )
