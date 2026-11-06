# SPDX-License-Identifier: Apache-2.0
"""Iroha-TTS-V1 on the S2-Pro pipeline, with emotion intensity and tag guidance."""

from sglang_omni.models.fishaudio_s2_pro import CAPABILITIES

from . import config

__all__ = ["CAPABILITIES", "config"]
