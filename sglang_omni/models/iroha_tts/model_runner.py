# SPDX-License-Identifier: Apache-2.0
"""S2-Pro's runner plus the per-row Iroha-TTS-V1 steering and guidance state."""

from __future__ import annotations

import logging

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni.models.iroha_tts.payload_types import IrohaRequestData
from sglang_omni.models.iroha_tts.sglang_model import IrohaTextModel, UncondBranch
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import ForwardBatch, ScheduleBatch

logger = logging.getLogger(__name__)

# Each guided request holds a private KV cache for its prompt plus max_new_tokens; past this many, new ones run unguided.
MAX_GUIDED_REQUESTS = 8


class IrohaModelRunner(FishS2ProModelRunner):

    model: IrohaTextModel

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.uncond = UncondBranch(self.model)

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        if any(req.rid in self.uncond.states for req in batch.reqs):
            # The guidance branch steps on the previous token, which lookahead resolves a step late.
            return False
        else:
            return super().lookahead_eligible(batch)

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        super().before_prefill(forward_batch, schedule_batch, requests)
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            if data.guidance > 0 and data.req.inflight_middle_chunks == 0:
                self.prefill_uncond(row_idx, data)
            else:
                pass

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        super().before_decode(
            forward_batch, schedule_batch, requests, is_lookahead=is_lookahead
        )
        # Runs after S2-Pro has placed this step's vq_codes, on its sync and async paths alike.
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            if data.guidance > 0 and data.req.rid in self.uncond.states:
                token = forward_batch.input_ids[row_idx]
                logits = self.uncond.step(
                    data.req.rid,
                    token,
                    self.model.vq_codes[row_idx],
                    self.semantic_begin_id <= int(token) <= self.semantic_end_id,
                )
                self.model.uncond_logits[row_idx].copy_(logits)
            else:
                pass
        self.uncond.keep({sched_req.data.req.rid for sched_req in requests})

    def sync_decode_row_state(self, row_idx: int, data: IrohaRequestData) -> None:
        super().sync_decode_row_state(row_idx, data)
        # A middle prefill chunk does not end the prompt: no steering, no guidance.
        middle = data.req.inflight_middle_chunks > 0
        if self.model.steer_emotions:
            self.model.steer_index[row_idx] = self.model.steer_emotions.get(
                data.steer_emotion, 0
            )
            self.model.steer_alpha[row_idx] = 0.0 if middle else data.steer_alpha
        else:
            pass
        self.model.cfg_weight[row_idx] = 0.0 if middle else data.guidance

    def prefill_uncond(self, row_idx: int, data: IrohaRequestData) -> None:
        rid = data.req.rid
        if (
            rid not in self.uncond.states
            and len(self.uncond.states) >= MAX_GUIDED_REQUESTS
        ):
            logger.warning(
                "%d guided requests already running; %s runs without guidance",
                len(self.uncond.states),
                rid,
            )
            data.guidance = 0.0
            self.model.cfg_weight[row_idx] = 0.0
        else:
            capacity = len(data.uncond_input_ids) + int(data.max_new_tokens) + 1
            logits = self.uncond.prefill(
                rid,
                data.uncond_input_ids,
                data.uncond_vq_mask_tokens,
                data.vq_parts,
                capacity,
            )
            self.model.uncond_logits[row_idx].copy_(logits)
