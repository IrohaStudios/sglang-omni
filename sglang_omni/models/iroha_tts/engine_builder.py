# SPDX-License-Identifier: Apache-2.0
"""S2-Pro's engine with Iroha-TTS-V1's text model, runner and request data."""

from __future__ import annotations

from collections.abc import Callable

from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fishaudio_s2_pro.engine_builder import FishS2ProEngineBuilder
from sglang_omni.models.iroha_tts.controls import IrohaConfig
from sglang_omni.models.iroha_tts.model_runner import IrohaModelRunner
from sglang_omni.models.iroha_tts.payload_types import IrohaRequestData, IrohaState
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class IrohaEngineBuilder(FishS2ProEngineBuilder):
    model_name = "Iroha-TTS-V1"
    model_arch_override = "IrohaTextModel"

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        from sglang.srt.runtime_context import get_schedule

        super().setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )
        model_worker.model_runner.model.setup_iroha(
            IrohaConfig.load(checkpoint_dir),
            max_batch_size=get_schedule().max_running_requests,
        )

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> IrohaModelRunner:
        return IrohaModelRunner(model_worker, output_proc)

    def make_adapters(self, model: object) -> tuple[
        Callable[[StagePayload], IrohaRequestData],
        Callable[[IrohaRequestData], StagePayload],
    ]:
        request_builder, result_adapter = super().make_adapters(model)

        def iroha_request_builder(payload: StagePayload) -> IrohaRequestData:
            return IrohaRequestData.from_s2pro(
                request_builder(payload), IrohaState.from_dict(payload.data)
            )

        return iroha_request_builder, result_adapter
