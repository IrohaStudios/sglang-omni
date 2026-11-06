# SPDX-License-Identifier: Apache-2.0
"""S2-Pro's text model with Iroha-TTS-V1 emotion steering and tag guidance.

Steering ``h <- renorm(h + alpha * v)`` runs as forward hooks on the steered layers, at each sequence's last
position; guidance ``l_c + w (l_c - l_u)`` mixes the semantic logits before S2-Pro samples them. The unconditional
logits come from UncondBranch, which runs the same weights outside SGLang's KV pool with a private cache per request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
from safetensors.torch import load_file
from torch.nn import functional

from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel
from sglang_omni.models.iroha_tts.controls import IrohaConfig


class IrohaTextModel(S2ProSGLangTextModel):

    def setup_iroha(self, cfg: IrohaConfig, max_batch_size: int) -> None:
        """Attach the steering vectors, hooks and guidance buffers (before graph capture)."""
        device = self.embed_tokens.weight.device
        self.steer_emotions: dict[str, int] = {}
        if cfg.intensity is not None:
            layers = [int(i) for i in cfg.intensity["layers"]]
            emotions = list(cfg.intensity["levels"])
            raw = load_file(str(cfg.vectors))
            vectors = torch.zeros(
                len(emotions) + 1, len(layers), self.hidden_size, device=device
            )  # row 0: no steering
            for e, emotion in enumerate(emotions, start=1):
                for slot, layer in enumerate(layers):
                    v = raw[f"{emotion}.{layer}"].float()
                    norm = float(cfg.intensity["act_norm"][str(layer)])
                    vectors[e, slot] = v / (v.norm() + 1e-6) * norm
            self.steer_vectors = vectors
            self.steer_emotions = {
                emotion: e for e, emotion in enumerate(emotions, start=1)
            }
            self.steer_index = torch.zeros(
                max_batch_size, dtype=torch.long, device=device
            )
            self.steer_alpha = torch.zeros(max_batch_size, device=device)
            for slot, layer in enumerate(layers):
                self.layers[layer].register_forward_hook(partial(self.steer, slot))
        else:
            pass
        self.cfg_weight = torch.zeros(max_batch_size, device=device)
        self.uncond_logits = torch.zeros(max_batch_size, self.vocab_size, device=device)

    def steer(self, slot: int, layer, args, output):
        forward_batch = args[2]
        last_index = None
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        else:
            pass
        return steer_last_tokens(self, *output, last_index, slot)

    @torch.no_grad()
    def decode_codebooks(self, logits: torch.Tensor, hidden_states: torch.Tensor):
        # Guidance: l_c + w (l_c - l_u); rows with w = 0 keep their logits.
        bs = logits.shape[0]
        weight = self.cfg_weight[:bs].unsqueeze(1)
        cond = logits.float()
        guided = cond + weight * (cond - self.uncond_logits[:bs])
        return super().decode_codebooks(
            torch.where(weight > 0, guided, cond), hidden_states
        )


def steer_last_tokens(
    model,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    last_index: torch.Tensor | None,
    slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """renorm(h + alpha v) on each sequence's last token, h = hidden + residual (the residual stream after the layer).

    Decode passes last_index=None (one token per row, CUDA-graph safe). Rows with alpha 0 are left untouched.
    """
    tokens = slice(None) if last_index is None else last_index
    h = hidden[tokens].float() + residual[tokens].float()
    rows = h.shape[0]
    alpha = model.steer_alpha[:rows].unsqueeze(-1)
    moved = h + alpha * model.steer_vectors[model.steer_index[:rows], slot]
    moved = moved * (
        h.norm(dim=-1, keepdim=True) / moved.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    )
    on = alpha != 0
    residual[tokens] = torch.where(on, moved.to(residual.dtype), residual[tokens])
    hidden[tokens] = torch.where(on, torch.zeros_like(hidden[tokens]), hidden[tokens])
    return hidden, residual


@dataclass
class UncondState:
    k: torch.Tensor  # [layers, capacity, kv_heads, head_dim]
    v: torch.Tensor
    length: int = 0


@dataclass
class UncondBranch:
    """The slow transformer on a guided request's tag-stripped prompt, with a private KV cache per request.

    emotts runs this branch beside the tagged one every frame and feeds it the same sampled token; here it runs before
    each prefill / decode step and leaves its last logits in model.uncond_logits for the captured decode graph.
    """

    model: object
    states: dict[str, UncondState] = field(default_factory=dict)
    fresh: set[str] = field(default_factory=set)  # prefilled since the last decode step

    @torch.no_grad()
    def prefill(
        self,
        rid: str,
        input_ids: torch.Tensor,
        vq_mask: torch.Tensor | None,
        vq_parts: list[torch.Tensor] | None,
        capacity: int,
    ) -> torch.Tensor:
        model = self.model
        device = model.embed_tokens.weight.device
        ids = input_ids.to(device)
        embeds = model.embed_tokens(ids)
        if vq_mask is not None and vq_parts and bool(vq_mask.any()):
            mask = vq_mask.to(device).reshape(-1)
            parts = torch.cat([p.to(device).T for p in vq_parts if p.dim() == 2], dim=0)
            vq = model.audio_decoder.embed_text_dim(
                embeds.unsqueeze(0), parts, mask.unsqueeze(0)
            )
            embeds[mask.nonzero(as_tuple=True)[0]] = vq.to(embeds.dtype)
        else:
            pass
        attn = model.layers[0].self_attn
        shape = (len(model.layers), capacity, attn.num_kv_heads, attn.head_dim)
        state = UncondState(
            k=torch.empty(shape, dtype=embeds.dtype, device=device),
            v=torch.empty(shape, dtype=embeds.dtype, device=device),
        )
        self.states[rid] = state
        self.fresh.add(rid)
        return self.run(state, embeds)

    @torch.no_grad()
    def step(
        self, rid: str, token: torch.Tensor, vq_codes: torch.Tensor, is_semantic: bool
    ) -> torch.Tensor:
        model = self.model
        embeds = model.embed_tokens(token.reshape(1))
        if is_semantic:
            parts = model.vq_codebook_embeddings(
                vq_codes.reshape(1, -1) + model.vq_codebook_offsets[None, :]
            )
            embeds = (embeds + parts.sum(dim=1).to(embeds.dtype)) * model.vq_scale
        else:
            pass
        return self.run(self.states[rid], embeds)

    def keep(self, rids: set[str]) -> None:
        """Free the caches of requests that finished or were retracted (a retracted request prefills again)."""
        live = rids | self.fresh
        for rid in [rid for rid in self.states if rid not in live]:
            del self.states[rid]
        self.fresh.clear()

    @torch.no_grad()
    def run(self, state: UncondState, embeds: torch.Tensor) -> torch.Tensor:
        """Last-position logits (float32, full vocab) after appending embeds; mirrors S2ProDecoderLayer.forward."""
        from sglang_omni.vendor.sglang.models import apply_qk_norm

        model = self.model
        start, n = state.length, embeds.shape[0]
        if n > 1 and start != 0:
            raise ValueError("the unconditional branch prefills once, from position 0")
        else:
            pass
        positions = torch.arange(start, start + n, device=embeds.device)
        hidden, residual = embeds, None
        for i, layer in enumerate(model.layers):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(hidden)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            if attn.qk_norm:
                q, k = apply_qk_norm(q, k, attn.q_norm, attn.k_norm, attn.head_dim)
            else:
                pass
            q, k = attn.rotary_emb(positions, q, k)
            state.k[i, start : start + n] = k.reshape(
                n, attn.num_kv_heads, attn.head_dim
            )
            state.v[i, start : start + n] = v.reshape(
                n, attn.num_kv_heads, attn.head_dim
            )
            out = functional.scaled_dot_product_attention(
                q.reshape(n, attn.num_heads, attn.head_dim)
                .transpose(0, 1)
                .unsqueeze(0),
                state.k[i, : start + n].transpose(0, 1).unsqueeze(0),
                state.v[i, : start + n].transpose(0, 1).unsqueeze(0),
                is_causal=n > 1,
                scale=attn.scaling,
                enable_gqa=True,
            )
            out, _ = attn.o_proj(
                out.squeeze(0)
                .transpose(0, 1)
                .reshape(n, attn.num_heads * attn.head_dim)
            )
            hidden, residual = layer.post_attention_layernorm(out, residual)
            gate, up = layer.gate_up_proj(hidden)[0].chunk(2, dim=-1)
            hidden, _ = layer.down_proj(functional.silu(gate) * up)
        state.length = start + n
        hidden, _ = model.norm(hidden, residual)
        return functional.linear(hidden[-1:], model.embed_tokens.weight)[0].float()
