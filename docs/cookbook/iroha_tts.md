# Iroha-TTS-V1

Iroha-TTS-V1 is a Japanese emotional TTS model fine-tuned from [Fish Audio S2-Pro](fishaudio_s2_pro.md). Merged into the S2-Pro weights, it runs on the S2-Pro pipeline and adds two request controls: emotion intensity, and guidance for control tags such as `[whisper]`.

## Server Configuration

Install the S2-Pro dependencies from [Fish Audio S2-Pro](fishaudio_s2_pro.md#prerequisites). The checkpoint folder holds the merged weights, `iroha.json` and the intensity vectors.

```bash
sgl-omni serve \
  --model-path /path/to/iroha-tts-v1-merged \
  --config examples/configs/iroha_tts.yaml \
  --port 8000
```

Without this config the same folder is served as plain S2-Pro, without the controls below.

## Request Controls

Requests take the S2-Pro [request parameters](fishaudio_s2_pro.md#request-parameters), plus these under `stage_params.preprocessing`:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "やっと会えたね！",
    "stage_params": {"preprocessing": {"emotion": "joy", "intensity": "strong"}}
  }' \
  --output joy.wav
```

| Parameter | Description |
|---|---|
| `emotion` | Whole-line emotion: `joy`, `anger`, `sadness`, `surprise`, `neutral` or `none` |
| `intensity` | `weakest`, `weak`, `default`, `strong`, or a number in that range. Needs `emotion` and no other tags in the text |
| `guidance` | Strength for `[laugh]`, `[sigh]`, `[cry]` and `[whisper]` in the text. Defaults to the model config, which guides `[whisper]`; `0` turns it off |

Write readings in kana; `漢字[かな]` is not converted. Guided requests decode synchronously and slow the batch they share; at most 8 run with guidance at once, and later ones run without it.
