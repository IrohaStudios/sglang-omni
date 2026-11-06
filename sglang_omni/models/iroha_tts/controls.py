# SPDX-License-Identifier: Apache-2.0
"""Iroha-TTS-V1 request controls: whole-line emotion, intensity and guidance on control tags.

The rules port Iroha's emotts package (engine.py, steer.py): the same request validation and defaults.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

EMOTION_TAGS = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "surprise": "surprised",
}
EMOTIONS = ("none", "neutral", *EMOTION_TAGS)
LEVELS = ("weakest", "weak", "default", "strong")
CONTROL_TAGS = ("cry", "laugh", "sigh", "whisper")  # tags guidance can strengthen
TAG_RE = re.compile(r"\[([^\]]*)\]")
CONTROL_RE = re.compile(r"\[(" + "|".join(CONTROL_TAGS) + r")\]")
LINE_EMOTION_RE = re.compile(r"^\s*\[(" + "|".join(EMOTION_TAGS.values()) + r")\]")


@dataclass
class IrohaConfig:
    intensity: dict | None
    guidance: dict[str, float]
    vectors: Path | None

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> IrohaConfig:
        """Read iroha.json, which a merged Iroha-TTS-V1 checkpoint carries next to the S2-Pro weights."""
        path = Path(checkpoint_dir) / "iroha.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; the Iroha-TTS pipeline needs a merged Iroha-TTS-V1 checkpoint"
            )
        else:
            pass
        cfg = json.loads(path.read_text(encoding="utf-8"))
        intensity = cfg.get("intensity")
        vectors = Path(checkpoint_dir) / intensity["file"] if intensity else None
        guidance = {str(t): float(w) for t, w in (cfg.get("guidance") or {}).items()}
        return cls(intensity=intensity, guidance=guidance, vectors=vectors)


@dataclass
class LineControls:
    text: str  # prompt text, whole-line emotion tag included
    uncond_text: str | None  # the same line without the guided tags, when guided
    emotion: str | None  # emotion to steer, when alpha is nonzero
    alpha: float
    guidance: float


def parse_intensity(intensity: object) -> str | float | None:
    """A preset name or a finite number, never snapped to a preset."""
    if intensity is None:
        return None
    else:
        pass
    if isinstance(intensity, str) and intensity.strip() in LEVELS:
        return intensity.strip()
    else:
        pass
    if isinstance(intensity, bool) or not isinstance(intensity, (str, Real)):
        raise ValueError(
            f"intensity must be one of {LEVELS} or a finite number, not {intensity!r}"
        )
    else:
        pass
    try:
        alpha = float(intensity)
    except (ValueError, OverflowError):
        raise ValueError(
            f"intensity must be one of {LEVELS} or a finite number, not {intensity!r}"
        ) from None
    if not math.isfinite(alpha):
        raise ValueError("intensity must be finite")
    else:
        pass
    return alpha


def resolve_intensity(
    cfg: dict | None, line: str, emotion: str | None, intensity: object
) -> float:
    """alpha for a request; zero is a no-op, nonzero needs one explicit whole-line emotion and no other tags."""
    level = parse_intensity(intensity)
    if level is None or level == "default" or level == 0.0:
        return 0.0
    else:
        pass
    if cfg is None:
        raise ValueError("this model has no intensity control")
    else:
        pass
    levels = cfg["levels"].get(emotion)
    if not levels:
        raise ValueError(
            f"intensity needs a whole-line emotion with levels ({sorted(cfg['levels'])})"
        )
    else:
        pass
    alpha = float(levels[level]) if isinstance(level, str) else level
    lower, upper = float(levels["weakest"]), float(levels["strong"])
    if not lower <= alpha <= upper:
        raise ValueError(
            f"intensity alpha {alpha:g} is outside [{lower:g}, {upper:g}] for {emotion}"
        )
    else:
        pass
    if any(tag in EMOTION_TAGS.values() for tag in TAG_RE.findall(line)):
        raise ValueError(
            "nonzero intensity needs emotion= with no inline emotion tags or transitions"
        )
    else:
        pass
    if CONTROL_RE.search(line):
        raise ValueError(
            "intensity together with [laugh] [sigh] [cry] [whisper] is not supported"
        )
    else:
        pass
    return alpha


def guidance_for(
    line: str, guidance: float | None, defaults: dict[str, float]
) -> tuple[float, str | None]:
    """(w, line without the guided tags), or (0, None). None uses the config defaults; a number guides every tag."""
    if guidance is None:
        guided = sorted({t for t in CONTROL_RE.findall(line) if defaults.get(t, 0) > 0})
        if not guided:
            return 0.0, None
        else:
            pass
        strip = re.compile(r"\[(" + "|".join(map(re.escape, guided)) + r")\]")
        return max(defaults[t] for t in guided), strip.sub("", line)
    else:
        pass
    weight = float(guidance)
    if not (math.isfinite(weight) and weight >= 0):
        raise ValueError(f"guidance must be a finite number >= 0, not {guidance!r}")
    else:
        pass
    if weight > 0 and CONTROL_RE.search(line):
        return weight, CONTROL_RE.sub("", line)
    else:
        return 0.0, None


def prepare_line(
    cfg: IrohaConfig,
    text: str,
    emotion: str | None,
    intensity: object,
    guidance: float | None,
) -> LineControls:
    """emotts EmotionTTS.build_text + resolve_intensity + guidance_for; readings stay with the caller."""
    line = text.strip()
    emotion = emotion or "none"
    if emotion not in EMOTIONS:
        raise ValueError(f"emotion must be one of {list(EMOTIONS)}, not {emotion!r}")
    else:
        pass
    whole = (
        None
        if emotion in ("none", "neutral") or LINE_EMOTION_RE.match(line)
        else emotion
    )
    alpha = resolve_intensity(cfg.intensity, line, whole, intensity)
    weight, uncond = guidance_for(line, guidance, cfg.guidance)
    prefix = f"[{EMOTION_TAGS[whole]}]" if whole else ""
    return LineControls(
        text=prefix + line,
        uncond_text=prefix + uncond if weight > 0 else None,
        emotion=whole if alpha else None,
        alpha=alpha,
        guidance=weight,
    )
