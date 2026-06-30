"""Manifest loader.

manifest.jsonl, one clip per line:
    {"audio_path": "...", "ref_text": "...", "lang": "hi",
     "contains_terms": ["..."], "partition": "term" | "noterm"}

The noterm partition is MANDATORY — a benchmark that only measures recall cannot
detect overbias (see metrics.false_insertion_rate).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal


Partition = Literal["term", "noterm"]


@dataclass
class Clip:
    audio_path: str
    ref_text: str
    lang: str
    partition: Partition
    contains_terms: list[str] = field(default_factory=list)

    def resolve(self, base_dir: Path) -> "Clip":
        """Resolve audio_path relative to the manifest's directory if not absolute."""
        p = Path(self.audio_path)
        if not p.is_absolute():
            self.audio_path = str((base_dir / p).resolve())
        return self


def load_manifest(path: str | Path) -> list[Clip]:
    path = Path(path)
    base = path.parent
    clips: list[Clip] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} — invalid JSON: {e}") from e
            partition = obj.get("partition")
            if partition not in ("term", "noterm"):
                raise ValueError(
                    f"{path}:{lineno} — partition must be 'term' or 'noterm', got {partition!r}"
                )
            clips.append(
                Clip(
                    audio_path=obj["audio_path"],
                    ref_text=obj["ref_text"],
                    lang=obj.get("lang", "hi"),
                    partition=partition,
                    contains_terms=list(obj.get("contains_terms", [])),
                ).resolve(base)
            )
    _warn_if_no_noterm(clips, path)
    return clips


def _warn_if_no_noterm(clips: list[Clip], path: Path) -> None:
    if clips and not any(c.partition == "noterm" for c in clips):
        import warnings

        warnings.warn(
            f"{path} has no 'noterm' clips — false-insertion (overbias) cannot be "
            "measured. Add non-term-bearing clips.",
            stacklevel=2,
        )


def iter_partition(clips: list[Clip], partition: Partition) -> Iterator[Clip]:
    return (c for c in clips if c.partition == partition)
