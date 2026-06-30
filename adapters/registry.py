"""Adapter registry — build OfflineSTTAdapters by short name.

Used by benchmark/run.py so model selection lives in one place. The local IndicConformer
is constructed lazily (it loads a multi-GB ONNX model); API adapters are cheap to
construct but only work if their *_API_KEY is set.

(Helper module: one place to select which systems run.)
"""
from __future__ import annotations

from typing import Callable

from adapters.offline_base import OfflineSTTAdapter

ALL_SYSTEMS = ("indicconformer", "sarvam", "smallest", "cartesia", "gnani")


def _make_indicconformer() -> OfflineSTTAdapter:
    from adapters.indicconformer_local import IndicConformerLocalAdapter

    return IndicConformerLocalAdapter()


def _make_sarvam() -> OfflineSTTAdapter:
    from adapters.sarvam_api import SarvamOfflineAdapter

    return SarvamOfflineAdapter()


def _make_smallest() -> OfflineSTTAdapter:
    from adapters.smallest_api import SmallestOfflineAdapter

    return SmallestOfflineAdapter()


def _make_cartesia() -> OfflineSTTAdapter:
    from adapters.cartesia_api import CartesiaOfflineAdapter

    return CartesiaOfflineAdapter()


def _make_gnani() -> OfflineSTTAdapter:
    from adapters.gnani_api import GnaniOfflineAdapter

    return GnaniOfflineAdapter()


_FACTORIES: dict[str, Callable[[], OfflineSTTAdapter]] = {
    "indicconformer": _make_indicconformer,
    "sarvam": _make_sarvam,
    "smallest": _make_smallest,
    "cartesia": _make_cartesia,
    "gnani": _make_gnani,
}


def build_adapter(name: str) -> OfflineSTTAdapter:
    name = name.lower()
    if name not in _FACTORIES:
        raise KeyError(f"unknown system {name!r}; choose from {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


def build_adapters(names: list[str]) -> dict[str, OfflineSTTAdapter]:
    return {n: build_adapter(n) for n in names}
