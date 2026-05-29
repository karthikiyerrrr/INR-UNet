"""Lightweight name->constructor registry and a config-driven model factory."""

from __future__ import annotations

from typing import Callable, TypeVar

import torch.nn as nn
from omegaconf import DictConfig

T = TypeVar("T")


class Registry:
    """Maps string keys to constructors so components can be built from config."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, Callable[..., object]] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator that registers a class/function under ``key``."""

        def decorator(obj: T) -> T:
            if key in self._registry:
                raise KeyError(f"{key!r} already registered in registry {self.name!r}")
            self._registry[key] = obj  # type: ignore[assignment]
            return obj

        return decorator

    def get(self, key: str) -> Callable[..., object]:
        if key not in self._registry:
            raise KeyError(f"{key!r} not found in registry {self.name!r}")
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self) -> list[str]:
        return list(self._registry)


BLOCKS = Registry("blocks")
ENCODERS = Registry("encoders")
DECODERS = Registry("decoders")
MODELS = Registry("models")


def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate a model from an experiment config via the MODELS registry."""
    model_cls = MODELS.get(cfg.model.name)
    return model_cls(cfg.model)  # type: ignore[return-value]
