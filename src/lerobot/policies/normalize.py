from typing import Any

import numpy as np

import torch

# Fake implementation of Normalization and Unnormalization (for now)

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

def _bufname(key: str) -> str:
    return "buffer_" + key.replace(".", "_")

class Normalize:
    def __init__(self, features, norm_map, stats=None) -> None:
        self.features = features
        self.norm_map = norm_map
        self.stats = stats or {}
        self._buffers = {}
        for key in self.features.keys():
            s = self.stats.get(key, None)
            if s is None:
                continue
            self._buffers[_bufname(key)] = {k: np.asarray(v) for k, v in s.items()}

    def forward(self, batch):
        out = dict(batch)
        for key, ft in self.features.items():
            if key not in out:
                continue
            mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if mode is NormalizationMode.IDENTITY:
                continue
            buf = self._buffers.get(_bufname(key), None)
            if buf is None:
                raise AssertionError(f"Missing stats buffer for '{key}'")
            x = out[key]
            if mode is NormalizationMode.MEAN_STD:
                mean = buf["mean"]; std = buf["std"]
                assert not np.isinf(mean).any(), "Invalid mean (inf)."
                assert not np.isinf(std).any(), "Invalid std (inf)."
                out[key] = (x - mean) / (std + 1e-8)
            elif mode is NormalizationMode.MIN_MAX:
                mn = buf["min"]; mx = buf["max"]
                assert not np.isinf(mn).any(), "Invalid min (inf)."
                assert not np.isinf(mx).any(), "Invalid max (inf)."
                out[key] = (x - mn) / (mx - mn + 1e-8)
                out[key] = out[key] * 2.0 - 1.0
            else:
                raise ValueError(mode)
            
            out[key] = out[key].to(torch.float32)
        return out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Unnormalize:
    def __init__(self, features, norm_map, stats=None) -> None:
        self.features = features
        self.norm_map = norm_map
        self.stats = stats or {}
        self._buffers = {}
        for key in self.features.keys():
            s = self.stats.get(key, None)
            if s is None:
                continue
            self._buffers[_bufname(key)] = {k: np.asarray(v) for k, v in s.items()}

    def forward(self, batch):
        out = dict(batch)
        for key, ft in self.features.items():
            if key not in out:
                continue
            mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if mode is NormalizationMode.IDENTITY:
                continue
            buf = self._buffers.get(_bufname(key), None)
            if buf is None:
                raise AssertionError(f"Missing stats buffer for '{key}'")
            x = out[key]
            if mode is NormalizationMode.MEAN_STD:
                mean = buf["mean"]; std = buf["std"]
                assert not np.isinf(mean).any(), "Invalid mean (inf)."
                assert not np.isinf(std).any(), "Invalid std (inf)."
                out[key] = x * std + mean
            elif mode is NormalizationMode.MIN_MAX:
                mn = buf["min"]; mx = buf["max"]
                assert not np.isinf(mn).any(), "Invalid min (inf)."
                assert not np.isinf(mx).any(), "Invalid max (inf)."
                out[key] = (x + 1.0) / 2.0
                out[key] = out[key] * (mx - mn) + mn
            else:
                raise ValueError(mode)

            out[key] = out[key].to(torch.float32)
        return out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)