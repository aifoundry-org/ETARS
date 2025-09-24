from typing import Any


# Fake implementation of Normalization and Unnormalization (for now)

class Normalize:
    def __init__(self, features, norm_map, stats=None) -> None:
        pass

    def forward(self, batch):
        return batch
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

class Unnormalize:
    def __init__(self, features, norm_map, stats=None) -> None:
        pass

    def forward(self, batch):
        return batch

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)