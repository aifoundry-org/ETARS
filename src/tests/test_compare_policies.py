import torch
import pytest
import random
import numpy as np
import os


from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.lerobot.policies.smolvla.modeling_smolvla_ort import SmolVLAPolicyOnnx

from lerobot.policies.factory import make_policy

def set_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

set_deterministic(0)

@pytest.fixture(scope="session")
def dataset():
    return LeRobotDataset("lerobot/svla_so101_pickplace")

@pytest.fixture(scope="session")
def policies():
    config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")  # ensures config present
    config.pretrained_path = "lerobot/smolvla_base"
    meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")

    # Initialize default torch smolVLA policy from lerobot
    pA = make_policy(cfg=config, ds_meta=meta).eval()
    pA.model.to(torch.float32)

    # Initialize drop-in replacement using onnx inference
    pB = SmolVLAPolicyOnnx(ds_meta=meta.stats)

    return pA, pB

def _batch_from_sample(sample):
    return {
        "observation.images.up":   sample["observation.images.up"].unsqueeze(0),   # [B,C,H,W]
        "observation.images.side": sample["observation.images.side"].unsqueeze(0), # [B,C,H,W]
        "observation.images.camera1": sample["observation.images.up"].unsqueeze(0),    # [B,C,H,W]
        "observation.images.camera2": sample["observation.images.side"].unsqueeze(0),  # [B,C,H,W]
        "observation.state": sample["observation.state"].unsqueeze(0),                 # [B,6]
        "task": ["Pick and place the object"],
    }


def _collect_actions(policy, dataset, idxs, is_onnx=False):
    set_deterministic(0)
    acts = []
    with torch.inference_mode():
        for i in idxs:
            if is_onnx:
                batch = _batch_from_sample(dataset[i])
                print(dataset[i]["observation.images.up"].mean())
                out = policy.select_action(batch)
            else:
                batch = _batch_from_sample(dataset[i])
                print(dataset[i]["observation.images.up"].mean())
                out = policy.select_action(batch)
            # select_action usually returns a tensor; if dict, adapt:
            if isinstance(out, dict) and "action" in out:
                out = out["action"]
            acts.append(torch.as_tensor(out))
    return acts

@pytest.mark.parametrize("steps,atol,rtol", [(5, 1e-3, 0.0)])
def test_actions_close_listwise(dataset, policies, steps, atol, rtol):
    set_deterministic(0)
    pA, pB = policies
    # deterministic, shared indices
    idxs = [i for i in range(steps)]
    A = _collect_actions(pA, dataset, idxs)
    B = _collect_actions(pB, dataset, idxs)

    assert len(A) == len(B) == steps
    for t, (a, b) in enumerate(zip(A, B)):
        assert a.shape == b.shape, f"[t={t}] shape mismatch: {a.shape} vs {b.shape}"
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            diff = (a - b).abs().max().item()
            print(diff)
            raise AssertionError(f"[t={t}] max |Δ| = {diff:.6g} ≥ atol={atol}")