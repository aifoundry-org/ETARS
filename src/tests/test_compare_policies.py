import torch
import pytest
import random
import numpy as np
import os


from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.lerobot.policies.smolvla.modeling_smolvla_ort import SmolVLAPolicyOnnx

from lerobot.policies.factory import make_pre_post_processors
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
    return LeRobotDataset("aifoundry-org/libero", episodes=[0, 1, 2, 3, 4, 5])

@pytest.fixture(scope="session")
def policies():
    config = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
    config.pretrained_path = "HuggingFaceVLA/smolvla_libero"
    meta = LeRobotDatasetMetadata("aifoundry-org/libero")

    preprocessor, _ = make_pre_post_processors(config, dataset_stats=meta.stats)

    # Initialize default torch smolVLA policy from lerobot
    policy_torch = make_policy(cfg=config, ds_meta=meta).eval()
    policy_torch.model.to(torch.float32)

    # Initialize drop-in replacement using onnx inference
    policy_onnx = SmolVLAPolicyOnnx(config=config, ds_meta=meta.stats)

    return policy_torch, policy_onnx, preprocessor

def _batch_from_sample(sample, preprocessor):
    return preprocessor(sample)


def _collect_actions(policy, dataset, preprocessor, idxs):
    set_deterministic(0)
    acts = []
    with torch.inference_mode():
        for i in idxs:
            batch = _batch_from_sample(dataset[i], preprocessor)
            out = policy.select_action(batch)
            # select_action usually returns a tensor; if dict, adapt:
            if isinstance(out, dict) and "action" in out:
                out = out["action"]
            acts.append(torch.as_tensor(out))
    return acts

@pytest.mark.parametrize("steps,atol,rtol", [(5, 1e-5, 0.0)])
def test_actions_close_listwise(dataset, policies, steps, atol, rtol):
    set_deterministic(0)
    policy_torch, policy_onnx, preprocessor = policies
    # deterministic, shared indices
    idxs = [i for i in range(steps)]
    A = _collect_actions(policy_torch, dataset, preprocessor, idxs)
    B = _collect_actions(policy_onnx, dataset, preprocessor, idxs)

    assert len(A) == len(B) == steps
    for t, (a, b) in enumerate(zip(A, B)):
        assert a.shape == b.shape, f"[t={t}] shape mismatch: {a.shape} vs {b.shape}"
        print("Policy TORCH action: ", a)
        print("Policy ONNX  action: ", b)
        print("\n")
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            diff = (a - b).abs().max().item()
            print(diff)
            raise AssertionError(f"[t={t}] max |Δ| = {diff:.6g} ≥ atol={atol}")