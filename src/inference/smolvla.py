import torch
import random
import numpy as np
import os

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.constants import OBS_IMAGE, OBS_STATE
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from src.lerobot.policies.smolvla.modeling_smolvla_ort import SmolVLAPolicyOnnx

REPO_ID = "lerobot/svla_so101_pickplace" 


N = 4
idx0 = 0
task_str = "Pick and place the object into the bin" 

# policy.config.image_features[OBS_IMAGE] = 0
# policy.config.chunk_size = 6 # Temp just for testing


config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
ds = LeRobotDataset("lerobot/svla_so101_pickplace")
ds_meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")
policy = SmolVLAPolicyOnnx(ds_meta=ds_meta.stats)

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
    torch.set_num_threads(1)  # reduce CPU reduction-order noise
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

set_deterministic(0)


ds = LeRobotDataset("lerobot/svla_so101_pickplace")

sample = ds[0]

batch = {
    "observation.image":   sample["observation.images.up"].unsqueeze(0),   # [B,C,H,W]
    # "observation.image2": sample["observation.images.side"].unsqueeze(0), # [B,C,H,W]
    "observation.state":       sample["observation.state"].unsqueeze(0),       # [B,6]
    "task": ["Pick and place the object"],  # dataset is single-task; any reasonable string works
}



with torch.inference_mode():
    out = policy.select_action(batch)

print(out)
# tensor([[108.8815, -70.4616, 114.5464,  94.5968, -20.5535,   1.2494]])
# tensor([[ 2.5643, -0.1751,  2.5281,  2.2133,  1.8083, -1.4187]])
# act = out["action"] if isinstance(out, dict) and "action" in out else out
# print(f"frame {i}: action shape {tuple(act.shape)}")