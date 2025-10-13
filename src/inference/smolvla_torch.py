import torch
import random
import numpy as np
import os

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.constants import OBS_IMAGE, OBS_STATE
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from src.lerobot.policies.smolvla.modeling_smolvla_ort import SmolVLAPolicyOnnx
from lerobot.policies.factory import make_policy

REPO_ID = "lerobot/svla_so101_pickplace" 


N = 4
idx0 = 0
task_str = "Pick and place the object into the bin" 

config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
ds = LeRobotDataset("lerobot/svla_so101_pickplace")
ds_meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")
config.pretrained_path = "lerobot/smolvla_base"
policy = make_policy(cfg=config, ds_meta=ds_meta).eval()

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
    "observation.images.up":   sample["observation.images.up"].unsqueeze(0),   # [B,C,H,W]
    "observation.images.side": sample["observation.images.side"].unsqueeze(0), # [B,C,H,W]
    "observation.state":       sample["observation.state"].unsqueeze(0),       # [B,6]
    "task": ["Pick and place the object"],  # dataset is single-task; any reasonable string works
}



with torch.inference_mode():
    policy.model.to(torch.float32)
    out = policy.select_action(batch)

print(out)