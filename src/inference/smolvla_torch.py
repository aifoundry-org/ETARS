import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy

from src.utils.deterministrc import set_deterministic

set_deterministic(0)

config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
ds = LeRobotDataset("lerobot/svla_so101_pickplace")
ds_meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")
config.pretrained_path = "lerobot/smolvla_base"
policy = make_policy(cfg=config, ds_meta=ds_meta).eval()

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

print("--------------------------------------------------------------")
print(f"Taken action is {out}")