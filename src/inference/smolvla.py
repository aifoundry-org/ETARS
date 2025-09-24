
import torch

from lerobot.constants import OBS_IMAGE, OBS_STATE
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from src.lerobot.policies.smolvla.modeling_smolvla_ort import SmolVLAPolicyOnnx

REPO_ID = "lerobot/svla_so101_pickplace" 

policy = SmolVLAPolicyOnnx()

ds = LeRobotDataset(REPO_ID, download_videos=True)

# OBS_IMAGE = "observation.images.up"

N = 4
idx0 = 0
# device = next(policy.parameters()).device
task_str = "Pick and place the object into the bin" 

# policy.config.image_features[OBS_IMAGE] = 0

for i in range(idx0, idx0 + N):
    sample = ds[i]  # returns dict with observation and (optionally) action
    # images = sample[OBS_IMAGE]           # dict: {"cam_*": Tensor(3,H,W)} or batched
    images = sample["observation.images.up"]           # dict: {"cam_*": Tensor(3,H,W)} or batched
    images = images.unsqueeze(0)
    # Ensure batch dimension and device for each camera
    # images = {k: v.unsqueeze(0) if v.ndim==3 else v.to(dtype=torch.float32)
    #           for k, v in images.items()}

    # batch = {OBS_IMAGE: images, "task": task_str}
    batch = {OBS_IMAGE: images, "task": task_str}
    if OBS_STATE in sample and isinstance(sample[OBS_STATE], torch.Tensor):
        batch[OBS_STATE] = sample[OBS_STATE].unsqueeze(0)


    with torch.inference_mode():
        out = policy.select_action(batch)
    act = out["action"] if isinstance(out, dict) and "action" in out else out
    print(f"frame {i}: action shape {tuple(act.shape)}")