import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.factory import make_policy

from src.utils.deterministrc import set_deterministic

set_deterministic(0)


config = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
config.pretrained_path = "HuggingFaceVLA/smolvla_libero"
ds_meta = LeRobotDatasetMetadata("aifoundry-org/libero")
ds = LeRobotDataset("aifoundry-org/libero", episodes=[0])
policy = make_policy(cfg=config, ds_meta=ds_meta).eval()
preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=ds_meta.stats)


batch = preprocessor(ds[0])

noise = torch.ones(1, 50, 32)

with torch.inference_mode():
    policy.model.to(torch.float32)
    out = policy.select_action(batch, noise=noise)

print("--------------------------------------------------------------")
print(f"Taken action is {out}")