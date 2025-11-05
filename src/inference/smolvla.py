import argparse
import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# from src.lerobot.policies.onnx.smolvla.modeling_smolvla_ort import SmolVLAPolicyOnnx
from src.lerobot.policies.tinygrad.modeling_smolvla import SmolVLAPolicyTinyOnnx
from src.utils.deterministrc import set_deterministic


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLA policy once on a sample.")
    parser.add_argument(
        "--device",
        choices=["CPU", "ET"],
        default="CPU",
        help='Execution device for policy backend (default: "CPU")',
    )
    args = parser.parse_args()

    set_deterministic(0)

    config = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")

    ds_name = "aifoundry-org/libero"
    ds = LeRobotDataset(ds_name, episodes=[0])
    ds_meta = LeRobotDatasetMetadata(ds_name)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=ds_meta.stats)
    noise = np.ones((1, 50, 32), dtype=np.float32)

    config.device = args.device  # "CPU" or "ET"
    # policy = SmolVLAPolicyOnnx(config=config, ds_meta=ds_meta.stats)
    policy = SmolVLAPolicyTinyOnnx(config=config, ds_meta=ds_meta.stats)

    batch = preprocessor(ds[0])

    out = policy.select_action(batch, noise=noise)
    print("--------------------------------------------------------------")
    print(f"Taken action is {out}")


if __name__ == "__main__":
    main()
