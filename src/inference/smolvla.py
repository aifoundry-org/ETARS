import argparse

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

from src.lerobot.policies.smolvla.modeling_smolvla_ort import SmolVLAPolicyOnnx
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

    config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
    config.device = args.device  # "CPU" or "ET"

    ds_name = "lerobot/svla_so101_pickplace"
    ds = LeRobotDataset(ds_name)
    ds_meta = LeRobotDatasetMetadata(ds_name)

    policy = SmolVLAPolicyOnnx(config=config, ds_meta=ds_meta.stats)

    sample = ds[0]
    batch = {
        "observation.images.camera1": sample["observation.images.up"].unsqueeze(0),
        "observation.images.camera2": sample["observation.images.side"].unsqueeze(0),
        "observation.state": sample["observation.state"].unsqueeze(0),
        "task": ["Pick and place the object"],
    }

    out = policy.select_action(batch)
    print("--------------------------------------------------------------")
    print(f"Taken action is {out}")


if __name__ == "__main__":
    main()
