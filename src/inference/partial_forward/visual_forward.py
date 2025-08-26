import numpy as np

from pathlib import Path
from argparse import Namespace
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoConfig
from transformers.image_utils import load_image

from src.session.session import setup_vision_session

repo_id = "ainekko/smolVLM"
model_filename = "vision_encoder_fixed.onnx"

if __name__ == "__main__":
    local_model_path = hf_hub_download(
        repo_id=repo_id, filename=model_filename)
    args = Namespace(
        model_path=local_model_path,
        num_layers=12,
        verbose=True,
        enable_tracing=False,
        run_dir="myrundir",
        use_kvc=False,
        window_size=1
        )
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
    ]

    image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="np")
    input_ids = inputs['input_ids']

    vision_session = setup_vision_session(Path(local_model_path), args)

    image_features = vision_session.run(
        ['image_features'],  # List of output names or indices
        {
            'pixel_values': inputs['pixel_values'],
            'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
        }
    )[0]

    print(image_features)