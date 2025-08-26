from pathlib import Path
from argparse import Namespace
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
from onnxruntime.transformers.io_binding_helper import IOBindingHelper, TypeHelper
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from transformers.image_utils import load_image

from src.session.session import setup_encoder_session

repo_id = "ainekko/smolVLM"
model_filename = "embed_tokens.onnx"

if __name__ == "__main__":
    local_model_path = hf_hub_download(
        repo_id=repo_id, filename=model_filename)
    args = Namespace(
        model_path=Path(local_model_path),
        num_layers=0,
        verbose=False,
        enable_tracing=False,
        run_dir="myrundir",
        use_kvc=False,
        window_size=1
        )
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    embed_session = setup_encoder_session(Path(local_model_path), args)

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

    inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

    print(inputs_embeds)