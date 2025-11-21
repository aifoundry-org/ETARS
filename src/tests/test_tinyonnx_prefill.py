import numpy as np

from src.lerobot.policies.tinygrad.smolvlm_with_expert import SmolVLMWithExpertModelTinyOnnx
from src.session.tinygrad.tiny_onnx import TinyOnnxModule


if __name__ == "__main__":
    vlme = SmolVLMWithExpertModelTinyOnnx(hf_repo="ainekko/smolvla_libero_onnx").get_vlme_module()
    dummy_data = {"vlm_embeds": np.random.randn(1, 151, 960).astype(np.float32),
                  "expert_embeds": None,
                  "attention_mask": np.ones((1, 151, 151)).astype(bool),
                #   "attention_mask": np.ones((1, 151, 151)),
                  "position_ids": np.expand_dims(np.array((range(0,151))), 0)
                  }
    
    _, past_key_values = vlme.forward(
        vlm_embeds=dummy_data["vlm_embeds"],
        expert_embeds=dummy_data["expert_embeds"],
        attention_mask=dummy_data["attention_mask"],
        position_ids=dummy_data["position_ids"],
        fill_kv_cache=True
    )

    print(past_key_values)