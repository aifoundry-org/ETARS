from pathlib import Path
from src.session import vlm_exp
from src.session.nn_session import OnnxModule

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

class SmolVLMWithExpertModelOnnx(OnnxModule):
    def __init__(self):
        self.model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # self.vlm_session = self.get_vlme_session(vlme_model_path)

        # Outputs are (vlm_emb, exp_emb, kv_cache...)

    def get_vlme_module(self, model_path_prefill: Path, model_path_decode: Path):
        self.vlm_session_prefill = vlm_exp.setup_vlme_session(model_path_prefill, "CPU")
        self.vlm_session_decode = vlm_exp.setup_vlme_session(model_path_decode, "CPU")
        # self.num_kv_outputs = len(self.output_names) - 2

        return self

    def get_text_encoder_module(self, model_path: Path) -> OnnxModule:
        module = OnnxModule(model_path=model_path,
                            provider="CPU",
                            fix_dimensions=False,
                            input_names=[],
                            output_names=[])
        # Place to fill onnx_symbols
        module.onnx_symbols = {}

        return module

    def get_visual_module(self, model_path: Path):
        module = OnnxModule(model_path=model_path,
                            provider="CPU",
                            fix_dimensions=False,
                            input_names=[],
                            output_names=[])
        # Place to fill onnx_symbols
        module.onnx_symbols = {}
        module.num_layers = 12

        return module

    # TODO implement past_key_values model switching logic
    def forward(self,
                vlm_embeds,
                expert_embeds,
                attention_mask,
                position_ids,
                fill_kv_cache=True,  # Don't know what to do with that yet
                past_key_values=None
                ):

        input_feed = {
            'vlm_embeds': vlm_embeds,
            'expert_embeds': expert_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }

        if past_key_values is not None:
            # Check if the number of provided KV tensors matches what the model expects
            # Subtract the 4 main inputs
            num_expected_kv_inputs = len(self.input_names) - 4
            if len(past_key_values) != num_expected_kv_inputs:
                raise ValueError(
                    f"Incorrect number of past_key_values provided. "
                    f"Expected {num_expected_kv_inputs}, but got {len(past_key_values)}."
                )

            for i in range(0, len(past_key_values), 2):
                layer_idx = i // 2
                input_feed[f'past_key_{layer_idx}'] = past_key_values[i]
                input_feed[f'past_value_{layer_idx}'] = past_key_values[i+1]
            
            outputs_embeds, past_key_values = self.vlm_session_decode.run(self.output_names,
                                                                          input_feed=input_feed)
        else:
            outputs_embeds, past_key_values = self.vlm_session_prefill.run(self.output_names,
                                                                   input_feed=input_feed)

        return outputs_embeds, past_key_values
