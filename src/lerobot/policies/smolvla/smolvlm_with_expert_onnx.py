from pathlib import Path
from src.session import vlm_exp
from src.session.nn_session import OnnxModule
from typing import Optional

from transformers import AutoProcessor


class SmolVLMWithExpertModelOnnx(OnnxModule):
    def __init__(self, hf_repo: Optional[str] = None):
        self.model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        self.hf_repo = hf_repo

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Outputs are (vlm_emb, exp_emb, kv_cache...)
        # super.__init__()

    def get_vlme_module(
        self,
        model_path_prefill: Optional[Path] = None,
        model_path_decode: Optional[Path] = None,
        provider="CPU",
    ):
        self.vlm_module_prefill = OnnxModule(
            model_path=model_path_prefill,
            hf_repo=self.hf_repo,
            hf_filename="smolvlm_expert_prefill.onnx",
            rundir="prefill_session",
            provider=provider,
        )
        self.vlm_module_decode = OnnxModule(
            model_path=model_path_decode,
            hf_repo=self.hf_repo,
            hf_filename="smolvlm_expert_decode.onnx",
            rundir="decode_session",
            provider=provider,
        )
        return self

    def get_text_encoder_module(
            self, model_path: Optional[Path] = None, provider="CPU") -> OnnxModule:
        module = OnnxModule(
            model_path=model_path,
            hf_repo=self.hf_repo,
            hf_filename="smolvlm_text.onnx",
            provider=provider,
            fix_dimensions=False,
            input_names=[],
            output_names=[],
            rundir="text_encoder",
        )
        # Place to fill onnx_symbols

        return module

    def get_visual_module(self, model_path: Optional[Path] = None, provider="CPU") -> OnnxModule:
        module = OnnxModule(
            model_path=model_path,
            hf_repo=self.hf_repo,
            hf_filename="smolvlm_vision.onnx",
            provider=provider,
            fix_dimensions=False,
            input_names=[],
            output_names=[],
        )
        # Place to fill onnx_symbols
        module.onnx_symbols = {}
        module.num_layers = 12

        return module

    def forward(
        self,
        vlm_embeds,
        expert_embeds,
        attention_mask,
        position_ids,
        fill_kv_cache=True,  # Don't know what to do with that yet
        past_key_values=None,
    ):

        input_feed = {
            "vlm_embeds": vlm_embeds,
            "expert_embeds": expert_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        if past_key_values is not None:
            # Check if the number of provided KV tensors matches what the model expects
            # Subtract the 4 main inputs (in case of onnx module it's 3)
            decode_input_names = self.vlm_module_decode.input_names
            num_expected_kv_inputs = len(decode_input_names) - 3
            if len(past_key_values) != num_expected_kv_inputs:
                raise ValueError(
                    f"Incorrect number of past_key_values provided. "
                    f"Expected {num_expected_kv_inputs}, but got {len(past_key_values)}."
                )

            for i in range(0, len(past_key_values), 2):
                layer_idx = i // 2
                input_feed[f"past_key_{layer_idx}"] = past_key_values[i]
                input_feed[f"past_value_{layer_idx}"] = past_key_values[i + 1]

            input_feed.pop(
                "vlm_embeds"
            )  # because onnx exporter prunes node that have None as input

            outputs_embeds, * \
                past_key_values = self.vlm_module_decode(**input_feed)
        else:

            outputs_embeds, * \
                past_key_values = self.vlm_module_prefill(**input_feed)

        return outputs_embeds, past_key_values
