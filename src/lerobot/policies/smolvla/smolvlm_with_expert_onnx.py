from src.session import vision, text_encoder, head, vlm_exp, state

class SmolVLMWithExpertModelOnnx:
    def __init__(self, 
                 vlme_model_path="") -> None:
        self.vlm_session = self.get_vlme_session(vlme_model_path)

        self.input_names = [inp.name for inp in self.vlm_session.get_inputs()]
        self.output_names = [out.name for out in self.vlm_session.get_outputs()]
        self.num_kv_outputs = len(self.output_names) - 2 # Outputs are (vlm_emb, exp_emb, kv_cache...)
        
        pass

    def get_vlme_session(self, model_path):
        return vlm_exp.setup_vlme_session(model_path, "CPU")

    def get_encoder_session(self):
        pass

    def get_visual_session(self):
        pass

    def embed_image(self, image):
        # return image_hidden_state
        pass

    def embed_language_tokens(self, tokens):
        # return input_embeddings
        pass
    
    # TODO implement past_key_values model switching logic
    def forward(self, 
                vlm_embeds, 
                expert_embeds, 
                attention_mask, 
                position_ids, 
                fill_kv_cache=True, # Don't know what to do with that yet
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
            num_expected_kv_inputs = len(self.input_names) - 4 # Subtract the 4 main inputs
            if len(past_key_values) != num_expected_kv_inputs:
                raise ValueError(
                    f"Incorrect number of past_key_values provided. "
                    f"Expected {num_expected_kv_inputs}, but got {len(past_key_values)}."
                )

            for i in range(0, len(past_key_values), 2):
                layer_idx = i // 2
                input_feed[f'past_key_{layer_idx}'] = past_key_values[i]
                input_feed[f'past_value_{layer_idx}'] = past_key_values[i+1]

        outputs_embeds, past_key_values = self.vlm_session.run(self.output_names, 
                                                               input_feed=input_feed)

        return outputs_embeds, past_key_values