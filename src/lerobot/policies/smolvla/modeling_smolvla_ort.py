import math
import numpy as np
import onnxruntime as ort


from src.session import vision, text_encoder, head, vlm_exp, state
from src.lerobot.policies.smolvla.smolvlm_with_expert_onnx import SmolVLMWithExpertModelOnnx
from src.utils.np_operations import silu


try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None 


def make_att_2d_masks(pad_masks: np.ndarray, att_masks: np.ndarray) -> np.ndarray:
    """Creates 2D attention masks from 1D padding and attention masks."""
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks.ndim is {att_masks.ndim}, expected 2")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks.ndim is {pad_masks.ndim}, expected 2")

    # Compute cumulative sum along the sequence dimension
    cumsum = np.cumsum(att_masks, axis=1)

    # Create a 2D mask where attention is allowed from position i to j if j >= i
    # This is done by comparing the cumulative sums using broadcasting
    att_2d_masks = cumsum[:, np.newaxis, :] <= cumsum[:, :, np.newaxis]

    # Create a 2D padding mask from the 1D padding mask
    pad_2d_masks = pad_masks[:, np.newaxis, :] & pad_masks[:, :, np.newaxis]

    # Combine the attention and padding masks
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

# numpy implementation of original, obvs slower then torch's
# TODO probably rewrite in onnx
def create_sinusoidal_pos_embedding(
    time: np.ndarray, dimension: int, min_period: float, max_period: float
) -> np.ndarray:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = np.float64
    fraction = np.linspace(0.0, 1.0, dimension // 2, dtype=dtype)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = np.concatenate([np.sin(sin_input), np.cos(sin_input)], axis=1)
    return pos_emb


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a numpy array along the sequence dimension to match max_len.

    Args:
        tensor (np.ndarray): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        np.ndarray: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create the shape for the new padded array
    padded_shape = (b, max_len, *tensor.shape[2:])

    # Create a padded array of max_len filled with the pad_value
    padded_tensor = np.full(padded_shape, pad_value, dtype=tensor.dtype)
    
    # Copy the existing values into the padded array
    padded_tensor[:, :d] = tensor

    return padded_tensor


# [B*C,3,H,W] ──► VISION sess ──reshape──► [B, C*Simg, Dvlm]
# [B,Stxt]    ──► TEXT   sess ────────────► [B, Stxt,  Dvlm]
# [B,…] (opt) ──► STATE  sess ────────────► [B, Sstate,Dvlm]
#                           concat tokens ─► emb0: [B, S0, Dvlm]
# action queries ──────────────────────────► emb1: [B, S1, de_in]
# build: mask [B,S,S] (bool), pos [B,S] (int64), where S=S0+S1
# (mask,pos,emb0,emb1) ──► CORE sess ──► expert_hidden [B,S1,Dexp]
# expert_hidden ──(opt)► HEAD sess ──► actions [B,S1,act_dim]

class smolVLAFlow():
    def __init__(self, 
                 vision_path, 
                 text_path, 
                 core_path, 
                 vlme_path, 
                 state_path=None, 
                 head_path=None) -> None:

        self.vision = vision.setup_vision_session(vision_path,
                                                   "CPU",
                                                   False,
                                                   num_layers=12)
        self.text = text_encoder.setup_text_encoder_session(text_path, "ET")
        self.state = state.get_state_session()
        self.head = head.setup_head_session()

        self.vlme = SmolVLMWithExpertModelOnnx(vlme_model_path=vlme_path)
        
        self.state_proj = None
        self.action_in_proj = None
        self.action_out_proj = None
        self.action_time_mlp_in = None
        self.action_time_mpl_out = None

        # Configurables
        self.add_image_special_tokens = False 
        self.prefix_length = 10
        self.min_period = 0
        self.max_period = 0
        self.chunk_size = 1
        self.max_action_dim = 10
        self.num_steps = 10

        # Properties
        self.expert_hidden_size = 256 #?

    @staticmethod
    def sample_noise(shape):
        noise = np.random.normal(
            loc=0.0,  
            scale=1.0,  
            size=shape,
        ).astype(np.float32) 
        return noise
    
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        embs = []
        pad_masks = []
        att_masks = []

        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (self.text.run()) # placaholder for img_Start_token gen
            

            img_emb = self.vision.run() # placeholder for vison embedding generation

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb  * img_emb_dim ** 0.5

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)

            if self.add_image_special_tokens:
                image_end_token = self.text.run()

                image_end_mask = np.ones_like(image_end_token[:, :, 0])
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        
        lang_emb = self.text.run()

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * np.sqrt(lang_emb_dim) # prev math.sqrt (why?)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj.run()
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]

        states_seq_len = state_emb.shape[1]
        state_mask = np.ones((bsize, states_seq_len), dtype=bool) # ?
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = np.concatenate(embs, axis=1)
        pad_masks = np.concatenate(pad_masks, axis=1)
        att_masks = np.array(att_masks, dtype=bool)
        att_masks = att_masks[None, :] # ?

        seq_len = pad_masks.shape[1]

        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
        
        # att_masks = att_masks.expand_dims(bsize, -1) #?
        att_masks = np.broadcast_to(att_masks, (bsize, -1))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestamp):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj.run(noisy_actions) # placeholder

        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(timestamp, 
                                                   self.expert_hidden_size,
                                                   self.min_period,
                                                   self.max_period)
        
        # time_emb = time_emb.type(dtype=dtype) #?

        time_emb = np.broadcast_to(time_emb[:, None, :], action_emb.shape).copy()
        action_time_emb = np.concatenate([action_emb, time_emb], axis=2)

        action_time_emb = self.action_time_mlp_in.run(action_time_emb) # placeholder
        action_time_emb = silu(action_time_emb) # TODO chech if swish is in supported onnx's opset
        action_time_emb = self.action_time_mpl_out.run(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = np.ones((bsize, action_time_dim), dtype=bool)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.chunk_size
        embs = np.concatenate(embs, axis=1)
        pad_masks = np.concatenate(pad_masks, axis=1)
        att_masks = np.array(att_masks)
        att_masks = np.broadcast_to(att_masks[None, :], (bsize, len(att_masks)))

        return embs, pad_masks, att_masks
        
    
    def forward(self):
        pass

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]

        if noise is None:
            actions_shape = (bsize, self.chunk_size, self.max_action_dim)
            noise = self.sample_noise(actions_shape)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        

        prefix_add_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = np.cumsum(prefix_pad_masks, axis=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.vlme.forward(vlm_embeds=prefix_embs, 
                                               expert_embeds=None, 
                                               attention_mask=prefix_add_2d_masks,
                                               position_ids=prefix_position_ids,
                                               fill_kv_cache=True)

        dt = -1.0 / self.num_steps
        dt = np.array(dt, dtype=np.float32)

        x_t = noise
        time = np.array(1.0, dtype=np.float32)

        while time >= -dt / 2:
            expanded_time = np.broadcast_to(time, bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestamp):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestamp)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = np.broadcast_to(prefix_pad_masks[:, np.newaxis, :], (batch_size, suffix_len, prefix_len)).copy()
        
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = np.concatenate([prefix_pad_2d_masks, suffix_att_2d_masks], axis=2)
        prefix_offsets = np.sum(prefix_pad_masks, axis=-1)[:, None]
        position_ids = prefix_offsets + np.cumsum(suffix_pad_masks, axis=1) - 1

        # output_embeds, _ = magical_vlm_forward_function() #
        output_embeds, _ = self.vlme.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            vlm_embeds=None,
            expert_embeds=suffix_embs

        )

        suffix_out = output_embeds[1]
        suffix_out = suffix_out[:, -self.chunk_size :]
        suffix_out = suffix_out.astype(dtype=np.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t


