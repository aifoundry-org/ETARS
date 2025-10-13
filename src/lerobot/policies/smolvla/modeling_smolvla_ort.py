import math
import numpy as np
import onnxruntime as ort
import torch

from pathlib import Path

from src.lerobot.policies.smolvla.smolvlm_with_expert_onnx import (
    SmolVLMWithExpertModelOnnx,
)
from src.utils.np_operations import silu, mse_loss

from src.session.nn_session import OnnxModule

from transformers import AutoProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig

from src.lerobot.policies.normalize import (
    Normalize,
    Unnormalize
)

from src.lerobot.policies.smolvla import constants


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


class SmolVLAPolicyOnnx(SmolVLAPolicy):

    def __init__(self, ds_meta=None):
        self.config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
        name = "smolvla"
        # self.config = SmolVLAConfig(input_features=constants.INPUT_FEATURES, output_features=constants.OUTPUT_FEATURES)
        config = self.config
        dataset_stats = ds_meta

        # Normalize is Fake
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(
            self.config.vlm_model_name
        ).tokenizer

        self.model = SmolVLAFlowOnnx(self.config)

        self.reset()

    def eval(self):
        return self

    def forward(self, batch, noise=None, time=None):
        raise NotImplementedError(
            "No training is supported, forward are not implemented"
        )


# [B*C,3,H,W] ──► VISION sess ──reshape──► [B, C*Simg, Dvlm]
# [B,Stxt]    ──► TEXT   sess ────────────► [B, Stxt,  Dvlm]
# [B,…] (opt) ──► STATE  sess ────────────► [B, Sstate,Dvlm]
#                           concat tokens ─► emb0: [B, S0, Dvlm]
# action queries ──────────────────────────► emb1: [B, S1, de_in]
# build: mask [B,S,S] (bool), pos [B,S] (int64), where S=S0+S1
# (mask,pos,emb0,emb1) ──► CORE sess ──► expert_hidden [B,S1,Dexp]
# expert_hidden ──(opt)► HEAD sess ──► actions [B,S1,act_dim]


class SmolVLAFlowOnnx:
    def __init__(self, config: SmolVLAConfig):

        self.config = config

        self.vlme = SmolVLMWithExpertModelOnnx(hf_repo="ainekko/smolvla_base_onnx")
        self.vlme_module = self.vlme.get_vlme_module(
            provider="CPU"
        )
        self.text = self.vlme.get_text_encoder_module(
            provider="CPU"
        )
        self.vision_module = self.vlme.get_visual_module(
            provider="CPU"
        )


        # self.state_proj = OnnxModule(
            # Path("/workspaces/hf_inference/models/smolvla_onnx/state_projector.onnx"),
            # "CPU",
        # )
        self.state_proj = OnnxModule(
            provider="CPU",
            hf_repo="ainekko/smolvla_base_onnx", hf_filename="state_projector.onnx"
        )

        self.action_in_proj = OnnxModule(
            provider="CPU",
            hf_repo="ainekko/smolvla_base_onnx", hf_filename="action_in_projector.onnx"
        )
        self.action_out_proj = OnnxModule(
            provider="CPU",
            hf_repo="ainekko/smolvla_base_onnx", hf_filename="action_out_projector.onnx"
        )
        self.action_time_mlp_in = OnnxModule(
            provider="CPU",
            hf_repo="ainekko/smolvla_base_onnx", hf_filename="time_in_projector.onnx"
        )
        self.action_time_mpl_out = OnnxModule(
            provider="CPU",
            hf_repo="ainekko/smolvla_base_onnx", hf_filename="time_out_projector.onnx"
        )


        # class instance vars
        self.fake_image_token = self.vlme.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlme.processor.tokenizer.global_image_token_id
        self.global_image_start_token = np.array(
            [self.fake_image_token, self.global_image_token]
        )
        self.image_end_token = np.array([self.fake_image_token])

        # Configurables
        self.add_image_special_tokens = False
        self.prefix_length = self.config.prefix_length

        # Properties
        self.expert_hidden_size = 720

    @staticmethod
    def sample_noise(shape):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32
        ).numpy()
        # noise = np.random.normal(
            # loc=0.0,
            # scale=1.0,
            # size=shape,
        # ).astype(np.float32)
        return noise

    def sample_time(self, bsize):
        # We sample from a Beta distribution with alpha=1.5 and beta=1.0.
        time_beta = np.random.beta(1.5, 1.0, size=bsize).astype(np.float32)

        # Scale and shift the values to be in the range [0.001, 1.0]
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        embs = []
        pad_masks = []
        att_masks = []

        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:

                # placaholder for img_start_token gen
                image_start_token = self.text(
                    **{self.text.input_names[0]: self.global_image_start_token}
                )

                image_start_mask = np.ones_like(image_start_token[:, :, 0])
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            # img = (img.numpy() * 255).astype(np.uint8)
            img = img.numpy()
            img_emb = self.vision_module(**{self.vision_module.input_names[0]: img})[0]

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * img_emb_dim**0.5

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)

            if self.add_image_special_tokens:
                image_end_token = self.text(
                    **{self.text.input_names[0]: self.image_end_token}
                )

                image_end_mask = np.ones_like(image_end_token[:, :, 0])
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        lang_tokens = lang_tokens.numpy()
        lang_emb = self.text(**{self.text.input_names[0]: lang_tokens})[0]

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # Interestingly enough state from dataset may be slightly different from time to time
        # Need to make it more deterministic
        state = state.numpy()  # TODO mode into onnx_module
        state_emb = self.state_proj(**{self.state_proj.input_names[0]: state})[0]
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]

        states_seq_len = state_emb.shape[1]
        state_mask = np.ones((bsize, states_seq_len), dtype=bool)  # ?
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = np.concatenate(embs, axis=1)
        pad_masks = np.concatenate(pad_masks, axis=1)
        att_masks = np.array(att_masks, dtype=bool)
        att_masks = att_masks[None, :]  # ?

        seq_len = pad_masks.shape[1]

        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        # att_masks = att_masks.expand_dims(bsize, -1) #?
        # att_masks = np.expand_dims(att_masks, axis=(0, -1))
        # att_masks = np.broadcast_to(att_masks, (bsize, -1))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestamp):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # noisy_actions = np.ones_like(noisy_actions)
        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(
            **{self.action_in_proj.input_names[0]: noisy_actions}
        )[
            0
        ]  # noisy actions (1, 50, 32)



        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestamp,
            self.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
        )

        # time_emb = time_emb.type(dtype=dtype) #?

        time_emb = np.broadcast_to(
            time_emb[:, None, :], action_emb.shape
        ).copy()  # action_emb.shape torch.Size([1, 50, 720]), time_emb.shape torch.Size([1, 720])
        action_time_emb = np.concatenate([action_emb, time_emb], axis=2).astype(
            np.float32
        )

        action_time_emb = self.action_time_mlp_in(
            **{self.action_time_mlp_in.input_names[0]: action_time_emb}
        )[
            0
        ]  # action_time_emb.shape torch.Size([1, 50, 1440])
        
        # TODO chech if swish is in supported onnx's opset
        action_time_emb = silu(action_time_emb)
        action_time_emb = self.action_time_mpl_out(
            **{self.action_time_mpl_out.input_names[0]: action_time_emb}
        )[
            0
        ]  # action_time_emb.shape torch.Size([1, 50, 720])

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = np.ones((bsize, action_time_dim), dtype=bool)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = np.concatenate(embs, axis=1)
        pad_masks = np.concatenate(pad_masks, axis=1)
        att_masks = np.array(att_masks)
        att_masks = np.broadcast_to(att_masks[None, :], (bsize, len(att_masks)))

        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ):
        raise NotImplementedError("Forward not yet implemented")
        if noise is None:
            noise = self.sample_noise(actions.shape)

        if time is None:
            time = self.sample_time(actions.shape[0])

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = np.concatenate([prefix_pad_masks, suffix_pad_masks], axis=1)
        att_masks = np.concatenate([prefix_att_masks, suffix_att_masks], axis=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = np.cumsum(pad_masks, axis=1) - 1

        (_, suffix_out), _ = self.vlme.forward(
            vlm_embeds=prefix_embs,
            expert_embeds=suffix_embs,
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            fill_kv_cache=False,
            past_key_values=None,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        # suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(**{self.action_out_proj.output_names[0]: suffix_out})
        losses = mse_loss(u_t, v_t, reduction="none")

        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ):
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        prefix_add_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = np.cumsum(prefix_pad_masks, axis=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.vlme.forward(
            vlm_embeds=prefix_embs,
            expert_embeds=None,
            attention_mask=prefix_add_2d_masks,
            position_ids=prefix_position_ids,
            fill_kv_cache=True,
        )


        dt = -1.0 / self.config.num_steps
        dt = np.array(dt, dtype=np.float32)

        x_t = noise  # torch.Size([1, 50, 32])
        time = np.array(1.0, dtype=np.float32)

        while time >= -dt / 2:
            expanded_time = np.broadcast_to(time, bsize)  # torch.Size([1])
            v_t = self.denoise_step(
                prefix_pad_masks, past_key_values, x_t, expanded_time
            )

            # Euler step
            # x_t += dt * v_t
            x_t += dt * v_t[0]
            time += dt

        return torch.from_numpy(x_t)  # back to torch type

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestamp):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, timestamp
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = np.broadcast_to(
            prefix_pad_masks[:, np.newaxis, :], (batch_size, suffix_len, prefix_len)
        ).copy()

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = np.concatenate(
            [prefix_pad_2d_masks, suffix_att_2d_masks], axis=2
        )
        prefix_offsets = np.sum(prefix_pad_masks, axis=-1)[:, None]
        position_ids = prefix_offsets + np.cumsum(suffix_pad_masks, axis=1) - 1

        # position_ids.shape torch.Size([1, 50])
        # suffix_embs.shape torch.Size([1, 50, 720])
        # full_att_2d_masks.shape torch.Size([1, 50, 163])
        # len(past_key_values) 16
        output_embeds, _ = self.vlme.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            vlm_embeds=None,
            expert_embeds=suffix_embs,
        )
        # outputs_embeds[1].shape torch.Size([1, 50, 720])
        suffix_out = output_embeds
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.astype(dtype=np.float32)
        v_t = self.action_out_proj(**{self.action_out_proj.input_names[0]: suffix_out})
        return v_t
