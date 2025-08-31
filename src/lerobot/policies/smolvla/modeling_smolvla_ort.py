from src.session import vision, text_encoder, head, vlm_exp, state

import numpy as np
import onnxruntime as ort

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None 


# [B*C,3,H,W] ──► VISION sess ──reshape──► [B, C*Simg, Dvlm]
# [B,Stxt]    ──► TEXT   sess ────────────► [B, Stxt,  Dvlm]
# [B,…] (opt) ──► STATE  sess ────────────► [B, Sstate,Dvlm]
#                           concat tokens ─► emb0: [B, S0, Dvlm]
# action queries ──────────────────────────► emb1: [B, S1, de_in]
# build: mask [B,S,S] (bool), pos [B,S] (int64), where S=S0+S1
# (mask,pos,emb0,emb1) ──► CORE sess ──► expert_hidden [B,S1,Dexp]
# expert_hidden ──(opt)► HEAD sess ──► actions [B,S1,act_dim]

class smolVLAFlow():
    def __init__(self, vision_path, text_path, core_path, *, state_path=None, head_path=None) -> None:
        self.vision = vision.setup_vision_session(vision_path,
                                                   "CPU",
                                                   False,
                                                   num_layers=12)
        self.text = text_encoder.setup_text_encoder_session(text_path, "ET")
        self.vlm_exp = vlm_exp.setup_vlm_expert_session()
        self.state = state.get_state_session()
        self.head = head.setup_head_session()
    
    def forward(self):
        pass

    def sample_action(self):
        pass