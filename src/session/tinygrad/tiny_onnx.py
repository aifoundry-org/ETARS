from pathlib import Path
from typing import Any, Literal, Optional
from src.session.utils import resolve_model_path

from tinygrad.nn.onnx import OnnxRunner


class TinyOnnxModule:
    def __init__(
        self,
        provider: Optional[str] = None, # stub for now
        model_path: Optional[Path] = None,
        input_names: list[str] = [],
        output_names: list[str] = [],
        # --- Hugging Face options ---
        hf_repo: Optional[str] = None,
        hf_filename: str = "model.onnx",
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_cache: Optional[str] = None,
    ) -> None:
        self.resolved_model_path = resolve_model_path(
            model_path=model_path,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            hf_revision=hf_revision,
            hf_token=hf_token,
            hf_cache=hf_cache,
        )

        self.input_names = input_names
        self.output_names = output_names

        self.session = OnnxRunner(self.resolved_model_path)

        if not self.input_names:
            self.input_names = [*self.session.graph_inputs.keys()]
        self.debug = 0 # 0,1,2,3


    
    def __call__(self, **kwargs: Any) -> Any:
        return self.forward(**kwargs)
    
    def forward(self, **kwargs):
        return [out.realize() for out in self.session(inputs=kwargs, debug=self.debug).values()]
        
