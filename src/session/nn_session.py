import onnxruntime

from pathlib import Path
from typing import Literal

from src.session.utils import get_etglow_provider_options, set_verbose_output, fix_model_dimensions


class OnnxModule:
    def __init__(self,
                 model_path: Path,
                 provider: Literal["CPU", "ET"],
                 fix_dimensions=False,
                 input_names=[],
                 output_names=[]) -> None:
        self.session_options = onnxruntime.SessionOptions()

        self.num_layers = 0
        self.input_names = input_names
        self.output_names = output_names

        self.rundir = "sessionrundir"
        self.onnx_symbols = {}
        self.session = self.__get_onnx_session(
            model_path, provider, fix_dimensions)

    def __get_onnx_session(self, model_path, provider, fix_dimensions):
        if provider == "ET":
            if not self.onnx_symbols:
                raise AssertionError(
                    "Using ET provider you should fill onnx_symbols dict!")
            if fix_dimensions:
                model_path = fix_model_dimensions(
                    model_path, self.onnx_symbols)

            # temporaryly not using that options
            provider_options = get_etglow_provider_options(self.num_layers,
                                                           run_dir=self.rundir,
                                                           use_kvc=False,
                                                           onnx_symbols=self.onnx_symbols)

            session = onnxruntime.InferenceSession(
                model_path, sess_options=self.session_options, providers=[
                    'EtGlowExecutionProvider']
            )
        elif provider == "CPU":
            session = onnxruntime.InferenceSession(
                model_path, sess_options=self.session_options)
        else:
            raise ValueError(f"Unknown provider {provider}")
        
        if not self.input_names:
            self.input_names = [inp.name for inp in session.get_inputs()]

        if not self.output_names:
            self.output_names = [out.name for out in session.get_outputs()]
        
        print(f"Loaded onnx session for {model_path.__str__().split('/')[-1]} for provider {provider}")

        return session

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    # Only keword argumnts are supported
    def forward(self, **kwargs):
        for name in self.input_names:
            if name not in kwargs.keys():
                raise ValueError(f"{name} was not found for input names!")

        outputs = self.session.run(
            output_names=self.output_names, input_feed=kwargs)

        return outputs
