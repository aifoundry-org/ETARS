import onnxruntime

from pathlib import Path
from typing import Literal, Optional

from src.session.utils import get_etglow_provider_options, set_verbose_output, fix_model_dimensions


class OnnxModule:
    """
    Lightweight wrapper around ONNX Runtime that loads a model either from a local
    file path or directly from the Hugging Face Hub, and provides a keyword-only
    `forward` interface matching ONNX input/output names.

    Parameters
    ----------
    model_path : pathlib.Path | None, default=None
        Local path to a `.onnx` file. Required if `hf_repo` is not set.
    provider : {"CPU", "ET"}, default="CPU"
        Execution provider to use. "CPU" uses the default ONNX Runtime CPU EP.
        "ET" uses EtGlowExecutionProvider (requires it to be available).
    fix_dimensions : bool, default=False
        If True (and provider=="ET"), calls `fix_model_dimensions(model_path, onnx_symbols)`
        before creating the session.
    input_names : list[str], default=[]
        Explicit ONNX input names. If empty, they are inferred from the model.
    output_names : list[str], default=[]
        Explicit ONNX output names. If empty, they are inferred from the model.
    rundir : str, default="sessionrundir"
        Working directory passed to ET provider options (if enabled).
    use_provider_options : bool, default=False
        If True and provider=="ET", builds provider options via
        `get_etglow_provider_options(num_layers, run_dir, use_kvc, onnx_symbols)`.

    Hugging Face Options
    --------------------
    hf_repo : str | None, default=None
        Repo ID on the Hugging Face Hub (e.g., "org/repo"). If set, the model is
        downloaded via `huggingface_hub.hf_hub_download`.
    hf_filename : str, default="model.onnx"
        File name within the repository to download.
    hf_revision : str | None, default=None
        Branch, tag, or commit hash to pin the download.
    hf_token : str | None, default=None
        Authentication token for private repos.
    hf_cache : str | None, default=None
        Custom cache directory for downloads.

    Attributes
    ----------
    session_options : onnxruntime.SessionOptions
        Session options used to construct the ONNX Runtime session.
    session : onnxruntime.InferenceSession
        The underlying ONNX Runtime session.
    input_names : list[str]
        Finalized list of input names used by the model/session.
    output_names : list[str]
        Finalized list of output names used by the model/session.
    num_layers : int
        Exposed for provider option builders (e.g., ET/Glow); not inferred here.
    onnx_symbols : dict[str, int]
        Symbolic dimension bindings used by ET utilities (default: {"batch": 1}).
    rundir : str
        Run directory for ET provider artifacts.

    Methods
    -------
    forward(**kwargs) -> list[np.ndarray]
        Run inference. `kwargs` must map every `input_names[i]` to a NumPy array
        (dtype/shape compatible with the model). Returns outputs in the order of
        `output_names`.
    __call__(**kwargs)
        Alias to `forward`.

    Dependencies
    ------------
    - Always: `onnxruntime`
    - If `hf_repo` is set: `huggingface_hub`
    - If `provider=="ET"`: EtGlowExecutionProvider must be installed and available.
      Optional utilities `fix_model_dimensions` and `get_etglow_provider_options`
      are expected to be defined elsewhere.

    Raises
    ------
    ValueError
        If neither `model_path` nor `hf_repo` is provided, or if an expected input
        is missing in `forward(**kwargs)`.
    RuntimeError
        If `huggingface_hub` is required but not installed.
    AssertionError
        If `provider=="ET"` and `onnx_symbols` is empty.

    Examples
    --------
    # Local model
    m = OnnxModule(model_path=Path("model.onnx"), provider="CPU")
    y = m.forward(**{m.input_names[0]: x})

    # From Hugging Face Hub (public repo)
    m = OnnxModule(hf_repo="org/repo", hf_filename="model.onnx", provider="CPU")

    # From HF with revision/token and custom cache
    m = OnnxModule(hf_repo="org/repo",
                   hf_filename="vision.onnx",
                   hf_revision="main",
                   hf_token="hf_***",
                   hf_cache="/tmp/hf-cache")

    # ET provider with options
    m = OnnxModule(model_path=Path("model.onnx"),
                   provider="ET",
                   use_provider_options=True,
                   fix_dimensions=True)
    """
    def __init__(self,
                 model_path: Optional[Path] = None,
                 provider: Literal["CPU", "ET"] = "CPU",
                 fix_dimensions: bool = False,
                 input_names: list[str] = [],
                 output_names: list[str] = [],
                 rundir: str = "sessionrundir",
                 use_provider_options: bool = False,
                 # --- Hugging Face options ---
                 hf_repo: Optional[str] = None,
                 hf_filename: str = "model.onnx",
                 hf_revision: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 hf_cache: Optional[str] = None) -> None:

        self.session_options = onnxruntime.SessionOptions()
        self.use_et_provider_options = use_provider_options

        self.num_layers = 0
        self.input_names = input_names
        self.output_names = output_names

        self.rundir = rundir
        self.onnx_symbols = {"batch": 1}
        resolved_model_path = self.__resolve_model_path(
            model_path=model_path,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            hf_revision=hf_revision,
            hf_token=hf_token,
            hf_cache=hf_cache,
        )
        self.session = self.__get_onnx_session(
            resolved_model_path, provider, fix_dimensions)
    
    def __resolve_model_path(self,
                             model_path: Optional[Path],
                             hf_repo: Optional[str],
                             hf_filename: str,
                             hf_revision: Optional[str],
                             hf_token: Optional[str],
                             hf_cache: Optional[str]) -> Path:
        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download
            except Exception as e:
                raise RuntimeError(
                    "huggingface_hub is required to load from Hugging Face. "
                    "Install with: pip install huggingface_hub"
                ) from e

            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                revision=hf_revision,
                token=hf_token,
                cache_dir=hf_cache,
                local_files_only=False,
            )
            return Path(downloaded)
        
        if model_path is None:
            raise ValueError("Provide either a local model_path or hf_repo + hf_filename.")
        return Path(model_path)

    def __get_onnx_session(self, model_path, provider, fix_dimensions):
        if provider == "ET":
            if not self.onnx_symbols:
                raise AssertionError(
                    "Using ET provider you should fill onnx_symbols dict!")
            if fix_dimensions:
                model_path = fix_model_dimensions(
                    model_path, self.onnx_symbols)

            if self.use_et_provider_options:
                provider_options = get_etglow_provider_options(self.num_layers,
                                                            run_dir=self.rundir,
                                                            use_kvc=False,
                                                            onnx_symbols=self.onnx_symbols)
            else:
                provider_options = {}

            session = onnxruntime.InferenceSession(
                model_path, 
                sess_options=self.session_options, 
                providers=['EtGlowExecutionProvider'], 
                provider_options=[provider_options]
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
