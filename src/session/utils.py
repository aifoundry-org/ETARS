import onnx
import onnxruntime

from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
from pathlib import Path

def get_etglow_api_params(num_layers,

                          run_dir,
                          use_kvc,
                          iobindings=False,
                          sep=';'):
    """Define etglow api parameters"""
    def get_device_placeholders(num_layers, sep=';'):
        """Define implicit placheolders."""
        dph = ""
        for id in range(num_layers):
            dph += f"{sep}implicit-placeholder=past_key_values.{id}.key"
            dph += f"{sep}implicit-placeholder=past_key_values.{id}.value"
            dph += f"{sep}placeholder=present.{id}.key"
            dph += f"{sep}placeholder=present.{id}.value"
        return dph

    dev_params = " ".join(
        ['--gccCompileThreads=32', '--logDisableCodeGenBits=-1', ])
    extra_params = '|'.join(['debug-glow=0', f'dev={dev_params}'])
    api_params = [
        "device-type=silicon",
        "glow-threads=2",
        f"runDir={run_dir}",
        f"extra-etsoc-params='{extra_params}'"
    ]
    api_params = sep.join(api_params)
    if use_kvc and iobindings:
        api_params += get_device_placeholders(num_layers, sep)
    return api_params

def get_onnx_shape_params(onnx_symbols):
    """Define onnx shape parameters"""
    onnx_shape_params = ''
    for key, value in onnx_symbols.items():
        onnx_shape_params += f"{key}={value},"
    # Return generated string without the last comma character
    return onnx_shape_params[:-1]

def get_etglow_provider_options(num_layers, 
                                run_dir, 
                                use_kvc, 
                                onnx_symbols) -> dict:
    """Constructs ETGLOW EP provider options to run LLMs"""
    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": get_onnx_shape_params(onnx_symbols),
        "etglow_api_params": get_etglow_api_params(num_layers, run_dir, use_kvc)
    }
    return poptions

def set_verbose_output(options, enabled):
    log_severity_verbose = 0
    log_severity_warning = 2
    log_severity_error = 3
    
    if not enabled:
        onnxruntime.set_default_logger_severity(log_severity_error)
        options.log_severity_level = log_severity_error
    else:
        onnxruntime.set_default_logger_severity(log_severity_verbose)
        options.log_severity_level = log_severity_warning


def fix_model_dimensions(model_path : Path, onnx_symbols: dict) -> Path:
    # Fix dimensions in the model
    model_noexternaldata = onnx.load(model_path, load_external_data=False)
    for key, value in onnx_symbols.items():
        make_dim_param_fixed(model_noexternaldata.graph, key, value)
    model_noexternaldata = model_noexternaldata.SerializeToString()
    fixeddim_model_path = model_path.parent / "model-fixed-dims.onnx"
    with open(fixeddim_model_path, "wb") as f:
        f.write(model_noexternaldata)
    return fixeddim_model_path