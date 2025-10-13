import onnxruntime
import time

from src.session.utils import get_etglow_provider_options, set_verbose_output, fix_model_dimensions

# DEPRECATED!

def get_vlme_onnx_symbols(sequence_size=20):
    symbols = {
        "batch": 1,
        # "batch_size": 1,
        # "sequence": sequence_size
    }

    return symbols


def setup_vlme_session(model_path, 
                               provider="CPU",
                               tracing=False, 
                               num_layers=0, 
                               run_dir="vlme_dir", 
                               kvc=False):
    model_name = str(model_path.parents[0]).replace(
        str(model_path.parents[1]), '').replace(r'/', '')

    session_options = onnxruntime.SessionOptions()
    print(session_options)
    set_verbose_output(session_options, True)

    session_options.enable_profiling = tracing
    # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    onnx_symbols = get_vlme_onnx_symbols()

    # fixed_model_path = fix_model_dimensions(model_path, onnx_symbols)

    provider_options = get_etglow_provider_options(num_layers,
                                                   run_dir,
                                                   kvc,
                                                   onnx_symbols)
    if provider == "CPU":
        session_etglow = onnxruntime.InferenceSession(
            model_path, sess_options=session_options)
    elif provider == "ET":
        session_etglow = onnxruntime.InferenceSession(
            # fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])
            # model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
            model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])

    return session_etglow
