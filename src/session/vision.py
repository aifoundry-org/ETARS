import onnxruntime
import time

from src.session.glow import get_etglow_provider_options, set_verbose_output, fix_model_dimensions


def get_vis_onnx_symbols():
    symbols = {
        "batch_size": 1,
        "batch": 1,
        "width": 512,
        "height": 512,
        "channles": 3,
        "num_images": 1,
        "NonZero_7_o0__d1": 13,
        # "NonZero_7_o0__d1": 1,
        "image_sequence": 64,
    }

    return symbols

def setup_vision_session(model_path, provider="CPU",
                         tracing=False,
                         num_layers=10,
                         run_dir="visiondir",
                         kvc=False):
    model_name = str(model_path.parents[0]).replace(
        str(model_path.parents[1]), '').replace(r'/', '')

    session_options = onnxruntime.SessionOptions()
    print(session_options)

    session_options.enable_profiling = tracing
    # session_options.graph_optimization_level = get_graph_optimization_level("ORT_DISABLE_ALL")
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    onnx_symbols = get_vis_onnx_symbols()

    fixed_model_path = fix_model_dimensions(model_path, onnx_symbols)

    provider_options = get_etglow_provider_options(num_layers,
                                                   run_dir,
                                                   kvc,
                                                   onnx_symbols)

    if provider == "CPU":
        session_etglow = onnxruntime.InferenceSession(
            model_path, sess_options=session_options)
    elif provider == "ET":
        session_etglow = onnxruntime.InferenceSession(
            fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])
        # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])

    return session_etglow