import onnxruntime
import time

from src.session.utils import get_etglow_provider_options, set_verbose_output, fix_model_dimensions


def get_encoder_onnx_symbols(sequence_size=20):
    symbols = {
        "batch": 1,
        "batch_size": 1,
        "sequence": sequence_size
    }

    return symbols


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


def setup_encoder_session(model_path, args):
    model_name = str(model_path.parents[0]).replace(
        str(model_path.parents[1]), '').replace(r'/', '')

    session_options = onnxruntime.SessionOptions()
    print(session_options)
    set_verbose_output(session_options, args.verbose)

    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    onnx_symbols = get_encoder_onnx_symbols()

    provider_options = get_etglow_provider_options(args.num_layers,
                                                   args.run_dir,
                                                   args.use_kvc,
                                                   onnx_symbols)

    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{args.window_size}'

    # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    session_etglow = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])

    etsoc_comp_time = time.time() - start

    return session_etglow


def setup_vision_session(model_path, args):
    model_name = str(model_path.parents[0]).replace(
        str(model_path.parents[1]), '').replace(r'/', '')

    session_options = onnxruntime.SessionOptions()
    print(session_options)
    set_verbose_output(session_options, args.verbose)

    session_options.enable_profiling = args.enable_tracing
    # session_options.graph_optimization_level = get_graph_optimization_level("ORT_DISABLE_ALL")
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    onnx_symbols = get_vis_onnx_symbols()

    fixed_model_path = fix_model_dimensions(model_path, onnx_symbols)

    provider_options = get_etglow_provider_options(args.num_layers,
                                                   args.run_dir, 
                                                   args.use_kvc,
                                                   onnx_symbols)

    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{args.window_size}'

    # session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    session_etglow = onnxruntime.InferenceSession(
        fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])
    # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options)
    # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])

    etsoc_comp_time = time.time() - start

    return session_etglow
