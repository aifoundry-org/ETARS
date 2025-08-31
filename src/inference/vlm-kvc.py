import onnxruntime
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
from onnxruntime.transformers.io_binding_helper import IOBindingHelper, TypeHelper

from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from transformers.image_utils import load_image
from scipy.special import softmax
import onnx
import numpy as np
import math
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from pathlib import Path
from difflib import SequenceMatcher
import sys
import os

class PathValidationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            raise argparse.ArgumentError(self, f"The file '{values}' does not exist.")
        setattr(namespace, self.dest, values)

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Get the perplexity for a prompt and a model")
    parser.add_argument("-m", '--model-path', metavar = "DIR", type = Path, default = './model.onnx',  action=PathValidationAction,
                        help = 'Path to the model.onnx')
    parser.add_argument("-t", '--tokenizer', metavar = "DIR", type = Path, default = '.', action=PathValidationAction,
                         help = 'Path to the tokenizer folder')
    parser.add_argument("-p", '--prompt', metavar = "TEXT", type = str, default = 'Who is Messi?',
                        help = 'Given Prompt')
    parser.add_argument("-s", '--sequence-length', type = int, default = 256,
                        help = 'Total sequence length for the run')
    parser.add_argument("-c", '--context', type = int, default = 2048,
                        help = 'Total context length for the run')
    parser.add_argument("-n", '--no-context', action="store_true", default=False, 
                        help = 'If used, the model does not have the context extension in its ONNX')
    parser.add_argument("-g", '--generate-tokens', type = int, default = 0,
                        help = 'Total sequence length for the run')
    parser.add_argument('--enable-tracing', action = 'store_true', default = False,
                        help = 'Enable onnxruntime profiling and neuralizer traces')
    parser.add_argument('-v', '--verbose', action = 'store_true', default = False,
                        help = 'Show more output. Including ONNXRuntime logs')

    # options to be validated
    parser.add_argument("-r", '--run-dir', metavar = "DIR", type = str, default = 'myrundir',
                        help = 'Path to the run directory created by neuralizer')
    parser.add_argument('--use-kvc', action="store_true", default=True,
                        help = 'Enables KV-cache')
    parser.add_argument("-w", '--window-size', type = int, default = 1,
                        help = 'Maximum number of tokens that a KV-cache model can process per inference (must be smaller than the total sequence length)')
    parser.add_argument("-b", '--batch', type = int, default = 1,
                        help = 'Number of batched inferences')
    parser.add_argument("--optimization-level", type = str, default = 'ORT_ENABLE_ALL', choices = ['ORT_DISABLE_ALL', 'ORT_ENABLE_BASIC', 'ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL'],
                        help = 'Graph Optimization level in ONNX Runtime')
    args = parser.parse_args()

    if args.window_size > args.sequence_length:
        parser.error(f"Argument --window-size ({args.window_size}) should be less or equal to --sequence-legth ({args.sequence_length})")

    if args.no_context:
        args.context = None
    return args


# Define ONNX Runtime Graph Optimization level
def get_graph_optimization_level(level):
    match level:
        case 'ORT_DISABLE_ALL':
            return onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        case 'ORT_ENABLE_BASIC':
            return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        case 'ORT_ENABLE_EXTENDED':
            return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        case 'ORT_ENABLE_ALL':
            return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        case _:
            sys.exit(f"Error in setting ORT Optimization Level. The optimization level {level} is not recognized")

def get_etglow_api_params(args, sep=';'):
    """Define etglow api parameters"""
    def get_device_placeholders(args, sep=';'):
        """Define implicit placheolders."""
        dph=""
        for id in range(args.num_layers):
            dph+=f"{sep}implicit-placeholder=past_key_values.{id}.key"
            dph+=f"{sep}implicit-placeholder=past_key_values.{id}.value"
            dph+=f"{sep}placeholder=present.{id}.key"
            dph+=f"{sep}placeholder=present.{id}.value"
        return dph

    dev_params = " ".join(['--gccCompileThreads=32','--logDisableCodeGenBits=-1', ])
    extra_params = '|'.join(['debug-glow=0', f'dev={dev_params}'])
    api_params = [
        "device-type=silicon",
        "glow-threads=2",
        f"runDir={args.run_dir}",
        f"extra-etsoc-params='{extra_params}'"
    ]
    api_params = sep.join(api_params)
#    if args.use_kvc and args.etglow_implementation == "llm_kvc_inference_iobindings":
#        api_params += get_device_placeholders(args, sep)
    return api_params

def get_sequence_size(args):
    """Get sequence size for a model"""
    return args.window_size if args.use_kvc else args.sequence_length

def get_past_sequence_size(args):
    """Get past sequence size for a model"""
    return args.sequence_length - args.window_size if args.use_kvc else 0

def get_context_size(args):
    """Get context size for a model"""
    if args.context is None:
        return get_sequence_size(args) + get_past_sequence_size(args)
    else:
        return args.context

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
        # "batch_size*num_images": 1
    }

    return symbols

def get_decoder_onnx_symbols():
    symbols = {
        "batch_size": 1,
        "sequence_length": 256,
        "total_sequence_length": 256,
        "past_sequence_length": 256
    }

    return symbols

# Defines dictionaries of onnx symbols
def get_onnx_symbols(args) -> dict[str, any]:
    if not args.use_kvc:
        symbols = {
            "batch": args.batch,
            "batch_size": args.batch,
            "sequence": get_sequence_size(args)
        }
    else:
        symbols = {
        "batch": args.batch,
        "sequence": get_sequence_size(args),
        "past_sequence": get_past_sequence_size(args),
        "past_sequence+sequence": get_sequence_size(args) + get_past_sequence_size(args),
        "context": get_context_size(args),
        "context-sequence": get_context_size(args) - get_sequence_size(args)
    }
    return symbols


def get_onnx_shape_params(onnx_symbols):
    """Define onnx shape parameters"""
    onnx_shape_params = ''
    for key, value in onnx_symbols.items():
        onnx_shape_params += f"{key}={value},"
    # Return generated string without the last comma character
    return onnx_shape_params[:-1]


def get_etglow_provider_options(args, onnx_symbols, fake=False) -> dict:
    """Constructs ETGLOW EP provider options to run LLMs"""
    poptions = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": get_onnx_shape_params(onnx_symbols),
        "etglow_api_params": get_etglow_api_params(args)
    }
    return poptions

def vlm_inference(embed_session, vision_session, decoder_session):
    # 2. Prepare inputs
    ## Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"}
            ]
        },
    ]

    ## Load image and apply processor
    image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="np")

    ## Prepare decoder inputs
    batch_size = inputs['input_ids'].shape[0]
    past_key_values = {
        f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
        for layer in range(num_hidden_layers)
        for kv in ('key', 'value')
    }
    image_features = None
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    position_ids = np.cumsum(inputs['attention_mask'], axis=-1)

    # 3. Generation loop
    max_new_tokens = 1024
    generated_tokens = np.array([[]], dtype=np.int64)
    for i in range(max_new_tokens):
        inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

        if image_features is None:
            ## Only compute vision features if not already computed
            image_features = vision_session.run(
                ['image_features'],  # List of output names or indices
                {
                    'pixel_values': inputs['pixel_values'],
                    'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
                }
            )[0]
            
            ## Merge text and vision embeddings
            inputs_embeds[inputs['input_ids'] == image_token_id] = image_features.reshape(-1, image_features.shape[-1])

        logits, *present_key_values = decoder_session.run(None, dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **past_key_values,
        ))

        ## Update values for next generation loop
        input_ids = logits[:, -1].argmax(-1, keepdims=True)
        attention_mask = np.ones_like(input_ids)
        position_ids = position_ids[:, -1:] + 1
        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_key_values[j]

        generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
        if (input_ids == eos_token_id).all():
            break

        ## (Optional) Streaming
        print(processor.decode(input_ids[0]), end='')
    print()

    # 4. Output result
    print(processor.batch_decode(generated_tokens))
    

def llm_kvc_inference(session : onnxruntime.InferenceSession, run_options : onnxruntime.RunOptions,
                      tokenizer : AutoTokenizer, input_tensors : dict,
                      prompt_tensor, num_tokens, context, sequence_len, window : int, batch : int) -> tuple[str, float]:
    sum_perplexity = 0
    new_token = np.array([10])
    prompt_size = len(prompt_tensor[0])
    total_input = prompt_tensor
    current_index = 0
    next_index = window

    inputs_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    # Run the inferences
    while next_index < num_tokens:
        if (new_token == tokenizer.eos_token_id).any():
            break

        output = session.run(output_names, input_tensors, run_options)

        outs_dictionary = {name: content for (name, content) in zip (output_names, output)}

        logits = outs_dictionary['logits']

        # Prepare next inference inputs
        for name in inputs_names:
            if name == 'input_ids':
                current_index = next_index
                next_index = update_next_index(next_index, prompt_size, window)
                j = next_index - window

                if current_index >= prompt_size:
                    top1 = logits.argmax(-1)
                    new_token = top1.reshape(batch, window) # Inf server
                    total_input = np.concatenate((total_input, new_token[: , -1:]), axis = 1)

                input_tensors['input_ids'] = total_input[:, j:next_index].reshape(batch, window) # inf server
            elif name == 'attention_mask':
                attention_mask = np.zeros((batch, sequence_len), dtype='int64')
                attention_mask[:, -next_index:] = 1
                input_tensors['attention_mask'] = attention_mask  # inf server
            elif name == 'position_ids':
                input_tensors['position_ids'] = np.concatenate(
                                                  (np.zeros([1, sequence_len - next_index], dtype='int64'),
                                                   np.arange(next_index, dtype='int64').reshape(1, next_index)), axis=1)
            elif name == 'tree_attention':
                continue
            else:
                old_name = name.replace("past_key_values", "present")
                start_idx = next_index - current_index
                end_idx = start_idx + context - window
                input_tensors[name] = outs_dictionary[old_name][:, :, start_idx:end_idx, :]

        sum_perplexity += -np.log(softmax(logits[0, 0])[input_tensors['input_ids'][0]])

    sum_perplexity /= (num_tokens - 1)
    # answers = tokenizer.batch_decode(total_input, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    answers = tokenizer.decode(total_input[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answers, sum_perplexity

# Function to calculate perplexity
def compute_perplexity(logits, next_input_ids):
    # Perform the softmax and perplexity calculation
    return -np.log(softmax(logits)[next_input_ids])

def preallocate_input_outputs(session : onnxruntime.InferenceSession, output_names: list[str], input_tensors: dict,
                              window: int, batch: int, nheads: int, context: int, hidden: int, logits_last_dim: int,
                              device_type="et", device_id=0) -> dict:
    ortvalues = {
        'input_ids':      onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['input_ids'], device_type, device_id),
        'attention_mask': onnxruntime.OrtValue.ortvalue_from_numpy(input_tensors['attention_mask'], device_type, device_id),
        'logits':      onnxruntime.OrtValue.ortvalue_from_shape_and_type((batch, window, logits_last_dim), TypeHelper.ort_type_to_numpy_type(TypeHelper.get_output_type(session, 'logits')), device_type, device_id),
    }
    # All KVC input (past_key_value) & output (present) tensors will share same underlying allocation.
    zeros = np.zeros((batch, nheads, context, hidden), dtype=TypeHelper.ort_type_to_numpy_type(TypeHelper.get_output_type(session, 'present.0.key')))
    zeros_padded = np.pad(zeros, ((0,0), (0,1), (0, 0), (0,0)), mode='constant')
    for name in output_names:
        if 'present' in name:
            ortvalues[name] = onnxruntime.OrtValue.ortvalue_from_numpy(zeros_padded, device_type, device_id)
    return ortvalues


def bind_input_outputs(io_binding, inputs_names, outputs_names, ortvalues, batch : int, nheads : int, context : int, hidden : int):
    # Bind inputs (will bind them to the allocated memory)
    for name in inputs_names:
        if name in ['input_ids', 'attention_mask']:
            # For input_ids or attention_mask lets bind the ortvalue directly
            io_binding.bind_ortvalue_input(name, ortvalues[name])
        else:
            # For 'past_key_value' we need to bind the buffer_ptr to the underlying allocation
            out_name = name.replace("past_key_values", "present")
            io_binding.bind_input(name,
                                  device_type=ortvalues[out_name].device_name(),
                                  device_id=0,
                                  element_type=np.float16,
                                  shape=(batch, nheads, context - 1, hidden),
                                  buffer_ptr=ortvalues[out_name].data_ptr())
    # Bind outputs (to non pre-allocated output)
    for name in outputs_names:
        if 'logits' in name:
            io_binding.bind_ortvalue_output(name, ortvalues[name])
        else:
            io_binding.bind_output(name,
                                   device_type=ortvalues[name].device_name(),
                                   device_id=0,
                                   element_type=np.float16,
                                   shape=(batch, nheads, context, hidden),
                                   buffer_ptr=ortvalues[name].data_ptr())

def update_kvc_view(io_binding, name, ortvalues, current_index, batch : int, nheads, context, hidden):
    out_name = name.replace("past_key_values", "present")
    last_dim_value = hidden
    element_size_bytes = 2
    num_bytes_per_cache_line = last_dim_value * element_size_bytes
    offset_bytes = current_index * num_bytes_per_cache_line
    buffer_ptr = ortvalues[out_name].data_ptr() + offset_bytes
    device_type = ortvalues[out_name].device_name()
    io_binding.bind_input(name,
                          device_type=device_type,
                          device_id=0,
                          element_type=np.float16,
                          shape=(batch, nheads, context - 1, hidden),
                          buffer_ptr=buffer_ptr)
    io_binding.bind_output(out_name,
                           device_type=device_type,
                           device_id=0,
                           element_type=np.float16,
                           shape=(batch, nheads, context, hidden),
                           buffer_ptr=buffer_ptr)

def update_next_index(next_index, prompt_size, window):
    if next_index < prompt_size:
        # Next token is part of the prompt
        if prompt_size - next_index >= window:
            next_index += window
        else:
            next_index = prompt_size
    else:
        # Next token is part of the answer
        next_index += 1
    return next_index


def get_prompt_tensor(prompt, tokenizer, batch):
    input_tensor = tokenizer(prompt, return_tensors="pt")
    input_ids = input_tensor['input_ids']
    batched_input_ids = np.tile(input_ids,(batch, 1))
    return batched_input_ids


def fix_model_dimensions(model_path : Path, onnx_symbols: dict) -> Path:
    # Fix dimensions in the model
    model_noexternaldata = onnx.load(model_path, load_external_data=False)
    for key, value in onnx_symbols.items():
        make_dim_param_fixed(model_noexternaldata.graph, key, value)
    model_noexternaldata = model_noexternaldata.SerializeToString()
    fixeddim_model_path = model_path.parent / f"{model_path.name}_model-fixed-dims.onnx"
    with open(fixeddim_model_path, "wb") as f:
        f.write(model_noexternaldata)
    return fixeddim_model_path


def preprocess_llm_input_tensors(session : onnxruntime.InferenceSession, input_ids_tensor: np.ndarray, args) -> dict:
    window = args.window_size
    batch = args.batch
    pads = args.sequence_length
    context = get_context_size(args)

    first_attention = np.concatenate((np.zeros([batch, pads - window], dtype = 'int64'), np.ones((batch, window), dtype = 'int64')), axis=1)
    index_pos = sum(first_attention[0])
    
    inputs_names = [input.name for input in session.get_inputs()]
    inputs_dict = {
        'input_ids': copy.deepcopy(input_ids_tensor[:, :window].reshape(batch, window)),
        'attention_mask': first_attention,
        'position_ids': np.concatenate((np.zeros([batch, pads - index_pos], dtype='int64'),
                                         np.arange(index_pos, dtype='int64').reshape(1, index_pos)), axis=1),
        'tree_attention': np.triu(-65504 * np.ones(pads), k=1).astype('float16').reshape(1, 1, pads, pads),
    }
    if args.use_kvc:
        # deduce n_heads and hidden from model using ORT
        n_heads = int(session.get_outputs()[1].shape[-3])
        hidden = int(session.get_outputs()[1].shape[-1])
        for name in inputs_names:
            if name not in ['input_ids','attention_mask','position_ids','tree_attention']:
                inputs_dict[name] = np.zeros([batch, n_heads, context - window, hidden], dtype="float16")
    return inputs_dict


def print_llm_inference_setup(model_name, prompt_size, generate_tokens, onnx_symbols):
    print(f"Running model: {model_name}")
    print(f"    Prompt size: : {prompt_size}")
    print(f"    Generating #tokens: {generate_tokens}")
    print(f"    ONNX symbols: {onnx_symbols}")


def print_llm_inference_results(label : str, comp_time : float, inf_time : float,  perplexity, answers : [str], num_tokens):
    print(f'{label}:')
    print(f'    Compilation took {comp_time:.2f}s')
    print(f'    Inference took {inf_time:.2f}s ({(num_tokens / inf_time):.2f} tokens/s)')
    print(f'    Perplexity value: {float(perplexity.item()):.2f}')
    for i in range(len(answers)):
        print(f'    Question and answer[{i}]: {answers[i]}')

def compare_results(perplexity_cpu, perplexity_etglow, answers_cpu : [str], answers_etglow : [str]):
    perplexity_tolerance = 0.05  # Define a relative tolerance for perplexity comparison
    similarity_threshold = 0.9  # Set a threshold for answer similarity

    if not math.isclose(perplexity_cpu, perplexity_etglow, rel_tol=perplexity_tolerance):
        # preplexities should be close
        raise ValueError(f"Perplexity scores differ significantly: CPU = {perplexity_cpu}, ETGlow = {perplexity_etglow}. "
                         f"Difference exceeds tolerance of {perplexity_tolerance*100}%.")

    for i, (answer_cpu, answer_etglow) in enumerate(zip(answers_cpu, answers_etglow)):
        # Calculate similarity ratio using SequenceMatcher
        similarity_ratio = SequenceMatcher(None, answer_cpu, answer_etglow).ratio()
        if similarity_ratio < similarity_threshold:
            raise ValueError(f"Answer {i+1} differs significantly between CPU and ETGlow. "
                             f"Similarity ratio: {similarity_ratio:.4f} (Threshold: {similarity_threshold}).\n"
                             f"CPU Answer: {answer_cpu}\nETGlow Answer: {answer_etglow}")

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

def setup_encoder_session(model_path, args):
    model_name = str(model_path.parents[0]).replace(str(model_path.parents[1]), '').replace(r'/', '')

    session_options = onnxruntime.SessionOptions()
    print(session_options)
    set_verbose_output(session_options, args.verbose)

    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = get_graph_optimization_level(args.optimization_level)

    onnx_symbols = get_onnx_symbols(args)

    provider_options = get_etglow_provider_options(args, onnx_symbols)

    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{args.window_size}'

    # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])

    etsoc_comp_time = time.time() - start 

    return session_etglow

def setup_vision_session(model_path, args):
    model_name = str(model_path.parents[0]).replace(str(model_path.parents[1]), '').replace(r'/', '')

    session_options = onnxruntime.SessionOptions()
    print(session_options)
    set_verbose_output(session_options, args.verbose)

    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = get_graph_optimization_level("ORT_DISABLE_ALL")

    onnx_symbols = get_vis_onnx_symbols()

    fixed_model_path = fix_model_dimensions(model_path, onnx_symbols)

    provider_options = get_etglow_provider_options(args, onnx_symbols)

    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{args.window_size}'

    # session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    # session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])
    session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options)
    # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'])

    etsoc_comp_time = time.time() - start 

    return session_etglow


def setup_decoder_session(model_path, args):
    context_len = get_context_size(args)

    model_name = str(model_path.parents[0]).replace(str(model_path.parents[1]), '').replace(r'/', '')
    
    session_options = onnxruntime.SessionOptions()
    print(session_options)
    set_verbose_output(session_options, args.verbose)
    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = get_graph_optimization_level("ORT_ENABLE_ALL")

    onnx_symbols = get_decoder_onnx_symbols()

    fixed_model_path = fix_model_dimensions(model_path, onnx_symbols)

    # Create etglow Provider options
    provider_options = get_etglow_provider_options(args, onnx_symbols)

    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{args.window_size}'

    # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])
    # session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options)
    # session_etglow = onnxruntime.InferenceSession(model_path, sess_options=session_options)

    etsoc_comp_time = time.time() - start 

    return session_etglow




def setup_inference_session(model_path, args):

    context_len = get_context_size(args)
    model_name = str(model_path.parents[0]).replace(str(model_path.parents[1]), '').replace(r'/', '')


    session_options = onnxruntime.SessionOptions()
    print(session_options)
    set_verbose_output(session_options, args.verbose)
    session_options.enable_profiling = args.enable_tracing
    session_options.graph_optimization_level = get_graph_optimization_level(args.optimization_level)

    onnx_symbols = get_onnx_symbols(args)

    fixed_model_path = fix_model_dimensions(model_path, onnx_symbols)

    # Create etglow Provider options
    provider_options = get_etglow_provider_options(args, onnx_symbols)

    start = time.time()
    session_options.profile_file_prefix = f'{model_name}_etglow_window_{args.window_size}'

    session_etglow = onnxruntime.InferenceSession(fixed_model_path, sess_options=session_options, providers=['EtGlowExecutionProvider'], provider_options=[provider_options])

    etsoc_comp_time = time.time() - start 

    return session_etglow



def main():
    args = parse_arguments()
    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"args {args}")

    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # Check sequence length is long enough
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("Esperanto/llama3-8b-Instruct-kvc-fp16-onnx")
    # prompt_tensor = get_prompt_tensor(args.prompt, tokenizer, args.batch)
    # prompt_size = len(prompt_tensor[0])
    # assert(prompt_size + args.generate_tokens <= args.sequence_length), f"Sequence length is not long enough (should be at least: {prompt_size + args.generate_tokens}"

    # Exit if the model does not use KVC
    # model = onnx.load(model_path)
    # assert(len(model.graph.input) > 2), "Non Key-Value Cache models are not yet supported"

    onnx_symbols = get_onnx_symbols(args)
    # print_llm_inference_setup(model_name, prompt_size, args.generate_tokens, onnx_symbols)

    # Data section
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
    ]

    image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)


    # EmbedSession

    embed_model_path =  Path("/home/et/onnx-experiments/smolVLM/embed_tokens.onnx")
    args.num_layers = 0
    # args.use_kvc = False
    args.use_kvc = True
    # embed_session = setup_encoder_session(embed_model_path, args)
    inputs = processor(text=prompt, images=[image], return_tensors="np")
    input_ids = inputs['input_ids']

    # inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

    print("Text embeds generated.")
    # # VisionSession

    # vision_model_path = Path("/home/et/onnx-experiments/smolVLM/vision_encoder_fixed.onnx")
    vision_model_path = Path("/home/et/onnx-experiments/smolVLM/vision_encoder.onnx")
    # vision_model_path = Path("/home/et/onnx-experiments/smolVLM/vision_encoder-kvc.onnx")
    args.num_layers = 0
    # vision_session = setup_vision_session(vision_model_path, args)

    # image_features = vision_session.run(
    #     ['image_features'],  # List of output names or indices
    #     {
    #         'pixel_values': inputs['pixel_values'],
    #         'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
    #     }
    # )[0]

    # return(0)


    print("Vision embeds generated.")


    # # DecoderSession

    # decoder_model_path = Path("/home/et/onnx-experiments/smolVLM/decoder_model_merged-kvc.onnx")
    # decoder_model_path = Path("/home/et/onnx-experiments/smolVLM/decoder_model_merged_noscap_decomposed.onnx")
    decoder_model_path = Path("/home/et/onnx-experiments/smolVLM/decoder_model_merged_noscap_decomposed.onnx")
    args.num_layers = 24
    decoder_session = setup_decoder_session(decoder_model_path, args)

    print("decoder session created")

    pass



    # etsoc_comp_time = time.time() - start 

    # run_options_etglow = onnxruntime.RunOptions()
    # run_options_etglow.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")  # force memory shrinkage

    # # Process inputs
    # input_tensors_etglow = preprocess_llm_input_tensors(session_etglow, prompt_tensor, args)

    # # Launch ETSoC inference
    # start = time.time()

    # answers_etglow, perplexity_etglow = llm_kvc_inference(session_etglow, run_options_etglow, tokenizer, input_tensors_etglow, prompt_tensor, args.generate_tokens, context_len, args.sequence_length, args.window_size, args.batch)

    # session_etglow.end_profiling()
    # inf_time_etsoc = time.time() - start

    # print_llm_inference_results('ETGlow EP results', etsoc_comp_time, inf_time_etsoc, perplexity_etglow, answers_etglow,  args.batch * args.generate_tokens)

    # compare_results(perplexity_cpu, perplexity_etglow, answers_cpu, answers_etglow)

if __name__ == "__main__":
    main()
