import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.utils
import numpy as np
import sys
import argparse
import os
import logging
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Modify vicuna ONNX model to solve the Rotary embeddings problems and allow a sequence size for input_ids")
    parser.add_argument("-n", '--name-model', metavar = "str", type = str, default = 'llava-7b',
                        help = 'Name of the model')
    parser.add_argument("-p", '--precision-fp16', action="store_true", default=False, 
                        help = 'Use FP16 precision')
    parser.add_argument("-i", '--input-dir', metavar = "DIR", type = str, default = '.',
                        help = 'Path of the input ONNX model')
    parser.add_argument("-o", '--output-dir', metavar = "DIR", type = str, default = '.',
                        help = 'Path of the output ONNX model')    
    args = parser.parse_args()
    return args

def main():    
    # Paths
    args = parse_arguments()
    logging.basicConfig(level = logging.INFO)
    useFP16 = args.precision_fp16
    # original_model_path = args.input_dir
    original_model_path = "/home/et/onnx-experiments/smolVLM/vision_encoder.onnx"
    modified_model_path =  "/home/et/onnx-experiments/smolVLM/vision_encoder-kvc.onnx"

    num_layers = 12
    num_heads = 16
    hidden_cache_size = 1152


    # Model parameters
    # if "llava-7b" in args.name_model:
        # num_layers = 32
        # num_heads = 32
        # hidden_cache_size = 128
    # else:
        # sys.exit(f"Error. Model not recognized")
    if (useFP16):
        internal_type = onnx.TensorProto.FLOAT16
    else:
        internal_type = onnx.TensorProto.FLOAT


    # Load model

    model = onnx.load(original_model_path, load_external_data=True)
    nodes = model.graph.node
    inputs = model.graph.input
    outputs = model.graph.output


    # Change the dimension of the attention mask

    for tensor in inputs:
      if tensor.name == "attention_mask":
        tensor.type.tensor_type.shape.dim[-1].dim_param = "past_sequence+sequence"


    # Add new inputs and outputs (KV-caches)

    past_key_names = ["past_key_values." + str(i) + ".key" for i in range(num_layers)]
    past_value_names = ["past_key_values." + str(i) + ".value" for i in range(num_layers)]
    new_key_names = ["present." + str(i) + ".key" for i in range(num_layers)]
    new_value_names = ["present." + str(i) + ".value" for i in range(num_layers)]

    for i in range(num_layers):
      past_key = onnx.helper.make_tensor_value_info(past_key_names[i], internal_type, ["batch", num_heads, "past_sequence", hidden_cache_size])
      inputs.append(past_key)
      past_value = onnx.helper.make_tensor_value_info(past_value_names[i], internal_type, ["batch", num_heads, "past_sequence", hidden_cache_size])
      inputs.append(past_value)
      present_key = onnx.helper.make_tensor_value_info(new_key_names[i], internal_type, ["batch", num_heads, "past_sequence+sequence", hidden_cache_size])
      outputs.append(present_key)
      present_value = onnx.helper.make_tensor_value_info(new_value_names[i], internal_type, ["batch", num_heads, "past_sequence+sequence", hidden_cache_size])
      outputs.append(present_value)


    # Compute sequence lengths and token indices at the beginning

    cache_sequence_dim = "/model/qkv_sequence_dim"
    past_sequence_length = "/model/past_sequence_length"
    current_sequence_length = "/model/sequence_length"
    total_sequence_length = "/model/past_sequence_length_plus_sequence_length"
    total_token_counter = "/model/all_token_indices_inside_sequence"
    current_token_counter = "/model/current_token_indices_inside_sequence"

    new_nodes = []

    new_nodes.append(onnx.helper.make_node(
      name = "/model/Constant_one_index",
      op_type = "Constant",
      value = onnx.numpy_helper.from_array(np.array(1, np.int64)), # 0-dimensional tensor (i.e., scalar)
      inputs = [],
      outputs = ["/model/Constant_one_index_output_0"]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Constant_zero_index",
      op_type = "Constant",
      value = onnx.numpy_helper.from_array(np.array(0, np.int64)), # 0-dimensional tensor (i.e., scalar)
      inputs = [],
      outputs = ["/model/Constant_zero_index_output_0"]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Constant_zero_vector",
      op_type = "Constant",
      value = onnx.numpy_helper.from_array(np.array([0], np.int64)), # 0-dimensional tensor (i.e., scalar)
      inputs = [],
      outputs = ["/model/Constant_zero_vector_output_0"]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Constant_two_vector",
      op_type = "Constant",
      value = onnx.numpy_helper.from_array(np.array([2], np.int64)), # 1-dimensional tensor (i.e., vector)
      inputs = [],
      outputs = [cache_sequence_dim]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Shape_attention_mask",
      op_type = "Shape",
      inputs = ["attention_mask"],
      outputs = ["/model/Shape_attention_mask_output_0"]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Gather_total_sequence_length",
      op_type = "Gather",
      axis = 0,
      inputs = ["/model/Shape_attention_mask_output_0", "/model/Constant_one_index_output_0"], # [data, indices]
      outputs = [total_sequence_length]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Unsqueeze_total_sequence_length",
      op_type = "Unsqueeze",
      inputs = [total_sequence_length, "/model/Constant_zero_vector_output_0"],
      outputs = [total_sequence_length + "_1d"]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Shape_input_ids",
      op_type = "Shape",
      inputs = ["input_ids"],
      outputs = ["/model/Shape_input_ids_output_0"]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Gather_sequence_length",
      op_type = "Gather",
      axis = 0,
      inputs = ["/model/Shape_input_ids_output_0", "/model/Constant_one_index_output_0"], # [data, indices]
      outputs = [current_sequence_length]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Sub_past_sequence_length",
      op_type = "Sub",
      inputs = [total_sequence_length, current_sequence_length],
      outputs = [past_sequence_length]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Unsqueeze_past_sequence_length",
      op_type = "Unsqueeze",
      inputs = [past_sequence_length, "/model/Constant_zero_vector_output_0"],
      outputs = [past_sequence_length + "_1d"]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Range_total_token_counter",
      op_type = "Range",
      inputs = ["/model/Constant_zero_index_output_0", total_sequence_length, "/model/Constant_one_index_output_0"], # [start, limit, delta]
      outputs = [total_token_counter]
    ))
    new_nodes.append(onnx.helper.make_node(
      name = "/model/Unsqueeze_total_token_counter",
      op_type = "Unsqueeze",
      inputs = [total_token_counter, "/model/Constant_zero_vector_output_0"],
      outputs = [total_token_counter + "_2d"]
    ))


    # The graph needs to be modified for three reasons:
    #  a) To merge past and current tokens being processed
    #  b) To get the correct attention mask in matrix form
    #  c) To guarantee that the rotary embeddings used in the attention sublayers use the correct indices/dimensions
    #
    # The necessary changes are the following:
    #  1. Introduce concatenations of past cache and new elements right after those are produced (in particular, for the keys, before the rotary embeddings)
    #  2. Use `past_sequence_length + sequence_length` instead of `sequence_length` for the columns of the matrix attention mask:
    #     i) Replace `sequence_length` with `past_sequence_length + sequence_length` before the creation of a triangular matrix.
    #     ii) Take the final `sequence_length`-many rows of that triangular matrix.
    #  3. Use `past_sequence_length + sequence_length` instead of `sequence_length` to extract the rotation tensors:
    #     i) Use the new token counter instead of the old one.
    #     ii) Take the final slice of the rotation tensors before applying them to the queries.
    
    value_numbers = [4*i -1 for i in range(num_layers)]

    key_producers = ["/model/layers." + str(i) + "/self_attn/Transpose_1" for i in range(num_layers)]
    value_producers = ["/model/layers." + str(i) + "/self_attn/Transpose_2" for i in range(num_layers)]
    old_key_names = ["/model/layers." + str(i) + "/self_attn/Transpose_1_output_0" for i in range(num_layers)]
    old_value_names = ["value_states." + str(i) for i in value_numbers]
    old_value_names[0] = "value_states"

    old_total_sequence_names = ["/model/Unsqueeze_1_output_0", "/model/Unsqueeze_2_output_0", "/model/Constant_14_output_0", "/model/Constant_15_output_0"]
    triangular_matrix_producer = "/model/Expand"
    triangular_matrix_name = "/model/Expand_output_0"
    slice_matrix_name = "/model/triangular_attention"

    old_range = ["/model/Reshape"]
    old_token_counter_name = "/model/Unsqueeze_output_0" # Counter of size sequence_length and shape (1, s)
    query_even_rotary_embeddings = ["/model/layers." + str(i) + "/self_attn/Mul" for i in range(num_layers)]
    query_odd_rotary_embeddings = ["/model/layers." + str(i) + "/self_attn/Mul_1" for i in range(num_layers)]
    even_rotation_slice_names = ["even_rotation." + str(i) + ".query" for i in range(num_layers)]
    odd_rotation_slice_names = ["odd_rotation." + str(i) + ".query" for i in range(num_layers)]

    for node in nodes:
      # Replace inputs if necessary
      for input_index, tensor_name in enumerate(node.input):
        if tensor_name == old_token_counter_name:
          # This node consumes a counter of tokens inside the sequence; the KV-cache version should
          # use a counter of size past_sequence_length + sequence_length instead of just sequence_length
          node.input[input_index] = total_token_counter + "_2d"
        elif tensor_name == triangular_matrix_name:
          node.input[input_index] = slice_matrix_name
        elif tensor_name in old_total_sequence_names:
          node.input[input_index] = total_sequence_length + "_1d"
        elif tensor_name in old_key_names:
          layer_index = old_key_names.index(tensor_name)
          node.input[input_index] = new_key_names[layer_index]
        elif tensor_name in old_value_names:
          layer_index = old_value_names.index(tensor_name)
          node.input[input_index] = new_value_names[layer_index]
      # Prepend nodes if necessary
      if node.name in query_even_rotary_embeddings:
        layer_index = query_even_rotary_embeddings.index(node.name)
        new_nodes.append(onnx.helper.make_node(
          name = "slice_even_rotary_embedding." + str(layer_index),
          op_type = "Slice",
          inputs = [node.input[1], past_sequence_length + "_1d", total_sequence_length + "_1d", cache_sequence_dim], # [data, starts, ends, axes]
          outputs = [even_rotation_slice_names[layer_index]]
        ))
        node.input[1] = even_rotation_slice_names[layer_index]
      
      elif node.name in old_range:
        node.input[0] = total_token_counter
      elif node.name in query_odd_rotary_embeddings:
        layer_index = query_odd_rotary_embeddings.index(node.name)
        new_nodes.append(onnx.helper.make_node(
          name = "slice_odd_rotary_embedding." + str(layer_index),
          op_type = "Slice",
          inputs = [node.input[1], past_sequence_length + "_1d", total_sequence_length + "_1d", cache_sequence_dim], # [data, starts, ends, axes]
          outputs = [odd_rotation_slice_names[layer_index]]
        ))
        node.input[1] = odd_rotation_slice_names[layer_index]
      # Take current node
      new_nodes.append(node)
      # Append nodes if necessary
      if node.name == triangular_matrix_producer:
        new_nodes.append(onnx.helper.make_node(
          name = "slice_triangular_attention_matrix",
          op_type = "Slice",
          inputs = [triangular_matrix_name, past_sequence_length + "_1d", total_sequence_length + "_1d", cache_sequence_dim], # [data, starts, ends, axes]
          outputs = [slice_matrix_name]
        ))
      elif node.name in key_producers:
        layer_index = key_producers.index(node.name)
        new_nodes.append(onnx.helper.make_node(
          name = "concat_key." + str(layer_index),
          op_type = "Concat",
          axis = 2,
          inputs = [past_key_names[layer_index], old_key_names[layer_index]],
          outputs = [new_key_names[layer_index]]
        ))
      elif node.name in value_producers:
        layer_index = value_producers.index(node.name)
        new_nodes.append(onnx.helper.make_node(
          name = "concat_value." + str(layer_index),
          op_type = "Concat",
          axis = 2,
          inputs = [past_value_names[layer_index], old_value_names[layer_index]],
          outputs = [new_value_names[layer_index]]
        ))

    del nodes[:]
    nodes.extend(new_nodes)


    # Get rid of dangling nodes (i.e., whose results are unused)

    input_names = [tensor.name for tensor in inputs]
    output_names = [tensor.name for tensor in outputs]
    model_extractor = onnx.utils.Extractor(model)
    clean_model = model_extractor.extract_model(input_names, output_names)


    # Export modified model

    with open(modified_model_path, "wb") as file:
        file.write(clean_model.SerializeToString())
    
    for file_name in os.listdir(os.path.dirname(original_model_path)):
        if file_name == "model.onnx":
            continue
        # construct full file path
        source = os.path.join(os.path.dirname(original_model_path), file_name)
        destination = os.path.join(os.path.dirname(modified_model_path), file_name)
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
    
if __name__ == "__main__":
    main()

