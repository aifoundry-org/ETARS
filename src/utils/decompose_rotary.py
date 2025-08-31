#!/usr/bin/env python3
# decompose_rotary.py
#
# Usage:
#   python decompose_rotary.py input.onnx output.onnx
#
# What it does:
#   Finds nodes with op_type == "RotaryEmbedding" (any domain), and
#   replaces each with an equivalent subgraph using only standard ONNX ops.
#
# Assumptions:
#   - X is shaped (B, S, H, D).  (Very common in HF-style exports.)
#   - cos/sin caches broadcast against X[..., :rotary_dim//2].
#     e.g. shapes like (S, D/2) or (B, S, D/2) work via broadcasting.
#   - Attributes handled: interleaved: int (0 or 1), rotary_embedding_dim: int (0 -> use full D).
#     If missing, defaults to interleaved=0, rotary_embedding_dim=0.
#   - Optional input[3] = position_ids is ignored here (typical caches already aligned).
#
# Notes:
#   - For interleaved=1, we interleave via two ScatterElements into zeros (even/odd positions).
#   - If rotary_embedding_dim < D, we keep the tail X[..., rotary_dim:] and concat back.

import sys
from typing import Dict, List, Tuple
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx import shape_inference

def get_attr_dict(node: onnx.NodeProto) -> Dict[str, onnx.AttributeProto]:
    return {a.name: a for a in node.attribute}

def get_attr_i(attrs: Dict[str, onnx.AttributeProto], name: str, default: int) -> int:
    if name in attrs and attrs[name].type == onnx.AttributeProto.INT:
        return int(attrs[name].i)
    return default

def make_tensor(name: str, np_array: np.ndarray) -> onnx.TensorProto:
    # Ensure int64 for index-like constants by dtype
    if np_array.dtype == np.int64:
        return numpy_helper.from_array(np_array, name=name)
    if np_array.dtype == np.int32:
        return numpy_helper.from_array(np_array.astype(np.int64), name=name)
    return numpy_helper.from_array(np_array, name=name)

def make_int64_const(name: str, vals: List[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [len(vals)], np.array(vals, dtype=np.int64))

def unique_name(base: str, used: set) -> str:
    name = base
    idx = 0
    while name in used:
        idx += 1
        name = f"{base}_{idx}"
    used.add(name)
    return name

def replace_output_all_uses(graph: onnx.GraphProto, old: str, new: str):
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == old:
                node.input[i] = new
    # Also update graph outputs/ValueInfo if needed
    for vi in graph.output:
        if vi.name == old:
            vi.name = new
    for vi in graph.value_info:
        if vi.name == old:
            vi.name = new

def infer_last_dim_from_value_info(model: onnx.ModelProto, tensor_name: str) -> int:
    """
    Best-effort: after shape_inference, try to read the last dimension of tensor.
    Returns -1 if unknown.
    """
    # Collect all value_info + inputs + outputs
    def _lookup(name):
        for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
            if vi.name == name and vi.type.HasField("tensor_type"):
                tt = vi.type.tensor_type
                if tt.HasField("shape"):
                    dims = tt.shape.dim
                    if len(dims) >= 1:
                        d_last = dims[-1]
                        if d_last.HasField("dim_value"):
                            return int(d_last.dim_value)
        return -1
    return _lookup(tensor_name)

def decompose_rotary_node(model: onnx.ModelProto, node: onnx.NodeProto, used_names: set) -> List[onnx.NodeProto]:
    """
    Build a replacement subgraph for a single RotaryEmbedding node.
    Returns the list of newly created nodes (in topological order) and updates model.graph initializers as needed.
    """
    g = model.graph
    attrs = get_attr_dict(node)

    interleaved = get_attr_i(attrs, "interleaved", 0)
    rotary_dim_attr = get_attr_i(attrs, "rotary_embedding_dim", 0)  # 0 => use full head size

    X = node.input[0]
    cos = node.input[1]
    sin = node.input[2]
    pos = node.input[3] if len(node.input) >= 4 else None  # ignored in this decomposition

    out = node.output[0]

    new_nodes: List[onnx.NodeProto] = []

    # --- Determine rotary_dim (from attr or inferred last dim) ---
    head_size = infer_last_dim_from_value_info(model, X)
    if head_size <= 0:
        # Fallback: we cannot know; assume attr or require user-specified
        # If attr is 0 and we don't know head size, we must use dynamic slicing to the full axis.
        # We'll treat "rotary_dim = full D" by using a very large end value (as Slice allows overrun)
        full_end = np.array([9223372036854775807], dtype=np.int64)
        use_full = True
        rotary_dim = None
    else:
        use_full = (rotary_dim_attr == 0 or rotary_dim_attr == head_size)
        rotary_dim = head_size if rotary_dim_attr in (0, head_size) else rotary_dim_attr
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_embedding_dim must be even; got {rotary_dim}")

    # --- Slice the "rotary" part and tail on the last axis ---
    # x_rot = X[..., :rotary_dim], x_tail = X[..., rotary_dim:]
    # Slice requires starts/ends/axes/steps as inputs (INT64 tensors).
    slice_axes_name = unique_name(f"{node.name or 'rotary'}_slice_axes", used_names)
    slice_steps_name = unique_name(f"{node.name or 'rotary'}_slice_steps", used_names)
    g.initializer.extend([
        make_int64_const(slice_axes_name, [-1]),
        make_int64_const(slice_steps_name, [1]),
    ])

    starts_rot_name = unique_name(f"{node.name or 'rotary'}_starts_rot", used_names)
    ends_rot_name   = unique_name(f"{node.name or 'rotary'}_ends_rot", used_names)
    if use_full:
        # start=0, end=+inf (consume full last axis)
        g.initializer.extend([
            make_int64_const(starts_rot_name, [0]),
            make_int64_const(ends_rot_name, [9223372036854775807]),
        ])
    else:
        g.initializer.extend([
            make_int64_const(starts_rot_name, [0]),
            make_int64_const(ends_rot_name, [rotary_dim]),
        ])

    x_rot = unique_name(f"{node.name or 'rotary'}_x_rot", used_names)
    new_nodes.append(helper.make_node(
        "Slice",
        inputs=[X, starts_rot_name, ends_rot_name, slice_axes_name, slice_steps_name],
        outputs=[x_rot],
        name=unique_name(f"{node.name or 'rotary'}_slice_rot", used_names)
    ))

    # Tail only if partial
    have_tail = (not use_full) and (rotary_dim is not None) and (rotary_dim < head_size)
    x_tail = None
    if have_tail:
        starts_tail_name = unique_name(f"{node.name or 'rotary'}_starts_tail", used_names)
        ends_tail_name   = unique_name(f"{node.name or 'rotary'}_ends_tail", used_names)
        g.initializer.extend([
            make_int64_const(starts_tail_name, [rotary_dim]),
            make_int64_const(ends_tail_name, [9223372036854775807]),
        ])
        x_tail = unique_name(f"{node.name or 'rotary'}_x_tail", used_names)
        new_nodes.append(helper.make_node(
            "Slice",
            inputs=[X, starts_tail_name, ends_tail_name, slice_axes_name, slice_steps_name],
            outputs=[x_tail],
            name=unique_name(f"{node.name or 'rotary'}_slice_tail", used_names)
        ))

    # --- Split/interleave the rotary half-channels ---
    # We want x1 = even channels, x2 = odd channels if interleaved=1,
    # else x1, x2 are the first/second halves.
    if use_full and head_size <= 0:
        # We cannot know dims to build a Split. Use even/odd slicing, which doesn't need rotary_dim.
        interleaved_mode = 1
        D2 = None
    else:
        interleaved_mode = interleaved
        D2 = (head_size if rotary_dim is None else rotary_dim) // 2

    if interleaved_mode == 0:
        # x1, x2 = Split(x_rot, axis=-1, split=[D2, D2])
        split_node = helper.make_node(
            "Split",
            inputs=[x_rot],
            outputs=[
                unique_name(f"{node.name or 'rotary'}_x1", used_names),
                unique_name(f"{node.name or 'rotary'}_x2", used_names),
            ],
            name=unique_name(f"{node.name or 'rotary'}_split", used_names),
            axis=-1
            # axis=-1,
            # split=[D2, D2] if D2 is not None else None
        )
        new_nodes.append(split_node)
        x1, x2 = split_node.output
    else:
        # Even/odd via Slice steps=2
        # x1 = x_rot[..., 0::2], x2 = x_rot[..., 1::2]
        starts_even_name = unique_name(f"{node.name or 'rotary'}_starts_even", used_names)
        starts_odd_name  = unique_name(f"{node.name or 'rotary'}_starts_odd", used_names)
        ends_big_name    = unique_name(f"{node.name or 'rotary'}_ends_big", used_names)
        steps_2_name     = unique_name(f"{node.name or 'rotary'}_steps2", used_names)
        g.initializer.extend([
            make_int64_const(starts_even_name, [0]),
            make_int64_const(starts_odd_name, [1]),
            make_int64_const(ends_big_name, [9223372036854775807]),
            make_int64_const(steps_2_name, [2]),
        ])
        x1 = unique_name(f"{node.name or 'rotary'}_x1", used_names)
        x2 = unique_name(f"{node.name or 'rotary'}_x2", used_names)
        new_nodes.append(helper.make_node(
            "Slice",
            inputs=[x_rot, starts_even_name, ends_big_name, slice_axes_name, steps_2_name],
            outputs=[x1],
            name=unique_name(f"{node.name or 'rotary'}_slice_even", used_names)
        ))
        new_nodes.append(helper.make_node(
            "Slice",
            inputs=[x_rot, starts_odd_name, ends_big_name, slice_axes_name, steps_2_name],
            outputs=[x2],
            name=unique_name(f"{node.name or 'rotary'}_slice_odd", used_names)
        ))

    # --- Apply rotation:
    # real = cos * x1 - sin * x2
    # imag = sin * x1 + cos * x2
    mul_cx = unique_name(f"{node.name or 'rotary'}_mul_cx", used_names)
    mul_sy = unique_name(f"{node.name or 'rotary'}_mul_sy", used_names)
    mul_sx = unique_name(f"{node.name or 'rotary'}_mul_sx", used_names)
    mul_cy = unique_name(f"{node.name or 'rotary'}_mul_cy", used_names)
    real   = unique_name(f"{node.name or 'rotary'}_real", used_names)
    imag   = unique_name(f"{node.name or 'rotary'}_imag", used_names)

    new_nodes.extend([
        helper.make_node("Mul", inputs=[cos, x1], outputs=[mul_cx], name=unique_name(f"{node.name or 'rotary'}_mul1", used_names)),
        helper.make_node("Mul", inputs=[sin, x2], outputs=[mul_sy], name=unique_name(f"{node.name or 'rotary'}_mul2", used_names)),
        helper.make_node("Sub", inputs=[mul_cx, mul_sy], outputs=[real], name=unique_name(f"{node.name or 'rotary'}_sub", used_names)),
        helper.make_node("Mul", inputs=[sin, x1], outputs=[mul_sx], name=unique_name(f"{node.name or 'rotary'}_mul3", used_names)),
        helper.make_node("Mul", inputs=[cos, x2], outputs=[mul_cy], name=unique_name(f"{node.name or 'rotary'}_mul4", used_names)),
        helper.make_node("Add", inputs=[mul_sx, mul_cy], outputs=[imag], name=unique_name(f"{node.name or 'rotary'}_add", used_names)),
    ])

    # --- Re-interleave or simple concat back to rotary_dim
    if interleaved_mode == 0:
        rotated = unique_name(f"{node.name or 'rotary'}_rotated", used_names)
        new_nodes.append(helper.make_node(
            "Concat", inputs=[real, imag], outputs=[rotated],
            name=unique_name(f"{node.name or 'rotary'}_concat_rot", used_names),
            axis=-1
        ))
    else:
        # Build indices for even/odd positions along the last axis and scatter
        # x_out = zeros_like(x_rot); place real at even idx, imag at odd idx
        shape_xrot = unique_name(f"{node.name or 'rotary'}_shape_xrot", used_names)
        zeros_shape = unique_name(f"{node.name or 'rotary'}_zeros_shape", used_names)
        zeros = unique_name(f"{node.name or 'rotary'}_zeros", used_names)

        new_nodes.append(helper.make_node("Shape", inputs=[x_rot], outputs=[shape_xrot],
                                          name=unique_name(f"{node.name or 'rotary'}_shape", used_names)))
        # ConstantOfShape wants 1D shape tensor
        const_zero = helper.make_tensor(name=unique_name(f"{node.name or 'rotary'}_zero_val", used_names),
                                        data_type=TensorProto.FLOAT, dims=[1], vals=[0.0])
        g.initializer.extend([const_zero])
        new_nodes.append(helper.make_node("ConstantOfShape", inputs=[shape_xrot], outputs=[zeros],
                                          name=unique_name(f"{node.name or 'rotary'}_zeros_of_shape", used_names),
                                          value=const_zero))

        # Prepare even/odd indices (1D) for last axis
        if D2 is None:
            raise RuntimeError("Cannot build interleaved indices without knowing rotary_dim/2. "
                               "Please export shapes or set rotary_embedding_dim.")
        even_idx_1d = np.arange(0, 2 * D2, 2, dtype=np.int64)   # [0,2,4,...]
        odd_idx_1d  = np.arange(1, 2 * D2, 2, dtype=np.int64)   # [1,3,5,...]

        even_idx_name = unique_name(f"{node.name or 'rotary'}_even_idx_1d", used_names)
        odd_idx_name  = unique_name(f"{node.name or 'rotary'}_odd_idx_1d", used_names)
        g.initializer.extend([
            make_tensor(even_idx_name, even_idx_1d),
            make_tensor(odd_idx_name, odd_idx_1d),
        ])

        # Expand these to the same shape as updates (x1/x2): (B, S, H, D2)
        shape_x1 = unique_name(f"{node.name or 'rotary'}_shape_x1", used_names)
        expand_even_shape = unique_name(f"{node.name or 'rotary'}_expand_even_shape", used_names)
        expand_odd_shape  = unique_name(f"{node.name or 'rotary'}_expand_odd_shape", used_names)
        even_idx_exp = unique_name(f"{node.name or 'rotary'}_even_idx", used_names)
        odd_idx_exp  = unique_name(f"{node.name or 'rotary'}_odd_idx", used_names)

        new_nodes.append(helper.make_node("Shape", inputs=[x1], outputs=[shape_x1],
                                          name=unique_name(f"{node.name or 'rotary'}_shape_x1", used_names)))
        # Expand indices: onnx.Expand takes (input, shape)
        new_nodes.append(helper.make_node("Expand", inputs=[even_idx_name, shape_x1], outputs=[even_idx_exp],
                                          name=unique_name(f"{node.name or 'rotary'}_expand_even", used_names)))
        new_nodes.append(helper.make_node("Expand", inputs=[odd_idx_name, shape_x1], outputs=[odd_idx_exp],
                                          name=unique_name(f"{node.name or 'rotary'}_expand_odd", used_names)))

        # Scatter real to even positions, imag to odd positions along last axis
        scatter1 = unique_name(f"{node.name or 'rotary'}_scatter1", used_names)
        rotated  = unique_name(f"{node.name or 'rotary'}_rotated", used_names)
        new_nodes.append(helper.make_node(
            "ScatterElements",
            inputs=[zeros, even_idx_exp, real],
            outputs=[scatter1],
            name=unique_name(f"{node.name or 'rotary'}_scatter_even", used_names),
            axis=-1
        ))
        new_nodes.append(helper.make_node(
            "ScatterElements",
            inputs=[scatter1, odd_idx_exp, imag],
            outputs=[rotated],
            name=unique_name(f"{node.name or 'rotary'}_scatter_odd", used_names),
            axis=-1
        ))

    # --- Attach tail if present, else rotated is final
    final_out = rotated
    if have_tail and x_tail is not None:
        final_out = unique_name(f"{node.name or 'rotary'}_final", used_names)
        new_nodes.append(helper.make_node(
            "Concat", inputs=[rotated, x_tail], outputs=[final_out],
            name=unique_name(f"{node.name or 'rotary'}_concat_tail", used_names),
            axis=-1
        ))

    # Wire outputs
    replace_output_all_uses(g, out, final_out)

    return new_nodes

def main():
    # if len(sys.argv) != 3:
    #     print("Usage: python decompose_rotary.py input.onnx output.onnx")
    #     sys.exit(1)

    inp = "/home/et/onnx-experiments/smolVLM/decoder_model_merged_noscap.onnx"
    out = "/home/et/onnx-experiments/smolVLM/decoder_model_merged_noscap_decomposed2.onnx"

    # inp, out = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)

    # First pass: infer shapes (best effort) so we can pick rotary_dim when possible
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape inference failed: {e}")

    # Collect names to avoid collisions
    used_names = {i.name for i in model.graph.initializer}
    used_names.update({vi.name for vi in model.graph.input})
    used_names.update({vi.name for vi in model.graph.output})
    used_names.update({vi.name for vi in model.graph.value_info})
    for n in model.graph.node:
        if n.name:
            used_names.add(n.name)
        for o in n.output:
            used_names.add(o)

    # Find rotary nodes
    rotary_idx: List[int] = []
    for i, n in enumerate(model.graph.node):
        if n.op_type == "RotaryEmbedding":
            rotary_idx.append(i)

    if not rotary_idx:
        print("No RotaryEmbedding nodes found; writing original model unchanged.")
        onnx.save(model, out)
        return

    # We'll rebuild node list with replacements appended near the original position
    new_node_list: List[onnx.NodeProto] = []
    for i, n in enumerate(model.graph.node):
        if i not in rotary_idx:
            new_node_list.append(n)
            continue

        print(f"Decomposing RotaryEmbedding node: {n.name or '(unnamed)'}")
        # Build replacement subgraph
        repl_nodes = decompose_rotary_node(model, n, used_names)
        # Append replacement nodes instead of the rotary node
        new_node_list.extend(repl_nodes)

    # Replace graph nodes and re-infer shapes
    del model.graph.node[:]
    model.graph.node.extend(new_node_list)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape inference (post) failed: {e}")

    # onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f"Saved decomposed model to: {out}")

if __name__ == "__main__":
    main()
