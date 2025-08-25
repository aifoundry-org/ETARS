#!/usr/bin/env python3
import sys, argparse
import onnx, numpy as np
from onnx import numpy_helper

def infer_shapes(model):
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape inference failed: {e}")
        return model

def collect_shapes(model):
    shapes = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        t = vi.type.tensor_type
        if not t.HasField("shape"): 
            continue
        rank = len(t.shape.dim)
        shapes[vi.name] = rank
    return shapes

def collect_constants(model):
    consts = {}
    for init in model.graph.initializer:
        try: consts[init.name] = numpy_helper.to_array(init)
        except: pass
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.HasField("t"):
                    try:
                        consts[n.output[0]] = numpy_helper.to_array(a.t)
                    except: pass
    return consts

def get_attr_axes(node):
    axes = []
    for a in node.attribute:
        if a.name == "axis" and a.HasField("i"):
            axes.append(int(a.i))
        elif a.name == "axes" and a.ints:
            axes.extend([int(x) for x in a.ints])
    return axes

def looks_like_axes(arr, rank_hint):
    if arr is None: return False
    if arr.dtype.kind not in ("i","u"): return False
    flat = np.ravel(arr)
    # Heuristic: tiny 1D/0D ints, values in a small range
    if flat.size == 0 or flat.size > max(4, (rank_hint or 4)): 
        return False
    if np.all((flat >= -16) & (flat <= 16)):
        return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Generic axis/axes checker.")
    ap.add_argument("--model", help="path to .onnx", default="/home/et/onnx-experiments/smolVLM/vision_encoder.onnx")
    ap.add_argument("--scan-const-inputs", action="store_true", default=False,
                    help="also treat small int Constant/initializer inputs as potential axes (heuristic).")
    args = ap.parse_args()

    model = onnx.load(args.model)
    model = infer_shapes(model)
    shapes = collect_shapes(model)
    consts = collect_constants(model)

    problem_nodes = []

    problems = 0
    def valid(ax, r): return (-r) <= ax <= (r-1)

    for node in model.graph.node:
        # rank of primary data input (0th)
        data = node.input[0] if node.input else ""
        rank = shapes.get(data)

        # attributes axis/axes
        attr_axes = get_attr_axes(node)
        if rank is not None and attr_axes:
            bad = [ax for ax in attr_axes if not valid(ax, rank)]
            if bad:
                problems += 1
                print(f"[axis-attr] {node.name} ({node.op_type}): axes={attr_axes} invalid for rank={rank} (bad={bad}, valid=[{-rank},{rank-1}]); data='{data}'")

                # Dirty fix just the single case of flatten operation
                node.attribute[0].i = 1
                print(f"Changed axis for {node.name} to 1")
    

    onnx.checker.check_model(model)
    onnx.save(model, args.model.split(".")[0] + "_axis." + args.model.split(".")[1])

    if problems == 0:
        print("[ok] No out-of-range axis values found (based on known ranks).")
    else:
        print(f"[done] Found {problems} potential axis issues.")

if __name__ == "__main__":
    sys.exit(main())
