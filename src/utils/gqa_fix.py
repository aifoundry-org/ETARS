from onnx import load, save, helper
from onnx.inliner import inline_selected_functions
import onnx_graphsurgeon as gs

path = "/home/et/onnx-experiments/smolVLM/decoder_model_merged.onnx"
m = load(path)

n_fixed = 0
for n in m.graph.node:
    if n.domain == "com.microsoft" and n.op_type == "GroupQueryAttention":
        kept = []
        for a in n.attribute:
            if a.name != "softcap":  # drop unsupported attr
                kept.append(a)
            else:
                n_fixed += 1
        del n.attribute[:]
        n.attribute.extend(kept)

if n_fixed:
    # m2 = inline_selected_functions(
        # m,
        # function_ids=[("com.microsoft", "RotaryEmbedding")],  # domain, name
        # exclude=False,
        # inline_schema_functions=True,                # include schema-defined functions
    # )
    m2 = inline_selected_functions(m, function_ids=[], exclude=True, inline_schema_functions=True)
    # graph = gs.import_onnx(m)
    # graph.cleanup()
    # m2 = gs.export_onnx(graph)
    # m2.ir_version = 9
    for node in m2.graph.node:
        if node.op_type == "RotaryEmbedding":
            print(node)
    out = path.replace(".onnx", "_noscap_decomposed.onnx")
    save(m2, out)
    print(f"Removed softcap from {n_fixed} nodes -> {out}")
else:
    print("No GroupQueryAttention.softcap found.")
