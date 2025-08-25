import argparse, os, onnx

def main():
    ap = argparse.ArgumentParser(description="Save ONNX with external weights (graph + single weights file).")
    ap.add_argument("--in", dest="src", required=True, help="Input .onnx")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--out-name", default=None, help="Output ONNX name (default: <src>_ext.onnx)")
    ap.add_argument("--weights-name", default="weights.bin", help="Weights file name (default: weights.bin)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = args.out_name or (os.path.splitext(os.path.basename(args.src))[0] + "_ext.onnx")
    out_path = os.path.join(args.out_dir, base)

    model = onnx.load(args.src)
    onnx.save_model(
        model,
        out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.weights_name,  # relative to the ONNX file
        size_threshold=0,            # force ALL tensors external
        convert_attribute=True       # also move Constant attributes out
    )
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Weights: {os.path.join(args.out_dir, args.weights_name)}")

if __name__ == "__main__":
    main()