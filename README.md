# ETARS Inference Framework

This repository extends the [LeRobot](https://github.com/huggingface/lerobot) framework by introducing an **ONNXRuntime-based inference backend** for the **SmolVLA** policy model.
It enables execution on both standard CPU and custom hardware backends such as **ETSoC** through the **ETGlow** provider.

The implementation preserves LeRobot’s dataset and policy interface while adding an interchangeable runtime layer for deterministic and hardware-specific inference.

---

## Overview

LeRobot provides a unified API for robot learning and policy evaluation.
This repository focuses exclusively on inference, providing an ONNXRuntime integration layer that mirrors LeRobot’s PyTorch-based inference flow.

### Main Features

* ONNXRuntime backend compatible with LeRobot’s SmolVLA policy
* Support for CPU and ETGlow execution providers
* Reference PyTorch implementation for output comparison
* Deterministic behavior for reproducible inference
* Lightweight testing utilities to validate backend equivalence

---

## Environment Setup

1. **Enter the development environment**

   For ETSoC execution:

   ```bash
   ./dev-env-et.bash
   ```

   For CPU-only execution:

   ```bash
   ./dev-env-cpu.bash
   ```

2. **Install virtual environment tools**

   ```bash
   apt update && apt install python3.10-venv
   ```

3. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   apt-get update && apt-get install -y libosmesa6 libosmesa6-dev git cmake libglib2.0-0
   pip install --upgrade-strategy only-if-needed -r requirements.txt
   ```

This ensures a minimal, dependency-controlled Python environment suitable for inference or testing.

---

## Usage

### ONNXRuntime-based inference

```bash
python -m src.inference.smolvla --device CPU
```

or, for ETSoC hardware:

```bash
python -m src.inference.smolvla --device ET
```

If you wish to eval model in simulation, consider using eval script

```bash
python -m src.inference.eval --device CPU
```
If you have available display this command will launch a small visualization of evaluation process.
Othervise it'll save current fram to the `sim_frame.png` file and later compile a video in outputs filder.

### Torch-based (LeRobot default) inference

```bash
python -m src.inference.smolvla_torch
```

Each script loads a dataset sample and performs a single policy `sample_action` step, returning one action vector.

---

## Testing and Validation

To verify consistency between ONNXRuntime and Torch-based outputs:

```bash
PYTHONPATH=. pytest -s  src/tests/test_compare_policies.py
```

This test executes both inference paths and performs a numerical comparison of their outputs.

To run a testing simulation:

`run_eval_cpu.bash` or `run_eval_et.bash`

---

## Repository Structure

```
src/
 ├── inference/
 │    ├── eval_torch.py         # Reference Torch-based evaluation script
 │    ├── eval.py               # ONNXRuntime script for evaluation (CPU/ ET)
 │    ├── smolvla_torch.py      # Reference Torch-based inference
 │    └── smolvla.py            # ONNXRuntime inference entrypoint (CPU / ET)
 └── tests/
      └── test_compare_policies.py  # Cross-backend consistency test
```

---

## Notes

* This repository is an extension, not a fork, of [LeRobot](https://github.com/huggingface/lerobot).
* The primary purpose is backend experimentation and ONNXRuntime integration.
* Designed for controlled inference evaluation rather than model training.
* Deterministic setup is provided to ensure reproducible CPU runs.

---