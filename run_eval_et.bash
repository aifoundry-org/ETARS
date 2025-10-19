#!/bin/bash
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

python -m src.inference.eval --device ET