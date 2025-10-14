#!/bin/bash
docker run --rm -it \
  --name hf_cpu_dev \
  --network host \
  --security-opt seccomp=unconfined \
  -v "$(pwd)":/workspace:cached \
  -w /workspace \
  ghcr.io/nekkoai/nutcracker-legacy:0.1 \
  /bin/bash