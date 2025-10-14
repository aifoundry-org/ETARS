#!/bin/bash
docker run --rm -it \
  --name hf_et_dev \
  --network host \
  --device /dev/et0_mgmt:/dev/et0_mgmt \
  --device /dev/et0_ops:/dev/et0_ops \
  --security-opt seccomp=unconfined \
  -v "$(pwd)":/workspace:cached \
  -w /workspace \
  ghcr.io/nekkoai/nutcracker-legacy:0.1 \
  /bin/bash