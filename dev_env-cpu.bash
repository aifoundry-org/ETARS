#!/bin/bash

xhost +SI:localuser:root
xhost +SI:localuser:$USER

docker run --rm -it \
  --name hf_cpu_dev \
  --network host \
  --security-opt seccomp=unconfined \
  -e DISPLAY=${DISPLAY} \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -v "$(pwd)":/workspace:cached \
  -w /workspace \
  ghcr.io/nekkoai/nutcracker-legacy:0.1 \
  /bin/bash