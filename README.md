# WIP for setting up smolVLA inference 

## To parially reproduce smolVLM

1. Enter the dev environment
```bash
./dev-env.bash
```

2. Install python virtual environment
```bash
apt update && apt install python3.10-venv
```

3. Setup virtual environment
```bash
python3 -m venv venv
```

4. Activate virtual environment
```bash
source venv/bin/activate
```

5. Install python dependencies
```bash
pip3 install -r requirements.txt
```

6. Execute the embedding generation model
```bash
python3 src/inference/partial_forward/embed_forward.py
```

The output should look like that
```
<onnxruntime.capi.onnxruntime_pybind11_state.SessionOptions object at 0x7fbd44948af0>
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0826 22:22:52.897938   947 etglow_execution_provider.cc:187] Adding thread id mapping: [140452031623808->947]
[[[-0.02941895 -0.02001953 -0.03613281 ...  0.08203125  0.00059128
   -0.125     ]
  [ 0.02172852  0.11132812  0.07666016 ...  0.05297852 -0.02514648
    0.16894531]
  [-0.06689453 -0.03857422 -0.04003906 ...  0.11328125  0.10205078
    0.09716797]
  ...
  [-0.03063965  0.02758789  0.04296875 ... -0.01501465  0.11328125
    0.11914062]
  [-0.03881836 -0.00250244  0.10253906 ...  0.07617188 -0.08447266
   -0.16113281]
  [-0.06689453 -0.03857422 -0.04003906 ...  0.11328125  0.10205078
    0.09716797]]]
I0826 22:22:53.429090   947 HostManager.cpp:327] Destroying host manager...
EtGlow provider attemped to access Glow but was already destroyed
```

7. Execute visual encoder model
```bash
python3 src/inference/partial_forward/visual_forward.py
```

This result at the end
```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : [ETGLOW EP] Glow API error: {compilation failed } compiling onnx model
2025-08-26 22:27:26.680369580 [V:onnxruntime:Default, etglow_execution_provider.cc:244 ~EtGlowExecutionProvider] [ETGLOW] lifetime: 16748 ms
```