# kev README serving benchmark on TensorRT and Jetson

Procedure: kev `scripts/serving_bench.py` (imported, not re-written) — model time per request as `kev.serve.Server`
reports it (`latency_ms`), median of 20 after two warm-ups; **new state / same state again** (prefix-cache hit).

## Latency (ms, new / cached)

| device | model | backend | 2 questions, short state | 6 questions, short state | 5 questions, 370-token state | 5 questions, 2,200-token state |
|---|---|---|---|---|---|---|
| Jetson Orin Nano 8 GB | kev-0.8b | PyTorch bf16 (kev.serve) | error | error | error | error |
| Jetson Orin Nano 8 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16, raw-engine runtime | 766.2 / 513.3 | 767.0 / 514.7 | 767.2 / 514.8 | n/a (too long for program) |
| Jetson AGX Orin 32 GB | kev-0.8b | PyTorch bf16 (kev.serve) | 245.9 / 126.1 | 250.8 / 130.8 | 250.0 / 129.9 | 448.4 / 133.2 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16 | 235.4 / 155.4 | 235.6 / 156.4 | 235.5 / 156.4 | n/a (too long for program) |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=2240, fp32 | 1342.2 / 444.0 | 1342.0 / 446.4 | 1341.4 / 446.2 | 1342.7 / 446.8 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, fp32 | 561.9 / 392.6 | 562.0 / 393.8 | 562.1 / 393.6 | n/a (too long for program) |
| Jetson AGX Orin 32 GB | kev-4b | PyTorch bf16 (kev.serve) | 315.9 / 162.1 | 421.2 / 266.8 | 453.0 / 233.8 | 1511.5 / 288.1 |
| Jetson AGX Orin 32 GB | kev-9b | PyTorch bf16 (kev.serve), LoRA unmerged | 472.3 / 240.3 | 724.4 / 492.8 | 811.7 / 411.6 | 2798.8 / 466.7 |
| RTX 6000 Ada (server3) | kev-0.8b | TensorRT 11.3.0.99 dynamic, fp32 | 46.1 / 11.2 | 71.7 / 36.1 | 81.8 / 31.2 | 234.6 / 35.5 |
| RTX 6000 Ada (server3) | kev-0.8b | PyTorch bf16 (kev.serve) | 94.3 / 48.5 | 105.1 / 59.9 | 128.8 / 57.7 | 278.6 / 60.4 |

## Throughput (requests/s at 1 / 8 / 32 / 64 concurrent clients; p50 / p99 ms at 64)

| device | model | backend | traffic | req/s @1 / 8 / 32 / 64 | p50 / p99 @64 | rejected |
|---|---|---|---|---|---|---|
| Jetson Orin Nano 8 GB | kev-0.8b | PyTorch bf16 (kev.serve) | 5 questions, 2,200-token state | error: RuntimeError: Error in dlopen: /experiment/zexin/venvs/kev-t | | |
| Jetson Orin Nano 8 GB | kev-0.8b | PyTorch bf16 (kev.serve) | 6 questions, new short state | error: RuntimeError: Error in dlopen: /experiment/zexin/venvs/kev-t | | |
| Jetson Orin Nano 8 GB | kev-0.8b | PyTorch bf16 (kev.serve) | decision-v7 development | error: RuntimeError: Error in dlopen: /experiment/zexin/venvs/kev-t | | |
| Jetson Orin Nano 8 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16, raw-engine runtime | 5 questions, 2,200-token state | n/a | | |
| Jetson Orin Nano 8 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16, raw-engine runtime | 6 questions, new short state | 1.29 / 1.30 / 1.30 / 1.30 | 49318.1 / 49347.1 | 0 |
| Jetson Orin Nano 8 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16, raw-engine runtime | decision-v7 development | 1.30 / 1.30 / 1.31 / 1.30 | 49104.6 / 62882.0 | 24 |
| Jetson AGX Orin 32 GB | kev-0.8b | PyTorch bf16 (kev.serve) | 5 questions, 2,200-token state | 2.20 / 1.30 / 0.65 / 0.62 | 102527.8 / 102613.6 | 0 |
| Jetson AGX Orin 32 GB | kev-0.8b | PyTorch bf16 (kev.serve) | 6 questions, new short state | 3.91 / 4.78 / 6.48 / 6.95 | 9203.1 / 9240.2 | 0 |
| Jetson AGX Orin 32 GB | kev-0.8b | PyTorch bf16 (kev.serve) | decision-v7 development | 4.02 / 4.99 / 6.88 / 7.27 | 8625.2 / 9067.1 | 0 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16 | 5 questions, 2,200-token state | n/a | | |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16 | 6 questions, new short state | 4.18 / 4.22 / 4.22 / 4.21 | 15203.3 / 15265.5 | 0 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16 | decision-v7 development | 3.06 / 3.07 / 3.07 / 3.06 | 19893.9 / 22811.8 | 0 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=2240, fp32 | 5 questions, 2,200-token state | 0.74 / 0.75 / 0.75 / 0.77 | 82951.7 / 83040.8 | 0 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, fp32 | 5 questions, 2,200-token state | n/a | | |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, fp32 | 6 questions, new short state | 1.77 / 1.78 / 1.77 / 1.78 | 36039.6 / 36115.4 | 0 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, fp32 | decision-v7 development | 1.23 / 1.23 / 1.23 / 1.23 | 49267.1 / 57514.3 | 0 |
| Jetson AGX Orin 32 GB | kev-4b | PyTorch bf16 (kev.serve) | 5 questions, 2,200-token state | 0.66 / 0.29 / 0.17 / 0.17 | 387471.3 / 387562.8 | 0 |
| Jetson AGX Orin 32 GB | kev-4b | PyTorch bf16 (kev.serve) | 6 questions, new short state | 2.36 / 2.44 / 2.53 / 2.54 | 25140.6 / 25183.4 | 0 |
| Jetson AGX Orin 32 GB | kev-4b | PyTorch bf16 (kev.serve) | decision-v7 development | 3.00 / 3.62 / 4.69 / 4.89 | 12784.9 / 14449.6 | 0 |
| Jetson AGX Orin 32 GB | kev-9b | PyTorch bf16 (kev.serve), LoRA unmerged | 5 questions, 2,200-token state | 0.35 / 0.16 / 0.09 / 0.09 | 717079.4 / 717171.4 | 0 |
| Jetson AGX Orin 32 GB | kev-9b | PyTorch bf16 (kev.serve), LoRA unmerged | 6 questions, new short state | 1.37 / 1.37 / 1.37 / 1.35 | 47080.2 / 48157.2 | 0 |
| Jetson AGX Orin 32 GB | kev-9b | PyTorch bf16 (kev.serve), LoRA unmerged | decision-v7 development | 1.88 / 2.23 / 2.77 / 2.94 | 21166.4 / 25957.6 | 0 |
| RTX 6000 Ada (server3) | kev-0.8b | TensorRT 11.3.0.99 dynamic, fp32 | 5 questions, 2,200-token state | 3.76 / 3.77 / 3.75 / 3.80 | 16790.4 / 16855.6 | 0 |
| RTX 6000 Ada (server3) | kev-0.8b | TensorRT 11.3.0.99 dynamic, fp32 | 6 questions, new short state | 12.98 / 13.27 / 13.14 / 13.09 | 4879.2 / 4944.9 | 0 |
| RTX 6000 Ada (server3) | kev-0.8b | TensorRT 11.3.0.99 dynamic, fp32 | decision-v7 development | 17.52 / 17.84 / 17.91 / 17.92 | 3550.7 / 3589.7 | 24 |
| RTX 6000 Ada (server3) | kev-0.8b | PyTorch bf16 (kev.serve) | 5 questions, 2,200-token state | 3.57 / 2.98 / 1.42 / 1.40 | 45646.9 / 45783.5 | 0 |
| RTX 6000 Ada (server3) | kev-0.8b | PyTorch bf16 (kev.serve) | 6 questions, new short state | 9.03 / 10.56 / 13.05 / 13.27 | 4807.8 / 4884.8 | 0 |
| RTX 6000 Ada (server3) | kev-0.8b | PyTorch bf16 (kev.serve) | decision-v7 development | 9.52 / 10.51 / 14.63 / 14.72 | 4149.1 / 7545.2 | 0 |

## Energy per request (J, module rail on Jetson)

| device | model | backend | 2 questions, short state | 6 questions, short state | 5 questions, 370-token state | 5 questions, 2,200-token state |
|---|---|---|---|---|---|---|
| Jetson Orin Nano 8 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16, raw-engine runtime | 8.165 | 8.123 | 8.208 | — |
| Jetson AGX Orin 32 GB | kev-0.8b | PyTorch bf16 (kev.serve) | 1.336 | 1.558 | 1.663 | 3.014 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, mixed fp16 | 1.922 | 1.969 | 1.986 | — |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=2240, fp32 | 8.218 | 8.102 | 8.300 | 8.346 |
| Jetson AGX Orin 32 GB | kev-0.8b | TensorRT 10.3.0 static P=384, fp32 | 4.206 | 4.229 | 4.317 | — |
| Jetson AGX Orin 32 GB | kev-4b | PyTorch bf16 (kev.serve) | 2.175 | 3.696 | 3.873 | 10.525 |
| Jetson AGX Orin 32 GB | kev-9b | PyTorch bf16 (kev.serve), LoRA unmerged | 3.502 | 6.587 | 7.080 | 20.197 |

Rails: Orin Nano `VDD_IN` (whole-module input); Orin AGX `VIN_SYS_5V0` (5 V system rail; the GPU/SoC and CPU
rails are in each JSON). Different scopes, so compare energy within a board, not across boards. The desktop GPU
(server3) is shared and `nvidia-smi` sees card power only; its energy is not reported.
