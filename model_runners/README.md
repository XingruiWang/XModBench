# Model runners

XModBench-authored evaluation scripts for every model in the paper. Each
`<Model>/` here holds **only the XModBench-side code** (the `run.py` that
loads the benchmark, builds prompts, calls the model, and scores) — not the
upstream model weights or implementation, which are large and installed
separately.

| Model | Files | Upstream to install |
|-------|-------|---------------------|
| Qwen2.5-Omni | run.py, generate_questions.py, submit.slurm, web_demo.py | `Qwen/Qwen2.5-Omni-7B` + `qwen-omni-utils` |
| Qwen3-Omni | run.py, web_demo.py | `Qwen/Qwen3-Omni-30B-A3B-Instruct` + `qwen-omni-utils` |
| Qwen2.5-VL | run.py, generate_questions.py, submit.slurm, web_demo.py | `Qwen/Qwen2.5-VL` + `qwen-vl-utils` |
| OmniVinci | run.py, environment_setup.sh | `nvidia/omnivinci` (run `environment_setup.sh`) |
| Baichuan-Omni | (see lmms-eval port) | `baichuan-inc/Baichuan-Omni-1d5` |
| VITA | run.py, video_audio_demo.py, command.sh | `VITA-MLLM/VITA-1.5` |
| EchoInk | run.py, setup.sh | EchoInk-R1 checkpoint |
| PandaGPT | run.py | PandaGPT weights |
| AnyGPT | run.py | AnyGPT weights |
| InternVL3 | run.py | `OpenGVLab/InternVL3` |
| Gemini | run.py, run_avt.py | Google GenAI API key |
| Reka | run.py | Reka API key |
| Random | run.py | none (chance baseline) |

## How to run

```bash
export audioBench=/path/to/XModBench
python model_runners/<Model>/run.py \
    --model <model_name> \
    --task_name perception/vggss_audio_text \
    --sample 1000
```

Set the data root and unzip `Data.zip` from
[HF `RyanWW/XModBench`](https://huggingface.co/datasets/RyanWW/XModBench)
first (see the repo README / dataset card).

## Recommended: the lmms-eval port

For a turnkey, reproducible path (auto data download, interleaved-multimedia
wrappers, Level-2 metrics) use
[**XingruiWang/lmms-eval**](https://github.com/XingruiWang/lmms-eval) —
Qwen2.5-Omni reproduces the paper within |Δ|<5 on XModBench-Lite there.
These `run.py` scripts are the original per-model harness used for the paper.
