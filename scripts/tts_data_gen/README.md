# TTS data generation (synthetic speech for XModBench)

These scripts synthesize the spoken-audio side of XModBench's **Linguistic**
(speech recognition / translation) items from text, using
[FireRedTTS](https://github.com/FireRedTeam/FireRedTTS).

| Script | Purpose |
|--------|---------|
| `01_text2sound_english.py` | English text → speech (voice-cloned from VCTK prompts) |
| `02_text2sound_chinese.py`  | Chinese-translation text → speech |

## Setup

FireRedTTS is **not vendored** here (the upstream checkout with model
weights is ~64 GB). Install it separately:

```bash
git clone https://github.com/FireRedTeam/FireRedTTS.git
cd FireRedTTS && pip install -e .
# download the FireRedTTS pretrained weights per its README
```

## Run

```bash
# from this directory, with `fireredtts` importable and weights available
python 01_text2sound_english.py     # writes synthesized .wav for EN prompts
python 02_text2sound_chinese.py     # writes synthesized .wav for ZH prompts
```

Edit the dataset/output paths at the top of each script to point at your
local prompt corpus (VCTK-style `txt/` + `wav48_silence_trimmed/`) and the
target `Data/` location. The generated audio is what ships in the published
dataset under `Data/` on
[HF `RyanWW/XModBench`](https://huggingface.co/datasets/RyanWW/XModBench);
you only need these scripts to regenerate or extend the speech split.
