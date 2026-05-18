---
license: other
library_name: transformers
---
# <span style="background: linear-gradient(45deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: bold; font-size: 1.1em;">**OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM**</span> <br />

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2510.15870)
[![Code](https://img.shields.io/badge/GitHub-Link-blue)](https://github.com/NVlabs/OmniVinci)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/nvidia/omnivinci)
[![Website](https://img.shields.io/badge/Web-Page-orange)](https://nvlabs.github.io/OmniVinci)


## Introduction
OmniVinci is an NVIDIA research project focused on exploring omni-modal LLMs that can not only see and read but also listen, speak, and reason.

We are among the best omni-modality understanding models. Check out our performance on some of the most popular omni-modality, audio, and vision benchmarks:
<p align="center">
    <img src="./asset/performance.png" width="80%"/>
<p>


## Quickstart

Below, we provide simple examples to show how to use our model with Transformers.

### Environment Setup

1. Download and navigate to the HuggingFace repository:
```
huggingface-cli download nvidia/omnivinci --local-dir ./omnivinci --local-dir-use-symlinks False
cd ./omnivinci
```

2. Install Python environment (based on NVILA codebase):
```
bash ./environment_setup.sh omnivinci
```

### 🤗 Transformers Usage

#### Video (with Audio) Inference Example
```python
from transformers import AutoProcessor, AutoModel, AutoConfig,AutoModelForCausalLM
import torch
import os

# default: Load the model on the available device(s)
model_path = "./"
video_path = "xxx.mp4"
generation_kwargs = {"max_new_tokens": 1024, "max_length": 99999999}
load_audio_in_video = True
num_video_frames = 128
audio_length = "max_3600"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(model_path,
                                  trust_remote_code=True,
                                  torch_dtype="torch.float16",
                                  device_map="auto")

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
generation_config = model.default_generation_config
generation_config.update(**generation_kwargs)

model.config.load_audio_in_video = load_audio_in_video
processor.config.load_audio_in_video = load_audio_in_video
if num_video_frames > 0:
    model.config.num_video_frames = num_video_frames
    processor.config.num_video_frames = num_video_frames
if audio_length != -1:
    model.config.audio_chunk_length = audio_length
    processor.config.audio_chunk_length = audio_length


conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "video":video_path},
            {"type": "text", "text": "Assess the video, followed by a detailed description of its video and audio contents."}
        ]
}]
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

inputs = processor([text])

output_ids = model.generate(
    input_ids=inputs.input_ids,
    media=getattr(inputs, 'media', None),
    media_config=getattr(inputs, 'media_config', None),
    generation_config=generation_config,
)
print(processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
```

- **For audio and image inference examples, please refer to `example_mini_audio.py` and `example_mini_image.py`.**


## License / Terms of Use
The model is released under the [NVIDIA OneWay Noncommercial License](asset/NVIDIA_OneWay_Noncommercial_License.docx).

## Citation
Please consider to cite our paper and this framework, if they are helpful in your research.

```bibtex
@article{ye2025omnivinci,
  title={OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM},
  author={Ye, Hanrong and Yang, Chao-Han Huck and Goel, Arushi and Huang, Wei and Zhu, Ligeng and Su, Yuanhang and Lin, Sean and Cheng, An-Chieh and Wan, Zhen and Tian, Jinchuan and others},
  journal={arXiv preprint arXiv:2510.15870},
  year={2025}
}
```