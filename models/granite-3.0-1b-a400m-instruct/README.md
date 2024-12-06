---
pipeline_tag: text-generation
inference: false
license: apache-2.0
library_name: transformers
tags:
- language
- granite-3.0
model-index:
- name: granite-3.0-2b-instruct
  results:
  - task:
      type: text-generation
    dataset:
      type: instruction-following
      name: IFEval
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.39
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: instruction-following
      name: MT-Bench
    metrics:
    - name: pass@1
      type: pass@1
      value: 6.17
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: human-exams
      name: AGI-Eval
    metrics:
    - name: pass@1
      type: pass@1
      value: 20.35
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: human-exams
      name: MMLU
    metrics:
    - name: pass@1
      type: pass@1
      value: 32
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: human-exams
      name: MMLU-Pro
    metrics:
    - name: pass@1
      type: pass@1
      value: 12.21
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: OBQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 38.4
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: SIQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 47.55
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: Hellaswag
    metrics:
    - name: pass@1
      type: pass@1
      value: 65.59
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: WinoGrande
    metrics:
    - name: pass@1
      type: pass@1
      value: 61.17
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: TruthfulQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 49.11
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reading-comprehension
      name: BoolQ
    metrics:
    - name: pass@1
      type: pass@1
      value: 70.12
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reading-comprehension
      name: SQuAD 2.0
    metrics:
    - name: pass@1
      type: pass@1
      value: 1.27
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reasoning
      name: ARC-C
    metrics:
    - name: pass@1
      type: pass@1
      value: 41.21
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reasoning
      name: GPQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 23.07
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reasoning
      name: BBH
    metrics:
    - name: pass@1
      type: pass@1
      value: 31.77
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: code
      name: HumanEvalSynthesis
    metrics:
    - name: pass@1
      type: pass@1
      value: 30.18
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: code
      name: HumanEvalExplain
    metrics:
    - name: pass@1
      type: pass@1
      value: 26.22
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: code
      name: HumanEvalFix
    metrics:
    - name: pass@1
      type: pass@1
      value: 21.95
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: code
      name: MBPP
    metrics:
    - name: pass@1
      type: pass@1
      value: 15.4
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: math
      name: GSM8K
    metrics:
    - name: pass@1
      type: pass@1
      value: 26.31
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: math
      name: MATH
    metrics:
    - name: pass@1
      type: pass@1
      value: 10.88
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: multilingual
      name: PAWS-X (7 langs)
    metrics:
    - name: pass@1
      type: pass@1
      value: 45.84
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: multilingual
      name: MGSM (6 langs)
    metrics:
    - name: pass@1
      type: pass@1
      value: 11.8
      veriefied: false
base_model:
- ibm-granite/granite-3.0-1b-a400m-base
---

<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/62cd5057674cdb524450093d/1hzxoPwqkBJXshKVVe6_9.png) -->
<!-- ![image/png](granite-3_0-language-models_Group_1.png)
 -->
# Granite-3.0-1B-A400M-Instruct

**Model Summary:**
Granite-3.0-1B-A400M-Instruct is an 1B parameter model finetuned from *Granite-3.0-1B-A400M-Base* using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets. This model is developed using a diverse set of techniques with a structured chat format, including supervised finetuning, model alignment using reinforcement learning, and model merging.

- **Developers:** Granite Team, IBM
- **GitHub Repository:** [ibm-granite/granite-3.0-language-models](https://github.com/ibm-granite/granite-3.0-language-models)
- **Website**: [Granite Docs](https://www.ibm.com/granite/docs/)
- **Paper:** [Granite 3.0 Language Models](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf)
- **Release Date**: October 21st, 2024
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Languages:** 
English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. Users may finetune Granite 3.0 models for languages beyond these 12 languages.

**Intended use:** 
The model is designed to respond to general instructions and can be used to build AI assistants for multiple domains, including business applications.

*Capabilities*
* Summarization
* Text classification
* Text extraction
* Question-answering
* Retrieval Augmented Generation (RAG)
* Code related tasks
* Function-calling tasks
* Multilingual dialog use cases

**Generation:** 
This is a simple example of how to use Granite-3.0-1B-A400M-Instruct model.

Install the following libraries:

```shell
pip install torch torchvision torchaudio
pip install accelerate
pip install transformers
```
Then, copy the snippet from the section that is relevant for your use case.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "auto"
model_path = "ibm-granite/granite-3.0-1b-a400m-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()
# change input text as desired
chat = [
    { "role": "user", "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location." },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, 
                        max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output)
```

**Model Architecture:** 
Granite-3.0-1B-A400M-Instruct is based on a decoder-only sparse Mixture of Experts (MoE) transformer architecture. Core components of this architecture are: Fine-grained Experts, Dropless Token Routing, and Load Balancing Loss.

| Model                        | 2B Dense | 8B Dense | 1B MoE       | 3B MoE   |
| :--------                    | :--------| :--------| :--------    |:-------- |
| Embedding size               | 2048     | 4096     | **1024**     | 1536     |
| Number of layers             | 40       | 40       | **24**       | 32       |
| Attention head size          | 64       | 128      | **64**       | 64       |
| Number of attention heads    | 32       | 32       | **16**       | 24       |
| Number of KV heads           | 8        | 8        | **8**        | 8        |
| MLP hidden size              | 8192     | 12800    | **512**      | 512      |
| MLP activation               | SwiGLU   | SwiGLU   | **SwiGLU**   | SwiGLU   |
| Number of Experts            | —        | —        | **32**       | 40       |
| MoE TopK                     | —        | —        | **8**        | 8        |
| Initialization std           | 0.1      | 0.1      | **0.1**      | 0.1      |
| Sequence Length              | 4096     | 4096     | **4096**     | 4096     |
| Position Embedding           | RoPE     | RoPE     | **RoPE**     | RoPE     |
| # Parameters                 | 2.5B     | 8.1B     | **1.3B**     | 3.3B     |
| # Active Parameters          | 2.5B     | 8.1B     | **400M**     | 800M     |
| # Training tokens            | 12T      | 12T      | **10T**      | 10T      |

**Training Data:** 
Overall, our SFT data is largely comprised of three key sources: (1) publicly available datasets with permissive license, (2) internal synthetic data targeting specific capabilities, and (3) very small amounts of human-curated data. A detailed attribution of datasets can be found in the [Granite Technical Report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) and [Accompanying Author List](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/author-ack.pdf).

**Infrastructure:**
We train Granite 3.0 Language Models using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs while minimizing environmental impact by utilizing 100% renewable energy sources.

**Ethical Considerations and Limitations:** 
Granite 3.0 Instruct Models are primarily finetuned using instruction-response pairs mostly in English, but also multilingual data covering eleven languages. Although this model can handle multilingual dialog use cases, its performance might not be similar to English tasks. In such case, introducing a small number of examples (few-shot) can help the model in generating more accurate outputs. While this model has been aligned by keeping safety in consideration, the model may in some cases produce inaccurate, biased, or unsafe responses to user prompts. So we urge the community to use this model with proper safety testing and tuning tailored for their specific tasks.

<!-- ## Citation
```
@misc{granite-models,
  author = {author 1, author2, ...},
  title = {},
  journal = {},
  volume = {},
  year = {2024},
  url = {https://arxiv.org/abs/0000.00000},
}
``` -->