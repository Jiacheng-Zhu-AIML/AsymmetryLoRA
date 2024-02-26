# Asymmetry in Low-Rank Adapters of Foundation Models

<!-- 🌟 [ArXiv Preprint](https://arxiv.org/)  -->

his repo hosts the code for the paper "Asymmetry in Low-Rank Adapters of Foundation Models". We discover and analysis the asymmetry of the LoRA adapter matrices `B` and `A`,  

## 🔗 Quick Links
- [Asymmetry in Low-Rank Adapters of Foundation Models](#asymmetry-in-low-rank-adapters-of-foundation-models)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Usage](#usage)
  - [Experiments](#experiments)
    - [GLUE benchmark](#step-1-warmup-training) 
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)


## Install Requirements
**Step 1**: Please follow the installation steps. First, make sure you have [Pytorch](https://pytorch.org/get-started/previous-versions/) installed. 
```
pip3 install torch==1.13.0 torchvision
```

**Step 2**: Then install the rest of the required packages:

```
cd AsymmetryLoRA
pip install -r requirement.txt
```

## Usage
Our LoRASYM module follows the structure of the [peft](https://github.com/huggingface/peft) module. Specifically, we provide a flexible interface to account for the initialization settings of matrices **A** and **B**: 

- **V** and **U**: Right and left singular matrices of the original weight matrix.
- **random**: Initializes with a random orthonormal matrix.
- **he**: Uses `torch.nn.init.kaiming_uniform_` for random uniform distribution, optimizing neural network layer activations.


You can customize matrices **A** and **B** with these options.

| Matrix | Options                     | Example          | Explanation                                      |
|--------|-----------------------------|------------------|--------------------------------------------------|
| **A**  | `V`, `rand`, `he`, `zero`   | `A_rand`         | **A** is intialized as random orthonormal matrix and is freezed during training.   |
| **B**  | `U`, `rand`, `he`, `zero`        | `hB_zero`        | **B** is initialized as zero and will be updated.                |

**Explaination:** `A_rand_hB_zero` means **A** is initialized as random orthonormal and unchanged, while **B** starts at zero and is being updated.

We provide a wrapper that compiles with other models from Huggingface's [transformer](https://github.com/huggingface/transformers) models. The following is an example of usage:

```python
from transformers import AutoModelForSequenceClassification
from LoRASYM_peft.local_peft_model_all import PeftModelForCausalLM_local,
from LoRASYM_peft.local_lorasym_all import LoRASYMConfig

model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
    )

update_rule_dict = para_dict = {"update_A": False, "update_B": True, 
"A_init": "rand", "B_init": "zero"}

lorasym_config = LoRASYMConfig(
                r=16,   
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                modules_to_save=["classifier"],
                update_rule=update_rule_dict,
                task_type="SEQ_CLS",
                )

lora_model = PeftModelForCausalLM_local(model, lorasym_config)

```
## GLUE benchmark

TODO

## Evaluation and analysis

TODO

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Jiacheng Zhu (zjc@mit.edu). 
Please feel free to open an issue if you encounter any problems when using the code.

<!-- ## Citation
Please cite our paper if you find the repo helpful in your work: -->
