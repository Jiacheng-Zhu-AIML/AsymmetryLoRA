import os
import math
import torch
# import click
from transformers import (
    GPTQConfig,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    RobertaForSequenceClassification)
from transformers.trainer import Trainer
from torch.utils import _pytree as pytree
from peft.tuners import lora
from peft import (
    LoraConfig,
    IA3Config,
    AdaLoraConfig,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    get_peft_model,
    prepare_model_for_kbit_training)
from peft import TaskType
from typing import List, Optional, Union, Dict, Any, cast

import sys
from pathlib import Path
upper_dir = Path(__file__).parents[1]
# print("upper_dir =", upper_dir)
sys.path.append(str(upper_dir))
from LoRASYM_peft.local_lorasym_all import LoRASYMConfig, LoRASYMModel
from LoRASYM_peft.local_peft_model_all import PeftModel_local, PeftModelForSequenceClassification_local,\
                                              PeftModelForCausalLM_local, 


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

TASKTYPE_TO_PEFTMODEL = {"CAUSAL_LM": PeftModelForCausalLM_local,
                         "SEQ_CLS": PeftModelForSequenceClassification_local}

def strategy_to_rule_dict(update_A=True, update_B=True, update_E=True,
                  A_init="V", B_init="U", E_init="E", use_E=True):
        
        para_dict = {"update_A": update_A, "update_B": update_B, "update_E": update_E,
                     "A_init": A_init, "B_init": B_init, "E_init": E_init,
                     "use_E": use_E,
                     }
        
        return para_dict


def STRATEGY_TO_RULE_func(strategy_str):
    command_list = strategy_str.split("_")
    if command_list[0][0] == "h":
        update_B_ = True
    else:
        update_B_ = False
    
    if command_list[2][0] == "h":
        update_A_ = True
    else:
        update_A_ = False
    
    return strategy_to_rule_dict(update_E=False, use_E=False,
                                update_B=update_B_, update_A=update_A_,
                                A_init=command_list[3], B_init=command_list[1])


def load_peft_model_test(model, method, strategy, model_cofig, task_type_input="SEQ_CLS"):
    
    if task_type_input not in ["CAUSAL_LM", "SEQ_CLS"]:
        print("task_type_input must be in ['CAUSAL_LM', 'SEQ_CLS']")
    
    if method == "LoRASYM":
        
        print("Using LoRASYM")
        print("model is ", type(model))
        print("strategy =", strategy)
        
        update_rule_dict = STRATEGY_TO_RULE_func(strategy)
        print("update_rule_dict =", update_rule_dict)
        
        lora_svd_config = LoRASYMConfig(
                r=model_cofig.lora_rank,    # default is 8
                lora_alpha=model_cofig.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                modules_to_save=["classifier"],
                update_rule=update_rule_dict,
                task_type=task_type_input,
                )
        lora_model = TASKTYPE_TO_PEFTMODEL[task_type_input](model, lora_svd_config)
        
        lora_model.print_trainable_parameters()
        
    else:
        print("model is ", type(model))
        print("Using", method)
        lora_config = LoraConfig(
                        r=model_cofig.lora_rank,    #  # default is 8
                        lora_alpha=model_cofig.lora_alpha,
                        lora_dropout=0.05,
                        bias="none",
                        modules_to_save=["classifier"],
                        task_type=task_type_input
                        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
    
    return lora_model


def load_peft_model_test_2(model, method, strategy, model_cofig, task_type_input="SEQ_CLS"):
    
    if task_type_input not in ["CAUSAL_LM", "SEQ_CLS"]:
        print("task_type_input must be in ['CAUSAL_LM', 'SEQ_CLS']")
    
    if method == "IA3":
        print("model is ", type(model))
        print("Using", method)
        
        ia3_config = IA3Config(
            task_type=task_type_input,
            inference_mode=False,
        )
        
        peft_model = get_peft_model(model, ia3_config)
        peft_model.print_trainable_parameters()
    
    elif method == "AdaLoRA":
        print("model is ", type(model))
        print("Using", method)
        
        adalora_config = AdaLoraConfig(
            task_type=task_type_input,
            inference_mode=False,
        )
        
        peft_model = get_peft_model(model, adalora_config)
        peft_model.print_trainable_parameters()
        
    else:
        print("model is ", type(model))
        print("Using", method)
        lora_config = LoraConfig(
                        r=model_cofig.lora_rank,    #  # default is 8
                        lora_alpha=model_cofig.lora_alpha,
                        lora_dropout=0.05,
                        bias="none",
                        modules_to_save=["classifier"],
                        task_type=task_type_input
                        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
    
    return peft_model