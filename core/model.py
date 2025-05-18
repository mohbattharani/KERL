# # https://colab.research.google.com/drive/1QWZlqdUOZhaSA1-TC-EPSmD-oyX-hV4K?usp=sharing#scrollTo=K4KFtes4FDI1

import os
import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers

import json, random, os
from pprint import pprint
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from torch.utils.data import Sampler
from transformers.trainer import has_length
from transformers import Trainer
from peft import  PeftModel, get_peft_config

from .args import IGNORE_INDEX
from .data import make_supervised_data_module

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print ("Printing args")
        print(*args)



def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class PhiTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



class PHIMODEL ():
    def __init__(self, model_args, data_args, training_args) -> None:
        self.model_args = model_args 
        self.data_args = data_args
        self.training_args = training_args
        self.compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        self.make_model()



    def get_recent_lora_weights (self):
        folders = list(pathlib.Path(self.training_args.output_dir).glob("checkpoint-*"))
        # Extract suffix as an integer from each folder name
        suffixes = [(folder, int(folder.name.split('-')[-1])) for folder in folders]

        # Find the folder with the highest suffix
        max_suffix_folder = max(suffixes, key=lambda x: x[1])[0]

        return os.path.join (self.training_args.output_dir, max_suffix_folder.name)



    def make_model (self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.training_args.cache_dir,
            model_max_length=self.training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = self.training_args.model_max_length
        self.tokenizer.pad_token = self.tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = 'right'

        if self.training_args.bits in [4, 8]:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=self.training_args.bits == 4,
                    load_in_8bit=self.training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=self.compute_dtype,
                    bnb_4bit_use_double_quant=self.training_args.double_quant,
                    bnb_4bit_quant_type=self.training_args.quant_type # {'fp4', 'nf4'}
                )
            bnb_model_from_pretrained_args = {}
            bnb_model_from_pretrained_args.update(dict(
                #device_map={"": self.training_args.device},
                quantization_config= bnb_config,
                torch_dtype="auto",
                trust_remote_code=True, 
                attn_implementation="flash_attention_2"
            ))

            self.model = AutoModelForCausalLM.from_pretrained(self.model_args.model_name_or_path,  **bnb_model_from_pretrained_args)
            #self.model = prepare_model_for_kbit_training(self.model)    
            self.model.config.torch_dtype=(torch.float32 if self.training_args.fp16 else (torch.bfloat16 if self.training_args.bf16 else torch.float32))
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=self.training_args.gradient_checkpointing)

        else:
            kwarg = {
                "torch_dtype": "auto", 
                "trust_remote_code":True,
                "_attn_implementation":"flash_attention_2"
                }
            self.model = AutoModelForCausalLM.from_pretrained(self.model_args.model_name_or_path,  **kwarg) 

        self.model.config.use_cache = False 
            
        if self.training_args.gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


        if self.training_args.lora_enable:
            if self.training_args.bits == 16:
                if self.training_args.bf16:
                    self.model.to(torch.bfloat16)
                if self.training_args.fp16:
                    self.model.to(torch.float16)
            
            self.lora_config = LoraConfig(
                r=self.training_args.lora_r,
                lora_alpha=self.training_args.lora_alpha,
                target_modules=find_all_linear_names(self.model),
                lora_dropout=self.training_args.lora_dropout,
                bias=self.training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            rank0_print("Adding LoRA adapters...")
            if list(pathlib.Path(self.training_args.output_dir).glob("checkpoint-*")):
                adapter = self.get_recent_lora_weights()
                self.model = PeftModel.from_pretrained(self.model , adapter, is_trainable=True)  
                #self.model = self.model.merge_and_unload()  
                #self.model._mark_only_adapters_as_trainable()
                # model.is_trainable = True    
            else:
                self.model = get_peft_model(self.model, self.lora_config)
            
        if self.training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    if self.training_args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if self.training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
                            

    def train (self):
        data_module = make_supervised_data_module(tokenizer=self.tokenizer,
                                              data_args=self.data_args)
        #self.model.model.requires_grad_(False)
        #print (self.model) 
        
        trainer = Trainer(model=self.model,
                    tokenizer=self.tokenizer,
                    args=self.training_args,
                    **data_module
                    )

        torch.cuda.empty_cache()
        if list(pathlib.Path(self.training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()


