import os

'''
This script is used to train the model. 
'''

import transformers

from core.args import ModelArguments, TrainingArguments, DataArguments
from core.model import PHIMODEL

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    custom_trainer = PHIMODEL (model_args, data_args, training_args)
    custom_trainer.train()



if __name__ == "__main__":
    train()