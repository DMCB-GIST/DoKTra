import numpy as np
import torch
import torch.nn as nn
import transformers
import logging
import csv
import os
import random

import time
import datetime

from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from sklearn.model_selection import train_test_split
import sklearn.metrics

import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict



MODEL_PATHS = {"0": './chemprot_model.bin',
               "1": './ddi_model.bin',
               "2": './ppr_model.bin'}

# MODEL_PATHS = {"0": '/NAS_Storage2/cdh/python_projects/DoKTra/training_outputs/doktra_RPl2Dxs3_chemprot_b32_l5e5_e25_cpl07_r09/1/pytorch_model.bin',
#                "1": '/NAS_Storage2/cdh/python_projects/DoKTra/training_outputs/doktra_RPl2Dxs2_ddi_b32_l2e5_e25_cpl03_r09/1/pytorch_model.bin'}

FROM_PT_DIR = 'microsoft/deberta-v3-xsmall'

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

label_list_dict = {'0' : ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"],
               '1':["DDI-mechanism", "DDI-effect", "DDI-advise", "DDI-int", "DDI-false"],
                   '2' : ["Increase", "Decrease", "Association", "Negative"]}

label_meaning_dict = {'0' : ["UPREGULATOR/ACTIVATOR", 
									"DOWNREGULATOR/INHIBITOR", 
									"AGONIST", 
									"ANTAGONIST", 
									"SUBSTRATE/PRODUCT_OF", 
									"NO_RELATION"],
               '1':["PK_MECHANISM", 
               		  "EFFECT/PD_MECHANISM", 
               		  "RECOMMENDATION/ADVISE", 
               		  "NO_RELATION", 
               		  "NO_DDI_CONTAINED"],
                   '2' : ["Increase", "Decrease", "Association", "Negative"]}


print("######################################################################")
print("####                 Relation Extraction Tool                     ####")
print("####             Please Select desired relation type              ####")
print("#### 0: chemical-protein  /  1: drug-drug  /  2: plant-phenotype  ####")
print("######################################################################")

task = input("Select :")

tasks_pool = ["chemical-protein", "drug-drug","plant-phenotype"]

assert task in ["0","1","2"], "Please select among 0 , 1 , 2"
print("\nYou've select: ", tasks_pool[int(task)])

config = transformers.AutoConfig.from_pretrained(FROM_PT_DIR, num_labels=len(label_meaning_dict[task]))

print("\n########## Initializing model ... ##########")


model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATHS[task],config=config)


model.to(device)

print("Done. \n")

tokenizer = AutoTokenizer.from_pretrained(FROM_PT_DIR)

sm = torch.nn.Softmax(dim=-1)

while True:
    print("###############################################################")
    print("####             Please enter a target sentence            ####")
    print("###############################################################")

    sentence = input("Enter :")

    # encoded_sent = tokenizer.encode(sentence, add_special_tokens = True, 
    # 								padding=True, truncation=True, max_length=128)

    # att_mask = [int(token_id > 0) for token_id in encoded_sent]

    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    model.eval()

    # t0 = time.time()

    # outputs = model(input_ids = torch.tensor([encoded_sent]),
    #                 attention_mask = [att_mask],
    #                 return_dict=True)

    with torch.no_grad():
        logits = model(**inputs).logits.cpu()
    
#     print(logits)
#     print(np.argmax(logits))
    print()
    print("Predicted relation is:",label_meaning_dict[task][np.argmax(logits)])
    print("Prediction score:", "{:.4f}".format(float(np.max(sm(logits).tolist()))))
    print()