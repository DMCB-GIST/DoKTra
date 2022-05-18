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

from sklearn.model_selection import train_test_split
import sklearn.metrics

import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir',default=None,type=str)
parser.add_argument('--task',default=None,type=str)

parser.add_argument('--vocab_path',default=None,type=str)
parser.add_argument('--vocab_path_teacher',default=None,type=str)

parser.add_argument('--init_checkpoint',default=None,type=str)
parser.add_argument('--init_checkpoint_teacher',default=None,type=str)

parser.add_argument('--config_file',default=None,type=str)
parser.add_argument('--config_file_teacher',default=None,type=str)

parser.add_argument('--from_pt_dir',default=None,type=str)

parser.add_argument('--output_dir',default=None,type=str)

parser.add_argument('--max_seq_len',default=256,type=int)
parser.add_argument('--batch_size',default=10,type=int)
parser.add_argument('--epochs',default=5,type=int)
parser.add_argument('--lr',default=2e-5,type=float)
parser.add_argument('--random_seed',default=42,type=int)
parser.add_argument('--margin',default=1.0,type=float)
parser.add_argument('--loss_switch_rate',default=0.7,type=float)
parser.add_argument('--confidence_penalty_weight',default=0.3,type=float)

parser.add_argument('--do_train',default=False,type=bool)
parser.add_argument('--do_eval',default=False,type=bool)
parser.add_argument('--do_predict',default=False,type=bool)

args = parser.parse_args()

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


assert args.task in ['chemprot','gad','ddi','i2b2','hoc'] , "The task is not supported"


print("########## SHOW ARGUMENT CONFIGURATIONS ##########")
print("TASK NAME : ",args.task)
print("MAX SEQ LEN : ",args.max_seq_len)
print("BATCH SIZE : ",args.batch_size)
print("EPOCHS  : ",args.epochs)
print("LEARNING RATE  : ",args.lr)
print("RANDOM SEED  : ",args.random_seed)
print("BETA  : ",args.confidence_penalty_weight)
print("GAMMA  : ",args.loss_switch_rate)
print("########## END OF ARGUMENT CONFIGURATIONS ##########")


def dataset_dict_loader(filedir, label_list=None,
                        train=True,trn_header=False,
                        dev=True,dev_header=False,
                        test=True,tst_header=True,
                        is_hoc=False):
    dataset_dict = {}
    dataset_types = ['train','dev','test']
    dataset_do = [train,dev,test]
    dataset_header = [trn_header,dev_header,tst_header]
    
    (idx_sen,idx_lab) = (0,1)
    if is_hoc:
        (idx_sen,idx_lab) = (0,-1)
    
    for d_type,do,header in zip(dataset_types,dataset_do,dataset_header):
        if do:
            with open(os.path.join(filedir,d_type+'.tsv'),'r',encoding='utf-8') as f:
                rdr = csv.reader(f,delimiter='\t')
                rdr = list(rdr)
                if header:
                    rdr = rdr[1:]
                    sentences = [item[idx_sen+1] for item in rdr]
                    labels = [item[idx_lab+1] for item in rdr]
                    if not is_hoc:
                        labels = [label_list.index(item) for item in labels]
                else:
                    sentences = [item[idx_sen] for item in rdr]
                    labels = [item[idx_lab] for item in rdr]
                    if not is_hoc:
                        labels = [label_list.index(item) for item in labels]
            dataset_dict[d_type] = [[sen,lab] for sen,lab in zip(sentences, labels)]
    return dataset_dict

dataset_dict = {}

label_list_dict = {'chemprot' : ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"],
                   'gad' : ['0','1'],
                   'ddi':["DDI-mechanism", "DDI-effect", "DDI-advise", "DDI-int", "DDI-false"],
                   'i2b2':['PIP', 'TeCP', 'TeRP', 'TrAP', 'TrCP', 'TrIP', 'TrNAP', 'TrWP', 'false']}

if args.task == 'chemprot':
    dataset_dict = dataset_dict_loader(args.input_dir,label_list_dict[args.task])
    config_teacher = transformers.BertConfig.from_pretrained(args.config_file_teacher, num_labels=len(label_list_dict[args.task]))
    config = transformers.RobertaConfig.from_pretrained(args.config_file, num_labels=len(label_list_dict[args.task]))

if args.task == 'gad':
    dataset_dict = dataset_dict_loader(args.input_dir,label_list_dict[args.task],dev_header=True,tst_header=False)
    config_teacher = transformers.BertConfig.from_pretrained(args.config_file_teacher, num_labels=len(label_list_dict[args.task]))
    config = transformers.RobertaConfig.from_pretrained(args.config_file, num_labels=len(label_list_dict[args.task]))

if args.task == 'ddi':
    dataset_dict = dataset_dict_loader(args.input_dir,label_list_dict[args.task],trn_header=True,dev_header=True)
    config_teacher = transformers.BertConfig.from_pretrained(args.config_file_teacher, num_labels=len(label_list_dict[args.task]))
    config = transformers.RobertaConfig.from_pretrained(args.config_file, num_labels=len(label_list_dict[args.task]))

if args.task == 'i2b2':
    dataset_dict = dataset_dict_loader(args.input_dir,label_list_dict[args.task],trn_header=True,dev_header=True)
    config_teacher = transformers.BertConfig.from_pretrained(args.config_file_teacher, num_labels=len(label_list_dict[args.task]))
    config = transformers.RobertaConfig.from_pretrained(args.config_file, num_labels=len(label_list_dict[args.task]))

else:
    dataset_dict = dataset_dict_loader(args.input_dir,label_list_dict[args.task],trn_header=True,dev_header=True,is_hoc=True)
    config_teacher = transformers.BertConfig.from_pretrained(args.config_file_teacher,num_labels=20,problem_type="multi_label_classification")
    config = transformers.RobertaConfig.from_pretrained(args.config_file,num_labels=20,problem_type="multi_label_classification")

print("########## Initializing model ... ##########")
teacher_model = transformers.BertForSequenceClassification.from_pretrained(args.init_checkpoint_teacher,config=config_teacher)
model = transformers.RobertaForSequenceClassification.from_pretrained(args.init_checkpoint,config=config)

teacher_model.cuda()
model.cuda()

print("Done.")

# TODO: directories should be changed into execution arguments :DONE
tokenizer_teacher = transformers.BertTokenizer.from_pretrained(args.vocab_path_teacher)
tokenizer = transformers.RobertaTokenizer.from_pretrained(args.vocab_path if args.vocab_path is not None else "roberta-large")

def preprocessing(tokenizer, sentences,labels,datatype,is_hoc=False):
    assert datatype in ['train','dev','test'], "Wrong datatype, only train,dev,test allowed"
    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
    
    print('\nPadding/truncating all sentences to %d values...' % args.max_seq_len)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=args.max_seq_len, dtype="long", 
                              value=0, truncating="post", padding="post")
    print('Done.')
    
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]

        attention_masks.append(att_mask)
    
    if is_hoc:
        # Label processing for HoC dataset
        label_list = []
        aspect_value_list = [0,1]
        for i in range(10):
            for value in aspect_value_list:
                label_list.append(str(i) + "_" + str(value))

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        
        label_ids = []

        for label in labels:
            # get list of label
            label_id_list = []
            label_list = label.split(",")
            for label_ in label_list:
                label_id_list.append(label_map[label_])
            # convert to multi-hot style
            label_id = [0 for l in range(len(label_map))]
            for j, label_index in enumerate(label_id_list):
                label_id[label_index] = 1
            label_ids.append(label_id)
        pp_labels = torch.tensor(label_ids)

    else:
        pp_labels = torch.tensor(labels)
    
    pp_inputs = torch.tensor(input_ids)
    pp_masks = torch.tensor(attention_masks)

    pp_data = TensorDataset(pp_inputs, pp_masks, pp_labels)
    pp_sampler = RandomSampler(pp_data)

    if datatype == 'test' or datatype == 'dev':
        pp_dataloader = DataLoader(pp_data, sampler=None, batch_size=args.batch_size)
    else:
        pp_dataloader = DataLoader(pp_data, sampler=pp_sampler, batch_size=args.batch_size)

    return pp_data,pp_sampler,pp_dataloader

def preprocessing_ABD(tokenizer_T,tokenizer_S, sentences,labels,datatype,is_hoc=False):
    assert datatype in ['train','dev','test'], "Wrong datatype, only train,dev,test allowed"
    input_ids_T = []
    input_ids_S = []
    for sent in sentences:
        encoded_sent = tokenizer_T.encode(sent, add_special_tokens = True)
        input_ids_T.append(encoded_sent)

        encoded_sent = tokenizer_S.encode(sent, add_special_tokens = True)
        input_ids_S.append(encoded_sent)
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids_S[0])
    
    print('\nPadding/truncating all sentences to %d values...' % args.max_seq_len)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids_T = pad_sequences(input_ids_T, maxlen=args.max_seq_len, dtype="long", 
                                value=0, truncating="post", padding="post")
    input_ids_S = pad_sequences(input_ids_S, maxlen=args.max_seq_len, dtype="long", 
                                value=0, truncating="post", padding="post")
    print('Done.')
    
    attention_masks_T = []
    attention_masks_S = []
    for sent_T,sent_S in zip(input_ids_T,input_ids_S):
        att_mask = [int(token_id > 0) for token_id in sent_T]
        attention_masks_T.append(att_mask)

        att_mask = [int(token_id > 0) for token_id in sent_S]
        attention_masks_S.append(att_mask)
    
    if is_hoc:
        # Label processing for HoC dataset
        label_list = []
        aspect_value_list = [0,1]
        for i in range(10):
            for value in aspect_value_list:
                label_list.append(str(i) + "_" + str(value))

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        
        label_ids = []

        for label in labels:
            # get list of label
            label_id_list = []
            label_list = label.split(",")
            for label_ in label_list:
                label_id_list.append(label_map[label_])
            # convert to multi-hot style
            label_id = [0 for l in range(len(label_map))]
            for j, label_index in enumerate(label_id_list):
                label_id[label_index] = 1
            label_ids.append(label_id)
        pp_labels = torch.tensor(label_ids)

    else:
        pp_labels = torch.tensor(labels)

    pp_inputs_T = torch.tensor(input_ids_T)
    pp_masks_T = torch.tensor(attention_masks_T)

    pp_inputs_S = torch.tensor(input_ids_S)
    pp_masks_S = torch.tensor(attention_masks_S)

    pp_data = TensorDataset(pp_inputs_T, pp_masks_T, pp_inputs_S, pp_masks_S, pp_labels)
    pp_sampler = RandomSampler(pp_data)

    if datatype == 'test' or datatype == 'dev':
        pp_dataloader = DataLoader(pp_data, sampler=None, batch_size=args.batch_size)
    else:
        pp_dataloader = DataLoader(pp_data, sampler=pp_sampler, batch_size=args.batch_size)

    return pp_data,pp_sampler,pp_dataloader

if args.do_train:
    _,_,train_dataloader = preprocessing_ABD(tokenizer_teacher, tokenizer,
                                         [item[0] for item in dataset_dict['train']],
                                         [item[1] for item in dataset_dict['train']],
                                         'train',is_hoc=True if args.task=='hoc' else False)
if args.do_eval:
    _,_,dev_dataloader = preprocessing_ABD(tokenizer_teacher, tokenizer,
                                       [item[0] for item in dataset_dict['dev']],
                                       [item[1] for item in dataset_dict['dev']],
                                       'dev',is_hoc=True if args.task=='hoc' else False)
if args.do_predict:
    _,_,test_dataloader = preprocessing_ABD(tokenizer_teacher, tokenizer,
                                        [item[0] for item in dataset_dict['test']],
                                        [item[1] for item in dataset_dict['test']],
                                        'test',is_hoc=True if args.task=='hoc' else False)

if args.do_train:
    optimizer = AdamW(model.parameters(),
                      lr = args.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = args.epochs
    # Total number of training steps is number of batches * number of epochs.

    total_steps = len(train_dataloader) * args.epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# torch.manual_seed(123)
torch.manual_seed(args.random_seed)
expander_model = nn.Sequential(nn.Linear(768,1024))
expander_model.apply(init_weights)
expander_model.cuda()

def dimension_expander(input_seqs,model):
    with torch.no_grad():
        outputs = model(input_seqs.cuda())
    return outputs

def intuitive_loss(gs,prd):
    gs = np.array(gs)
    prd = np.array(prd)

    assert gs.shape == prd.shape
    
    def ro(x):
        res = []
        for item in x: 
            res.append(1 if item>0 else 0)
        return np.array(res)
    
    total = 0
    for item1, item2 in zip(gs,prd):
        diff = ro(item1) - ro(item2)
        total += np.linalg.norm(diff,1)
    return total/len(gs)

def criterion_alternative_L2(source, target, margin):
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).float() +
            (source - margin)**2 * ((source <= margin) & (target > 0)).float())
    return torch.abs(loss).sum()

if args.do_train:
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    epoch_switch = int(epochs * args.loss_switch_rate)

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        # Measure how long the training epoch takes.
        t0 = time.time()
        
        # Reset the total loss for this epoch.
        total_loss = 0
        
        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        teacher_model.eval()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:

                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Load batch for teacher
            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_labels = batch[4].to(device).long() if args.task != 'hoc' else batch[4].to(device).float()

            with torch.no_grad():
                outputs = teacher_model(input_ids = b_input_ids,
                                                  attention_mask = b_input_mask,
                                                  return_dict=True,
                                                  output_hidden_states=True,
                                                  labels = b_labels)

            cls_T = outputs['hidden_states'][-1][:,0,:]
            cls_T = dimension_expander(cls_T,expander_model)

            # Load batch for student (keep label ids since they use same label id)
            b_input_ids = batch[2].to(device).long()
            b_input_mask = batch[3].to(device).long()

            model.zero_grad()        

            outputs = model(input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            labels = b_labels
                           )

            cls_S = outputs['hidden_states'][-1][:,0,:]

            # loss = outputs['loss']

            loss_AT = ((cls_S > 0) ^ (cls_T > 0)).sum().float() / cls_T.nelement()

            logits = outputs['logits']
            sm = torch.nn.Softmax(dim=-1)
            lsm = torch.nn.LogSoftmax(dim=-1)            

            # loss switch version
            if epoch_i < epoch_switch:
                # margin = 1.0
                margin = args.margin
                loss_alter = criterion_alternative_L2(cls_S, cls_T, margin) / 1000
                # loss_alter = loss_alter - beta*negative_entropy.sum()
                loss = loss_alter / args.batch_size

            else:
                probabilities = sm(logits)
                log_probs = lsm(logits)

                negative_entropy = - torch.sum(probabilities * log_probs, dim=-1)

                criterion_ce = torch.nn.CrossEntropyLoss()
                loss_ce = criterion_ce(logits.view(-1, len(label_list_dict[args.task])), b_labels.view(-1))
                loss = torch.mean(loss_ce - args.confidence_penalty_weight * negative_entropy)

            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    print('Saving model ...')
    model.save_pretrained(os.path.join(args.output_dir))
    print("Done")

def biotask_eval(prd_result,test_answer,task):
    assert task in ['chemprot','gad','ddi','i2b2','hoc'] , "The task is not supported"

    pred_class = [np.argmax(v) for v in prd_result]

    if task == 'chemprot':
        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, 
                                                              labels=[0,1,2,3,4], average="micro")
    elif task == 'gad':
        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer,average="binary")
    elif task == 'ddi':
        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, 
                                                              labels=[0,1,2,3], average="micro")
    elif task == 'i2b2':
        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, 
                                                              labels=[0,1,2,3,4,5,6,7], average="micro")
    else:
        print("Internal validation for HoC is not supported, only export logits")
        p,r,f,s = 0,0,0,0

    results = dict()
    results["f1 score"] = f
    results["recall"] = r
    results["precision"] = p

    print(task)
    for k,v in results.items():
        print("{:11s} : {:.2%}".format(k,v))
    print()

    return results


if args.do_eval:
    print("")
    print("Running Validation...")
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    teacher_model.eval()
    model.eval()
    # Tracking variables 
    t0 = time.time()

    eval_accuracy = 0
    nb_eval_steps = 0
    logits_all = []

    output_all_T = []
    output_all_S = []

    for batch in dev_dataloader:

        # Add batch to GPU
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[4].to(device).long() if args.task != 'hoc' else batch[4].to(device).float()

        with torch.no_grad():
            outputs = teacher_model(input_ids = b_input_ids,
                                              attention_mask = b_input_mask,
                                              return_dict=True,
                                              output_hidden_states=True,
                                              labels = b_labels)

        cls_T = outputs['hidden_states'][-1][:,0,:]
        cls_T = dimension_expander(cls_T,expander_model)

        # Load batch for student (keep label ids since they use same label id)
        b_input_ids = batch[2].to(device).long()
        b_input_mask = batch[3].to(device).long()

        with torch.no_grad():
            outputs = model(input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            labels = b_labels
                           )

        cls_S = outputs['hidden_states'][-1][:,0,:]

        # loss = outputs['loss']

        loss_AT = ((cls_S > 0) ^ (cls_T > 0)).sum().float() / cls_T.nelement()
            
        logits = outputs['logits']
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        logits_all += list(logits)
        
        cls_T = cls_T.detach().cpu().numpy()
        cls_S = cls_S.detach().cpu().numpy()

        output_all_T.append(cls_T)
        output_all_S.append(cls_S)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1
    # Report the final accuracy for this validation run.
    if args.task != 'hoc':
        print("  Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))

    output_all_T = np.concatenate(output_all_T,axis=0)
    output_all_S = np.concatenate(output_all_S,axis=0)

    intuitive_dev = intuitive_loss(output_all_T,output_all_S)

    results = biotask_eval(logits_all,[item[1] for item in dataset_dict['dev']],args.task)
    
    if args.task != 'hoc':
        with open(os.path.join(args.output_dir,'eval_results_'+args.task+'.txt'),'w') as f_ev:
            f_ev.write("Accuracy: {0:.4f}\n".format(eval_accuracy/nb_eval_steps))
            f_ev.write("intuitive loss: {0:.4f}\n".format(intuitive_dev))
            for k,v in results.items():
                f_ev.write("{:11s} : {:.2%}\n".format(k,v))
    else:
        with open(os.path.join(args.output_dir,'eval_results_'+args.task+'.tsv'),'w',newline='') as f_prd:
            wr = csv.writer(f_prd,delimiter='\t')
            for logit in logits_all:
                wr.writerow(logit)

    print("  Validation took: {:}".format(format_time(time.time() - t0)))

if args.do_predict:
    print("")
    print("Running Predictiontion...")
    
    model.eval()
    teacher_model.eval()
    # Tracking variables 
    t0 = time.time()

    eval_accuracy = 0
    nb_eval_steps = 0
    logits_all = []

    output_all_T = []
    output_all_S = []

    for batch in test_dataloader:

        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[4].to(device).long() if args.task != 'hoc' else batch[4].to(device).float()

        with torch.no_grad():
            outputs = teacher_model(input_ids = b_input_ids,
                                              attention_mask = b_input_mask,
                                              return_dict=True,
                                              output_hidden_states=True,
                                              labels = b_labels)

        cls_T = outputs['hidden_states'][-1][:,0,:]
        cls_T = dimension_expander(cls_T,expander_model)

        # Load batch for student (keep label ids since they use same label id)
        b_input_ids = batch[2].to(device).long()
        b_input_mask = batch[3].to(device).long()

        with torch.no_grad():
            outputs = model(input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            labels = b_labels
                           )

        cls_S = outputs['hidden_states'][-1][:,0,:]

        # loss = outputs['loss']

        loss_AT = ((cls_S > 0) ^ (cls_T > 0)).sum().float() / cls_T.nelement()
            
        logits = outputs['logits']
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        logits_all += list(logits)
        
        cls_T = cls_T.detach().cpu().numpy()
        cls_S = cls_S.detach().cpu().numpy()

        output_all_T.append(cls_T)
        output_all_S.append(cls_S)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1
    
    with open(os.path.join(args.output_dir,'test_results.tsv'),'w',newline='') as f_prd:
        wr = csv.writer(f_prd,delimiter='\t')
        for logit in logits_all:
            wr.writerow(logit)

    # Report the final accuracy for this validation run.
    if args.task != 'hoc':
            print("  Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    
    output_all_T = np.concatenate(output_all_T,axis=0)
    output_all_S = np.concatenate(output_all_S,axis=0)

    intuitive_tst = intuitive_loss(output_all_T,output_all_S)

    results = biotask_eval(logits_all,[item[1] for item in dataset_dict['test']],args.task)

    if args.task != 'hoc':
        with open(os.path.join(args.output_dir,'test_eval_results_'+args.task+'.txt'),'w') as f_ev:
            f_ev.write("Accuracy: {0:.4f}\n".format(eval_accuracy/nb_eval_steps))
            f_ev.write("intuitive loss: {0:.4f}\n".format(intuitive_tst))
            for k,v in results.items():
                f_ev.write("{:11s} : {:.2%}\n".format(k,v))

    print("  Prediction took: {:}".format(format_time(time.time() - t0)))