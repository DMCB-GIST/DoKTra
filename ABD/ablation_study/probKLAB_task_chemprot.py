import numpy as np
import os
import random
import argparse
import csv

import sklearn.metrics

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import transformers
from transformers import AlbertTokenizer

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AlbertTokenizer
from transformers import AlbertConfig, AlbertModel, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split

import torch

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',default=None,type=str)
parser.add_argument('--train_prob_path',default=None,type=str)
parser.add_argument('--dev_prob_path',default=None,type=str)
parser.add_argument('--test_prob_path',default=None,type=str)

parser.add_argument('--spm_model_path',default=None,type=str)
parser.add_argument('--init_checkpoint',default=None,type=str)
parser.add_argument('--albert_config_file',default=None,type=str)

parser.add_argument('--output_dir',default=None,type=str)

parser.add_argument('--max_seq_len',default=256,type=int)
parser.add_argument('--batch_size',default=10,type=int)
parser.add_argument('--epochs',default=5,type=int)
parser.add_argument('--lr',default=2e-5,type=float)
parser.add_argument('--margin',default=1.0,type=float)

parser.add_argument('--do_train',default=False,type=bool)
parser.add_argument('--do_eval',default=False,type=bool)
parser.add_argument('--do_predict',default=False,type=bool)

parser.add_argument('--loss_mode',default='switch',type=str)
parser.add_argument('--sum_loss_ratio',default=0.5,type=float)
parser.add_argument('--loss_switch_rate',default=0.7,type=float)
parser.add_argument('--confidence_penalty_weight',default=0.3,type=float)

args = parser.parse_args()

assert args.loss_mode in ['switch','sum'], "only switch or sum allowed as loss_mode"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('%d GPU(s) available.' % torch.cuda.device_count())
    print('use GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU.')
    device = torch.device("cpu")


def preprocessing(tokenizer, sentences,labels,cls,datatype):
    assert datatype in ['train','dev','test'], "Wrong datatype, only train,dev,test allowed"
    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
        
    MAX_LEN = args.max_seq_len
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
    print('Done.')
    
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]

        attention_masks.append(att_mask)
    
    pp_inputs = torch.tensor(input_ids)
    pp_labels = torch.tensor(labels)
    pp_masks = torch.tensor(attention_masks)
    pp_cls = torch.tensor(cls)

    pp_data = TensorDataset(pp_inputs, pp_masks, pp_labels,pp_cls)
    pp_sampler = RandomSampler(pp_data)
    if datatype == 'test' or datatype == 'dev':
        pp_dataloader = DataLoader(pp_data, sampler=None, batch_size=args.batch_size)
    else:
        pp_dataloader = DataLoader(pp_data, sampler=pp_sampler, batch_size=args.batch_size)
    
    return pp_data,pp_sampler,pp_dataloader

import time
import datetime
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

labels_all = ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]

tokenizer = AlbertTokenizer(args.spm_model_path)

print('Loading original data...')
# Feed sentence data here
if args.do_train == True:
    with open(os.path.join(args.data_dir, "train.tsv"),'r',encoding='utf-8') as f_in:
        rdr = csv.reader(f_in,delimiter='\t')
        rdr = list(rdr)
        sentences_trn = [item[0] for item in rdr]
        labels_trn = [item[1] for item in rdr]
        labels_trn = [labels_all.index(item) for item in labels_trn]

    num_trn_data =len(sentences_trn)

    with open(args.train_prob_path,'r') as f:
        rdr = csv.reader(f,delimiter='\t')
        cls_biobert_trn = [list(map(float,item)) for item in rdr]

    train_data,train_sampler,train_dataloader = preprocessing(tokenizer,sentences_trn,labels_trn,cls_biobert_trn,'train')

if args.do_eval == True:
    with open(os.path.join(args.data_dir, "dev.tsv"),'r',encoding='utf-8') as f_in:
        rdr = csv.reader(f_in,delimiter='\t')
        rdr = list(rdr)
        sentences_dev = [item[0] for item in rdr]
        labels_dev = [item[1] for item in rdr]
        labels_dev = [labels_all.index(item) for item in labels_dev]

    num_dev_data =len(sentences_dev)
    with open(args.dev_prob_path,'r') as f:
        rdr = csv.reader(f,delimiter='\t')
        cls_biobert_dev = [list(map(float,item)) for item in rdr]  

    dev_data,dev_sampler,dev_dataloader = preprocessing(tokenizer,sentences_dev,labels_dev,cls_biobert_dev,'dev')

if args.do_predict == True:
    with open(os.path.join(args.data_dir, "test.tsv"),'r',encoding='utf-8') as f_in:
        rdr = csv.reader(f_in,delimiter='\t')
        rdr = list(rdr)[1:]
        sentences_tst = [item[1] for item in rdr]
        labels_tst = [item[2] for item in rdr]
        labels_tst = [labels_all.index(item) for item in labels_tst]

    num_tst_data =len(sentences_tst)
    with open(args.test_prob_path,'r') as f:
        rdr = csv.reader(f,delimiter='\t')
        cls_biobert_tst = [list(map(float,item)) for item in rdr]  

    test_data,test_sampler,test_dataloader = preprocessing(tokenizer,sentences_tst,labels_tst,cls_biobert_tst,'test')

print('Done')

tf.io.gfile.makedirs(args.output_dir)

config = AlbertConfig.from_json_file(args.albert_config_file)
config.num_labels = len(labels_all) 

print(config)

model = AlbertForSequenceClassification.from_pretrained(args.init_checkpoint,config=config)
# model = AlbertModel.from_pretrained(args.init_checkpoint,config=config)
model.cuda()

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
print('The ALBERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

if args.do_train:
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
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

# seed_val = 42
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

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

def alternative_loss(gs,prd):
    margin = 1.0
    
    target = np.array(gs)
    source = np.array(prd)
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).astype(float) +
            (source - margin)**2 * ((source <= margin) & (target > 0)).astype(float))
    loss = abs(loss).sum()
    return loss/1000/len(gs)

import random

if args.do_train:
    loss_values = []

    epoch_switch = int(epochs * args.loss_switch_rate)

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        t0 = time.time()
        
        total_loss = 0
        
        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_labels = batch[2].to(device).long()
            b_cls = batch[3].to(device).float()

            model.zero_grad()        

            outputs = model(input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            labels=b_labels
                           )

            # cls_S = outputs['last_hidden_state'][:,0,:]
            # cls_S = outputs['pooler_output']

            logits = outputs['logits']
            sm = torch.nn.Softmax(dim=-1)
            lsm = torch.nn.LogSoftmax(dim=-1)

            probabilities = sm(logits)
            log_probs = lsm(logits)

            beta = args.confidence_penalty_weight
            negative_entropy = - torch.sum(probabilities * log_probs, dim=-1)

            # loss switch version
            if epoch_i < epoch_switch:
                criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
                loss = criterion_kl(log_probs,b_cls)
            else:
                criterion_ce = torch.nn.CrossEntropyLoss()
                loss_ce = criterion_ce(logits.view(-1, len(labels_all)), b_labels.view(-1))
                loss = torch.mean(loss_ce - beta*negative_entropy)

            # # When use AB and CE loss simultaneously
            # margin = args.margin
            # loss_alter = criterion_alternative_L2(cls_S, cls_T, margin) / args.batch_size
            
            # loss = loss_alter / 1000
            
            # loss_ce = outputs['loss']
            
            # loss += loss_ce

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    print('Saving model ...')
    model.save_pretrained(os.path.join(args.output_dir))
    print("Done")

def ChemprotEval_new_variable(prd_result,test_answer):
    pred_class = [np.argmax(v) for v in prd_result]

    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, 
                                                              labels=[0,1,2,3,4], average="micro")
    results = dict()
    results["f1 score"] = f
    results["recall"] = r
    results["precision"] = p

    for k,v in results.items():
        print("{:11s} : {:.2%}".format(k,v))

    return f


if args.do_eval:
    print("")
    print("Running Validation...")
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables 
    eval_accuracy = 0
    nb_eval_steps = 0

    output_all_dev = []
    logits_all_dev = []

    for batch in dev_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()
        b_cls = batch[3].to(device).float()


        with torch.no_grad():        
            outputs = model(input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            labels = b_labels)
        
        # cls_S = outputs['pooler_output']
        # cls_S = torch.atanh(cls_S)
        cls_S = outputs['hidden_states'][-1][:,0,:]

        logits = outputs['logits']

        # Move logits and labels to CPU
        cls_S = cls_S.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        output_all_dev.append(cls_S)
        logits_all_dev += list(logits)
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # Track the number of batches
        nb_eval_steps += 1
    # Report the final accuracy for this validation run.
    output_all_dev = np.concatenate(output_all_dev,axis=0)

    f1_dev = ChemprotEval_new_variable(logits_all_dev,labels_dev)
    print("Accuracy: {0:.6f}".format(eval_accuracy/nb_eval_steps))
    print("micro F1 : ",f1_dev)

    with open(os.path.join(args.output_dir,'eval_results.txt'),'w',newline='') as f_ev:
        f_ev.write("Accuracy: {0:.6f}".format(eval_accuracy/nb_eval_steps))
        f_ev.write('\n')
        f_ev.write("micro F1 : {0:.6f}".format(f1_dev))

if args.do_predict:
    print("")
    print("Running Predictiontion...")
    model.eval()

    logits_all_tst = []
    output_all_tst = []

    for batch in test_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()
        b_cls = batch[3].to(device).float()

        with torch.no_grad():        
            outputs = model(input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            labels = b_labels)
        
        # cls_S = outputs['pooler_output']
        # cls_S = torch.atanh(cls_S)

        cls_S = outputs['hidden_states'][-1][:,0,:]
        logits = outputs['logits']

        # Move logits and labels to CPU
        cls_S = cls_S.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()

        output_all_tst.append(cls_S)
        logits_all_tst += list(logits)
    output_all_tst = np.concatenate(output_all_tst,axis=0)

    f1_tst = ChemprotEval_new_variable(logits_all_tst,labels_tst)

    print("micro F1 : ",f1_tst)

    with open(os.path.join(args.output_dir,'test_results.tsv'),'w',newline='') as f_prd:
        wr = csv.writer(f_prd,delimiter='\t')
        for logit in logits_all_tst:
            wr.writerow(logit)
    with open(os.path.join(args.output_dir,'test_eval_results.txt'),'w',newline='') as f_ev:
        f_ev.write("micro F1 : {0:.6f}".format(f1_tst))

