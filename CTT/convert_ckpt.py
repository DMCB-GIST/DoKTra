import transformers
from transformers.models.bert import convert_bert_original_tf_checkpoint_to_pytorch
import os
import shutil

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir',default=None,type=str)
parser.add_argument('--output_dir',default=None,type=str)

args = parser.parse_args()

os.mkdir(args.output_dir)

convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(os.path.join(args.input_dir,"model.ckpt-1000000"),
    os.path.join(args.input_dir,"bert_config.json"),
    os.path.join(args.output_dir,"biobert.pt"))

shutil.copyfile(os.path.join(args.input_dir,"vocab.txt"), os.path.join(args.output_dir,"vocab.txt"))
shutil.copyfile(os.path.join(args.input_dir,"bert_config.json"), os.path.join(args.output_dir,"bert_config.json"))