# DoKTra
DoKTRa is a **domain knowledge transferring framework** for pre-trained language models. 

In DoKTra, the domain knowledge from the existing in-domain pre-trained language model is extracted and transferred into other PLMs by applying knowledge distillation.

By applying the DoKTra framework on biomedical, clinical, and financial domain downstream tasks, the student models generally retain a high percentage of teacher performance, and even outperform the teachers in certain tasks.

## Workflow
<p align="center"><img src="https://user-images.githubusercontent.com/101086211/157597476-78c41eaa-8805-45bf-86e0-39d88c25868e.jpg" width="800"/></p>

## Main experimental results on biomedical & clinical domains
<p align="center"><img src="https://user-images.githubusercontent.com/101086211/157597627-3427f2ed-72f2-4438-a3ba-43cd7d3e1e3c.jpg" width="800"/></p>

## Main experimental results on financial domain
<p align="center"><img src="https://user-images.githubusercontent.com/101086211/157597647-e4a168ae-2ec9-485d-a60b-c9f5a5c3a9bf.jpg" width="800"/></p>

## Requirements
```
Python==3.7
Pytorch==1.8.1
transformers==4.7.0
```

## Execution examples
### Convert the original BioBERT Tensorflow checkpoint into PyTorch checkpoint
Set ```$ORIGINAL_BIOBERT_CKPT_DIR``` as a folder which contains ```.ckpt``` file, ```bert_config.json``` file, and ```vocab.txt``` file from the original BioBERT repository
```
python CTT/convert_ckpt.py \
--input_dir=$ORIGINAL_BIOBERT_CKPT_DIR \
--output_dir=$CONVERTED_BIOBERT_DIR
```

### Calibrated Teacher Training
Let ```$INPUT_DIR``` indicate a folder for a task dataset which contains ```train.tsv```, ```dev.tsv``` and ```test.tsv```
```
python CTT/BioBERT_ft_cpl.py \
--task=chemprot \
--input_dir=$INPUT_DIR \
--init_checkpoint=$CONVERTED_BIOBERT_DIR/biobert.pt \
--config_file=$CONVERTED_BIOBERT_DIR/bert_config.json \
--vocab_path=$CONVERTED_BIOBERT_DIR/vocab.txt \
--do_train=True \
--do_eval=True \
--do_predict=True \
--output_dir=$OUTPUT_DIR_T \
--max_seq_len=128 \
--batch_size=50 \
--lr=2e-5 \
--random_seed=123 \
--epochs=5 \
--confidence_penalty_weight=0.3
```

### Initial fine-tuning of the student model
```
python ABD/BioMed_RoBERTa_ft.py \
--task=chemprot \
--input_dir=$INPUT_DIR \
--output_dir=$OUTPUT_DIR_S \
--do_train=True \
--do_eval=True \
--do_predict=True \
--max_seq_len=128 \
--batch_size=50 \
--lr=1e-5 \
--random_seed=123 \
--epochs=5 
```

### Calibrated Activation Boundary Distillation
```
python ABD/BioMed_RoBERTa_DoKTra_cpl.py \
--task=chemprot \
--input_dir=$INPUT_DIR \
--init_checkpoint=$OUTPUT_DIR_S/pytorch_model.bin \
--init_checkpoint_teacher=$OUTPUT_DIR_T/pytorch_model.bin \
--config_file=$OUTPUT_DIR_S/config.json \
--config_file_teacher=$OUTPUT_DIR_T/config.json \
--vocab_path_teacher=$CONVERTED_BIOBERT_DIR/vocab.txt \
--output_dir=$OUTPUT_DIR_FINAL \
--do_train=True \
--do_eval=True \
--do_predict=True \
--max_seq_len=128 \
--batch_size=50 \
--lr=1e-5 \
--random_seed=123 \
--epochs=10 \
--loss_switch_rate=0.8 \
--confidence_penalty_weight=0.5
```

## Citation
```
@inproceedings{choi2022domain,
  title={Domain Knowledge Transferring for Pre-trained Language Model via Calibrated Activation Boundary Distillation},
  author={Choi, Dongha and Choi, HongSeok and Lee, Hyunju},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1658--1669},
  year={2022}
}
```
