import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import{
    classification_report,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
}

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,trainer,TrainingArguments

from logging_utils import archive_existing_output_txts
from preprocess import clean_text

from train import LABEL_MAPS,LABEL_ORDER

LABEL2ID = {label:idx for idx, label in enumerate(LABEL_ORDER)}
ID2LABEL = {idx:label for idx, label in LABEL2ID.items()}

class TextDataset(Dataset):
    """Tokenize each row of the dataframe and return input_ids, attention_mask, and labels."""
    def __init__(self,texts,labels,tokenizer,max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation = True,
            padding = "max_length"
            max_length= self.max_length,
        )
        item = {k:torch.tensor(v,dtype=torch.long)for k,v in enc.items()}
        item["labels"] = torch.tensor(label,dtype=torch.long)
        return item
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Train a transformer model.")
        parser.add_argument("--data",required = True)
        parser.add_argument("--text-col",default="tweets")
        parser.add_argument("--label-col",default="class")
        parser.add_argument("--label-map, choices")