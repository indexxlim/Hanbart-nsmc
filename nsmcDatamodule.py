from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from soynlp.normalizer import repeat_normalize
import pandas as pd
import emoji
import re

class nsmcDataset(Dataset):
    '''
        To read Korean Corpus
    '''
    def __init__(self,data_path, tokenizer, max_len):
        self.df = self.read_data(data_path)
        self.df = self.df.fillna('')
        self.df['document'] = self.df['document'].apply(self.clean_df)        

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        #return self.df.iloc[index]
        single_row = self.df.iloc[index]
        
        document = single_row.document
        labels = single_row.label

        encoding = self.tokenizer.encode_plus(
            document,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "comment_text": document,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.LongTensor([labels])
        }
        
    def __len__(self):
        return len(self.df)
    
    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        elif path.endswith('json'):
            return pd.read_json(path)
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt)/json are Supported')

    def clean_df(self, x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)

        return x

class NsmcDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=32, max_len=512):
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = nsmcDataset(self.train_df, self.tokenizer, self.max_len)
        self.test_dataset = nsmcDataset(self.test_df, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )        