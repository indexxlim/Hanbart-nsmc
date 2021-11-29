import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule, Trainer, seed_everything

from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BartForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration,BartConfig

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize
from nsmcDatamodule import NsmcDataModule
from tokenization_hanbert import HanBertTokenizer



class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        
        #self.model = BertForSequenceClassification.from_pretrained(self.hparams.pretrained_model)
        # self.tokenizer = BertTokenizer.from_pretrained(
        #     self.hparams.pretrained_tokenizer
        #     if self.hparams.pretrained_tokenizer
        #     else self.hparams.pretrained_model
        # )
        self.tokenizer = HanBertTokenizer.from_pretrained('HanBart-54kN')
        self.model = BartForSequenceClassification.from_pretrained('../model_checkpoint/HanBart_202110220849/saved_checkpoint_350', num_labels=2)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def step(self, batch, batch_idx):
        #data, labels = batch
        #output = self(input_ids=data, labels=labels)
        input_ids, attention_mask, labels =batch['input_ids'], batch['attention_mask'], batch['labels']
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            
        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', accuracy_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_precision', precision_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_recall', recall_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_f1', f1_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    # def read_data(self, path):
    #     if path.endswith('xlsx'):
    #         return pd.read_excel(path)
    #     elif path.endswith('csv'):
    #         return pd.read_csv(path)
    #     elif path.endswith('tsv') or path.endswith('txt'):
    #         return pd.read_csv(path, sep='\t')
    #     else:
    #         raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    # def preprocess_dataframe(self, df):
    #     emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    #     pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    #     url_pattern = re.compile(
    #         r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    #     def clean(x):
    #         x = pattern.sub(' ', x)
    #         x = url_pattern.sub('', x)
    #         x = x.strip()
    #         x = repeat_normalize(x, num_repeats=2)
    #         return x

    #     df['document'] = df['document'].map(lambda x: self.tokenizer.encode(
    #         clean(str(x)),
    #         padding='max_length',
    #         max_length=self.hparams.max_length,
    #         truncation=True,
    #     ))
    #     return df

    # def dataloader(self, path, shuffle=False):
    #     df = self.read_data(path)
    #     df = self.preprocess_dataframe(df)

    #     dataset = TensorDataset(
    #         torch.tensor(df['document'].to_list(), dtype=torch.long),
    #         torch.tensor(df['label'].to_list(), dtype=torch.long),
    #     )
    #     return DataLoader(
    #         dataset,
    #         batch_size=self.hparams.batch_size or self.batch_size,
    #         shuffle=shuffle,
    #         num_workers=self.hparams.cpu_workers,
    #     )

    # def train_dataloader(self):
    #     return self.dataloader(self.hparams.train_data_path, shuffle=True)

    # def val_dataloader(self):
    #     return self.dataloader(self.hparams.val_data_path, shuffle=False)



from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filename='epoch{epoch}-val_acc{val_acc:.4f}',
    monitor='val_acc',
    save_top_k=3,
    mode='max',
)        

args = {
    'random_seed': 42, # Random Seed
    'pretrained_model': 'beomi/kcbert-base',  # Transformers PLM name
    'pretrained_tokenizer': '',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    'batch_size': 4,
    'lr': 5e-6,  # Starting Learning Rate
    'epochs': 20,  # Max Epochs
    'max_length': 150,  # Max Length input size
    'train_data_path': "../nsmc/ratings_train.txt",  # Train Dataset file 
    'val_data_path': "../nsmc/ratings_test.txt",  # Validation Dataset file 
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',  # AdamW vs AdamP
    'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
    'fp16': True,  # Enable train on FP16
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    'cpu_workers': 4,
    'max_len': 200
}


def main():
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])
    seed_everything(args['random_seed'])
    model = Model(**args)

    tokenizer = BertTokenizer.from_pretrained(
                args['pretrained_tokenizer']
                if args['pretrained_tokenizer']
                else args['pretrained_model']
            )

    datamodule = NsmcDataModule(args['train_data_path'], args['val_data_path'], tokenizer, max_len=args['max_len'])


    print(":: Start Training ::")
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args['epochs'],
        fast_dev_run=args['test_mode'],
        num_sanity_val_steps=None if args['test_mode'] else 0,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args['fp16'] else 32,
        # For TPU Setup
        # tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    main()