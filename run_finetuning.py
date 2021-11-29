import os
import pandas as pd

from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BartForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration,BartConfig, PreTrainedTokenizerFast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import emoji
from soynlp.normalizer import repeat_normalize
from tokenization_hanbert import HanBertTokenizer
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from nsmcDatamodule import NsmcDataModule
from easydict import EasyDict 
from hanbert_tokenizer import HanBert_Tokenizer
from pytorch_lightning import loggers as pl_loggers


args = {
    'model_name': 'HanBart-54kN',
    'random_seed': 42, # Random Seed
    'pretrained_model': 'beomi/kcbert-base',  # Transformers PLM name
    'pretrained_tokenizer': '',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    'batch_size': 16,
    'lr': 5e-6,  # Starting Learning Rate
    'epochs': 20,  # Max Epochs
    'max_length': 150,  # Max Length input size
    'train_data_path': "../nsmc/ratings_train.txt",  # Train Dataset file 
    'val_data_path': "../nsmc/ratings_test.txt",  # Validation Dataset file 
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',  # AdamW vs AdamP
    'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
    'fp16': False,  # Enable train on FP16
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    'cpu_workers': 4,
    'max_len': 200,
}

class Model(LightningModule):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)

        # self.model = BertForSequenceClassification.from_pretrained(self.hparams.pretrained_model)
        # self.tokenizer = BertTokenizer.from_pretrained(
        #     self.hparams.pretrained_tokenizer
        #     if self.hparams.pretrained_tokenizer
        #     else self.hparams.pretrained_model
        # )

        #self.tokenizer = HanBertTokenizer.from_pretrained('HanBart-54kN')
        #self.model = BartForSequenceClassification.from_pretrained('../model_checkpoint/HanBart_202110220849/saved_checkpoint_350')
        self.model = model 
        self.tokenizer = tokenizer
        


    def forward(self, **kwargs):
        return self.model(**kwargs)

    def step(self, batch, batch_idx):
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

from pytorch_lightning.callbacks import ModelCheckpoint


def model_load(ckpt_file):
    ckpt = torch.load(ckpt_file)
    epoch = ckpt['epoch']
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    trained_data = ckpt['trained_data_list']

    return  epoch, model_state_dict, optimizer_state_dict, trained_data    


def main():
    log_path = f"model_checkpoint/HanBart_base_300/"
    os.makedirs(log_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath = log_path,
        filename="{epoch}-{val_acc:.4f}-{val_f1:.4f}",
        monitor='val_acc',
        save_top_k=3,
        mode='max',
        #auto_insert_metric_name=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(log_path)


    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])
    seed_everything(args['random_seed'])

    tokenizer = HanBertTokenizer.from_pretrained('HanBart-54kN')
    model = BartForSequenceClassification.from_pretrained('../model_checkpoint/HanBart_base/saved_checkpoint_300', num_labels=2)
 

    #Kobart
    #tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    #model = BartForSequenceClassification.from_pretrained('gogamza/kobart-base-v1')
    
    model = Model(model, tokenizer, **args)

    datamodule = NsmcDataModule(args['train_data_path'], args['val_data_path'], tokenizer, batch_size=args['batch_size'], max_len=args['max_len'])
    #datamodule.setup()

    print(":: Start Training ::")
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args['epochs'],
        fast_dev_run=args['test_mode'],
        num_sanity_val_steps=None if args['test_mode'] else 0,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=[0],#-1 if torch.cuda.is_available() else None,
        precision=16 if args['fp16'] else 32,
        #strategy='dp',
        replace_sampler_ddp=True,
        # For TPU Setup
        tpu_cores=args['tpu_cores'] if args['tpu_cores'] else None,
        logger = tb_logger
    )
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    main()