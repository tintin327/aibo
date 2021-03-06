# coding=utf-8
from __future__ import print_function
from collections import defaultdict
import data, result
import model as m
import math
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import time
from transformers import AdamW, Adafactor, AdamWeightDecay
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 2e-5 #[5e-5, 3e-5, 2e-5]
unfreeze_layers = ['layer.1','layer.2','layer.3','layer.4','layer.5','layer.6','layer.7','layer.8','layer.9','layer.10','layer.11','bert.pooler']
bert_output = 'pooler_output' # or 'last_hidden_state'


articles = data.data_preprocessing()
articles_train, articles_val = train_test_split(
    articles,
    test_size=0.2,
    random_state=6
)
data.print_labels_number(articles_train=articles_train,articles_val=articles_val)
            

best_accuracy = 0

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []
train_fscore_history = []
val_fscore_history = []
    
model = (m.Classifier()).to(device)


# for n, p in model.named_parameters():
#     if 'bert' in n:
#         p.requires_grad = False
#     for layer in unfreeze_layers:
#         if layer in n:
#             p.requires_grad = True

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=True) # optimizer = Adafactor(model.parameters())
loss_function = nn.CrossEntropyLoss().to(device)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_set = data.AiboDataset("train", tokenizer=tokenizer,articles=articles_train)
val_set = data.AiboDataset("train", tokenizer=tokenizer,articles=articles_val)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=data.create_batch, shuffle = True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=data.create_batch,shuffle = True)


writer = SummaryWriter("tensorboard")
# writer.flush()

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('----------')
    train_acc, train_loss, train_cfs_matrix = m.train_epoch(
        model,
        train_dataloader,
        loss_function,
        optimizer,
        device,
        bert_output,
        len(articles_train)
    )
    
    val_acc, val_loss, val_cfs_matrix = m.eval_model(
        model,
        val_dataloader,
        loss_function,
        device,
        bert_output,
        len(articles_val)
    )
    train_recall, train_precision, train_F1_score = result.classification_report(train_cfs_matrix)
    val_recall, val_precision, val_F1_score = result.classification_report(val_cfs_matrix)
    print("#Train")
    print(f'loss : {round(float(train_loss),7)} \naccuracy : {round(float(train_acc),7)}')
    result.print_report(train_recall, train_precision, train_F1_score)
    print("Confusion Matrix")
    print(train_cfs_matrix)
    print()
    print("#Validation")
    print(f'loss : {round(float(val_loss),7)} \naccuracy : {round(float(val_acc),7)}')
    result.print_report(val_recall, val_precision, val_F1_score)
    print("Confusion Matrix")
    print(val_cfs_matrix)
    print()

    result.record_point(writer,epoch,train_acc,val_acc,train_loss,val_loss,train_F1_score,val_F1_score)

    # train_acc_history.append(train_acc.item())
    # val_acc_history.append(val_acc.item())
    # train_loss_history.append(np.asscalar(train_loss))
    # val_loss_history.append(np.asscalar(val_loss))
    # train_fscore_history.append(np.asscalar(train_F1_score))
    # val_fscore_history.append(np.asscalar(val_F1_score))
                
writer.close()
print("--- %s seconds ---" % (time.time() - start_time))
