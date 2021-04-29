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
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, XLMRobertaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "xlmr" #else bert
BATCH_SIZE = 6
EPOCHS = 20
LEARNING_RATE = 2e-6 #[5e-5, 3e-5, 2e-5]
output_type = 'pooler_output' # or 'last_hidden_state'

freeze = True
unfreeze_layers = ["layer." + str(i) for i in range(18, 25)]
unfreeze_layers.append("xlmr.pooler_output")
# unfreeze_layers = ["layer." + str(i) for i in range(4, 12)]
# unfreeze_layers.append("bert.pooler_output")

articles = data.data_preprocessing()
articles_train, articles_val = train_test_split(
    articles,
    test_size=0.4,
    random_state=6
)
articles_test, articles_val = train_test_split(
    articles_val,
    test_size=0.5,
    random_state=6
)

writer = SummaryWriter("tensorboard")
data.print_labels_number(writer,articles_train,articles_val,articles_test)
            

best_accuracy = 0

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []
train_fscore_history = []
val_fscore_history = []
    
if model_name == "xlmr":
    model = (m.xlmr_Classifier()).to(device)
elif model_name == "bert":
    model = (m.bert_Classifier()).to(device)
    

if freeze:
    for n, p in model.named_parameters():
        if model_name in n:
            p.requires_grad = False
        for layer in unfreeze_layers:
            if layer in n:
                p.requires_grad = True

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=True) # optimizer = Adafactor(model.parameters())
loss_function = nn.CrossEntropyLoss().to(device)

if model_name == "xlmr":
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
elif model_name == "bert":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

train_set = data.AiboDataset("train", tokenizer=tokenizer,articles=articles_train)
val_set = data.AiboDataset("val", tokenizer=tokenizer,articles=articles_val)
test_set = data.AiboDataset("val", tokenizer=tokenizer,articles=articles_test)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=data.train_create_batch, shuffle = True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=data.val_create_batch,shuffle = True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=data.val_create_batch,shuffle = True)


# writer.flush()
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('----------')
    train_acc, train_loss, train_cfs_matrix = m.train_epoch(
        model,
        train_dataloader,
        loss_function,
        optimizer,
        device,
        output_type,
        len(articles_train)
    )
    
    val_acc, val_loss, val_cfs_matrix, _ = m.eval_model(
        model,
        val_dataloader,
        loss_function,
        device,
        output_type,
        len(articles_val)
    )
    
    train_recall, train_precision, train_F1_score = result.classification_report(train_cfs_matrix)
    val_recall, val_precision, val_F1_score = result.classification_report(val_cfs_matrix)
    print("#Train")
    print(f'loss : {round(float(train_loss),7)} \naccuracy : {round(float(train_acc),7)}')
    print(f'Recall : {round(train_recall, 5)}  Precision : {round(train_precision, 5)}  F1_score : {round(train_F1_score, 5)}')
    print("Confusion Matrix")
    print(train_cfs_matrix)
    print()
    print("#Validation")
    print(f'loss : {round(float(val_loss),7)} \naccuracy : {round(float(val_acc),7)}')
    print(f'Recall : {round(val_recall, 5)}  Precision : {round(val_precision, 5)}  F1_score : {round(val_F1_score, 5)}')
    print("Confusion Matrix")
    print(val_cfs_matrix)
    print()

    result.record_point(writer,epoch,train_acc,val_acc,train_loss,val_loss,train_F1_score,val_F1_score)

    train_acc_history.append(train_acc.item())
    val_acc_history.append(val_acc.item())
    train_loss_history.append(np.asscalar(train_loss))
    val_loss_history.append(np.asscalar(val_loss))
    if val_acc.item() > best_accuracy and val_acc.item()> 0.85:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc.item()
    if train_acc.item() > 0.99:
        break

model.load_state_dict(torch.load('best_model_state.bin'))
model = model.to(device)

test_acc, _ , test_cfs_matrix, predicted_of_url = m.eval_model(
    model,
    test_dataloader,
    loss_function,
    device,
    output_type,
    len(articles_test)
)

test_recall, test_precision, test_F1_score = result.classification_report(test_cfs_matrix)

test_msg = f"""#Test
accuracy : {round(float(test_acc),7)}         
Recall : {round(test_recall, 5)}  Precision : {round(test_precision, 5)}  F1_score : {round(test_F1_score, 5)}       
True positive : {test_cfs_matrix[1][1]}        
False positive : {test_cfs_matrix[0][1]}        
True negative : {test_cfs_matrix[0][0]}         
False negative : {test_cfs_matrix[1][0]}"""
print(test_msg)
writer.add_text('RESULT', test_msg)

result.record_predicted_of_url(writer,predicted_of_url)
result.save_predicted_file(predicted_of_url)

writer.close()
print("--- %s seconds ---" % (time.time() - start_time))
