# coding=utf-8
from __future__ import print_function
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
import numpy
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import sqlite3
import math
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

BATCH_SIZE = 30
EPOCHS = 30
LEARNING_RATE = 2e-5
unfreeze_layers = ['layer.10','layer.11','bert.pooler']
bert_output = 'pooler_output' # or 'last_hidden_state'


def classification_report(cfs_matrix):
    TN, FP, FN, TP = cfs_matrix.ravel()
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    return Recall, Precision, F1_score
    
def print_report(Recall,Precision,F1_score):
    print(f'Recall : {round(Recall, 5)}  Precision : {round(Precision, 5)}  F1_score : {round(F1_score, 5)}')
    return

def save_plot(train_history,val_history,title,EPOCHS,ax):
    points = EPOCHS
    x =  (range(points+1))[1:]
    ax.plot(x, train_history,label = "train")
    ax.plot(x, val_history,label = "val")
    ax.set_title(title)
    return
    
def data_preprocessing(url_in_collect_log,url_in_parsed_news,article_in_parsed_news,checked_in_collect_log):
    articles = []
    for url in url_in_collect_log:
        if url in url_in_parsed_news:
            i_parsed_news = url_in_parsed_news.index(url)
            i_collect_log = url_in_collect_log.index(url)
            label = checked_in_collect_log[i_collect_log][0]
            article_text = article_in_parsed_news[i_parsed_news][0]
            if label!=1:
                label = 0
            if(article_text!=None):
                article_len = len(article_text)
                article_text = article_text.replace("\n","")
                article_text = article_text.replace("\s","")
                article_text = article_text.strip()
                # segment = article_len//510 + 1
                if(article_len>50):
                    articles.append([article_text,label])
#                 if(segment==1 and article_len>50):
#                     articles.append([article_text,label])
#                 else:
#                     for i in range(segment-2):
#                         articles.append([article_text[510*i:510*(i+1)],label])
#                     len_of_last_two = math.ceil(0.5*(article_len%510)) + 256
#                     articles.append([article_text[510*(segment-2):510*(segment-2)+len_of_last_two],label])
#                     articles.append([article_text[510*(segment-2)+len_of_last_two:],label])
    return articles

class AiboDataset(Dataset):
    def __init__(self, mode, tokenizer,articles):
        self.mode = mode
        self.data = articles
        self.len = len(articles)
        self.tokenizer = tokenizer  
    
    def __getitem__(self, idx):
        if self.mode == "train":
            article, label = self.data[idx]
            label_tensor = torch.tensor(label)
        elif self.mode == "test":
            article, label = articles[idx]
            label_tensor = None
        encoded_input = tokenizer(article,return_tensors="pt",max_length=512,truncation=True)
        return (encoded_input['input_ids'], encoded_input['token_type_ids'], encoded_input["attention_mask"], label_tensor)
    
    def __len__(self):
        return self.len

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size,2)
    def forward(self, input_ids, attention_mask):
        output = (torch.tensor([])).to(device)
        if(bert_output == 'pooler_output'):
            _ , output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        elif(bert_output == 'last_hidden_state'):
            last_hidden_state, _ = self.bert(input_ids=input_ids,attention_mask=attention_mask)
            for l in last_hidden_state:
                output = torch.cat([output,l[0]],dim = 0)
            output = torch.reshape(output, (-1, 768))
            
        return self.out(output)


def create_batch(samples):
    tokens_tensors = []
    segments_tensors = []
    masks_tensors = []
    label_ids = []

    tokens_tensors = [s[0][0] for s in samples]
    segments_tensors = [s[1][0] for s in samples]
    masks_tensors = [s[2][0] for s in samples]
    label_ids = torch.stack([s[3] for s in samples])
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,batch_first=True)
    masks_tensors = pad_sequence(masks_tensors,batch_first=True)
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def train_epoch(model,data_loader,loss_function,optimizer,device,n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    cfs_matrix = np.array([[0,0],[0,0]])
    for data in data_loader:
        input_ids, _ , attention_mask, targets = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)      
        loss = loss_function(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        matrix = confusion_matrix(y_true=targets.cpu(), y_pred=preds.cpu())
        if(matrix.shape==(1,1)):
            if(preds[0]==1):
                matrix = np.array([[0,0],[0,matrix[0][0]]])
            else:
                matrix = np.array([[matrix[0][0],0],[0,0]])
        cfs_matrix = cfs_matrix + matrix
            
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples , np.mean(losses)  , cfs_matrix

def eval_model(model, data_loader, loss_function, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    cfs_matrix = np.array([[0,0],[0,0]])
    with torch.no_grad():
        for data in data_loader:
            input_ids, _ , attention_mask, targets = data
#             print(input_ids.shape, targets.shape)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
#           print(preds.shape, outputs.shape, targets.shape)
            loss = loss_function(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            matrix = confusion_matrix(y_true=targets.cpu(), y_pred=preds.cpu())
            if(matrix.shape==(1,1)):
                if(preds[0]==1):
                    matrix = np.array([[0,0],[0,matrix[0][0]]])
                else:
                    matrix = np.array([[matrix[0][0],0],[0,0]])
            cfs_matrix = cfs_matrix + matrix
            losses.append(loss.item())

            
    return correct_predictions.double() / n_examples, np.mean(losses) , cfs_matrix

start_time = time.time()

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

conn = sqlite3.connect('aibo.db')
cursor = conn.execute("SELECT url TEXT FROM collect_log")
url_in_collect_log = cursor.fetchall()
cursor = conn.execute("SELECT checked FROM collect_log")
checked_in_collect_log = cursor.fetchall()
cursor = conn.execute("SELECT url TEXT FROM parsed_news")
url_in_parsed_news = cursor.fetchall()
cursor = conn.execute("SELECT article TEXT FROM parsed_news")
article_in_parsed_news = cursor.fetchall() ##清掉\n

articles = data_preprocessing(url_in_collect_log,url_in_parsed_news,article_in_parsed_news,checked_in_collect_log)
# print(len(articles))
# print(len(articles[0])) = 2 = [[一篇文章分割的set],label] 
# print(len(articles[0][0])) = segament = [一篇文章分割的set]
# print(len(articles[0][0][0])) = part of article
conn = sqlite3.connect('negative_data.db')
cursor = conn.execute("SELECT ARTICLE FROM NEWS")
negative_data = cursor.fetchall()

for negative_article in negative_data:
    if(negative_article!=None):
        articles.append([negative_article,0])

articles_train, articles_val = train_test_split(
    articles,
    test_size=0.4,
    random_state=6
)


pos_train = 0
pos_val = 0
neg_train = 0
neg_val = 0
for i in articles_train:
    if i[1]==1:
        pos_train = pos_train+1
    if i[1]==0:
        neg_train = neg_train+1
    

for i in articles_val:
    if i[1]==1:
        pos_val = pos_val+1
    if i[1]==0:
        neg_val = neg_val+1


print(f'number of articles: {len(articles)}')
print(f'number of train: {len(articles_train)}')
print(f'number of train_POS: {(pos_train)}')
print(f'number of train_NEG: {(neg_train)}')
print(f'number of val: {len(articles_val)}')
print(f'number of val_POS: {(pos_val)}')
print(f'number of val_NEG: {(neg_val)}\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_set = AiboDataset("train", tokenizer=tokenizer,articles=articles_train)
val_set = AiboDataset("train", tokenizer=tokenizer,articles=articles_val)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=create_batch, shuffle = True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=create_batch,shuffle = True)
model = Classifier()

for n, p in model.named_parameters():
    if 'bert' in n:
        p.requires_grad = False
    for layer in unfreeze_layers:
        if layer in n:
            p.requires_grad = True
            
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=True)
loss_function = nn.CrossEntropyLoss().to(device)
best_accuracy = 0

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []
train_fscore_history = []
val_fscore_history = []

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('----------')
    train_acc, train_loss, train_cfs_matrix = train_epoch(
        model,
        train_dataloader,
        loss_function,
        optimizer,
        device,
        len(articles_train)
    )
    
    val_acc, val_loss, val_cfs_matrix = eval_model(
        model,
        val_dataloader,
        loss_function,
        device,
        len(articles_val)
    )
    if(len(articles_val)!=np.sum(val_cfs_matrix)):
                print("VAL")
                exit(0)
    if(len(articles_train)!=np.sum(train_cfs_matrix)):
                print("ERRRR")
                exit(0)
    
    train_recall, train_precision, train_F1_score = classification_report(train_cfs_matrix)
    val_recall, val_precision, val_F1_score = classification_report(val_cfs_matrix)
    print("#Train")
    print(f'loss : {round(float(train_loss),7)} \naccuracy : {round(float(train_acc),7)}')
    print_report(train_recall, train_precision, train_F1_score)
    print("Confusion Matrix")
    print(train_cfs_matrix)
    print()
          
    print("#Validation")
    print(f'loss : {round(float(val_loss),7)} \naccuracy : {round(float(val_acc),7)}')
    print_report(val_recall, val_precision, val_F1_score)
    print("Confusion Matrix")
    print(val_cfs_matrix)
    print()
   
    train_acc_history.append(train_acc.item())
    val_acc_history.append(val_acc.item())
    train_loss_history.append(np.asscalar(train_loss))
    val_loss_history.append(np.asscalar(val_loss))
    train_fscore_history.append(np.asscalar(train_F1_score))
    val_fscore_history.append(np.asscalar(val_F1_score))
    

writer = SummaryWriter("tensorboard")

for e in range(EPOCHS):
    writer.add_scalars("accuracy",{
    'train': train_acc_history[e],
    'validation': val_acc_history[e],
    }, e+1)
    writer.add_scalars("loss",{
    'train': train_acc_history[e],
    'validation': val_acc_history[e],
    }, e+1)
    writer.add_scalars("F1_score",{
    'train': train_fscore_history[e],
    'validation': val_fscore_history[e],
    }, e+1)
    # print(train_acc_history[e])
    # print(type(train_acc_history[e]))
    # writer.add_scalar("accuracy",{'validation': val_acc_history[e]}, e)
    # writer.add_scalar("loss",{'train': train_loss_history[e]}, e)
    # writer.add_scalar("loss",{'validation': val_loss_history[e]}, e)
    # writer.add_scalar("F1_score",{'train': train_fscore_history[e]}, e)
    # writer.add_scalar("F1_score",{'train': val_fscore_history[e]}, e)
writer.close()

# save_plot(train_acc_history,val_acc_history,'accuracy',EPOCHS,ax1)
# save_plot(train_loss_history,val_loss_history,'loss',EPOCHS,ax2)
# save_plot(train_fscore_history,val_fscore_history,'F1 score',EPOCHS,ax3)
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.savefig('history')
print("--- %s seconds ---" % (time.time() - start_time))
