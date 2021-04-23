# coding=utf-8
import sqlite3
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
class AiboDataset(Dataset):
    def __init__(self, mode, tokenizer,articles):
        self.mode = mode
        self.data = articles
        self.len = len(articles)
        self.tokenizer = tokenizer  
    
    def __getitem__(self, idx):
        if self.mode == "train":
            article, label, _ = self.data[idx]
        elif self.mode == "val":
            article, label, url = self.data[idx]
            
        label_tensor = torch.tensor(label)
        encoded_input = self.tokenizer(article,return_tensors="pt",max_length=512,truncation=True)
        return (encoded_input['input_ids'], encoded_input["attention_mask"], label_tensor,url) 

    
    def __len__(self):
        return self.len

def train_create_batch(samples):
    tokens_tensors = []
    segments_tensors = []
    masks_tensors = []
    label_ids = []
    tokens_tensors = [s[0][0] for s in samples]
    masks_tensors = [s[1][0] for s in samples]
    label_ids = torch.stack([s[2] for s in samples])
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    masks_tensors = pad_sequence(masks_tensors,batch_first=True)
    return tokens_tensors, masks_tensors, label_ids

def val_create_batch(samples):
    tokens_tensors = []
    segments_tensors = []
    masks_tensors = []
    label_ids = []
    tokens_tensors = [s[0][0] for s in samples]
    masks_tensors = [s[1][0] for s in samples]
    label_ids = torch.stack([s[2] for s in samples])
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    masks_tensors = pad_sequence(masks_tensors,batch_first=True)
    return tokens_tensors, masks_tensors, label_ids

def data_preprocessing():
    conn = sqlite3.connect('aibo.db')
    cursor = conn.execute("SELECT * FROM collect_log")
    collect_log = cursor.fetchall()
    cursor = conn.execute("SELECT * FROM parsed_news")
    parsed_news = cursor.fetchall() 
    articles = []
    
    cursor = conn.execute("SELECT * FROM parsed_news")
    DATA = cursor.fetchall()

    for collect in collect_log:
        for parsed in parsed_news:
            if collect[2] == parsed[1]:
                label = collect[5]
                article_text = parsed[3]
                if label!=1:
                    label = 0
                if(article_text!=None):
                    url =  collect[2]
                    article_len = len(article_text)                         
                    article_text = article_text.replace("\n","")
                    article_text = article_text.replace("\s","")
                    article_text = article_text.strip()
                    if(article_len>50):
                        articles.append([article_text,label,url])
                        if(label==0):
                            neg = neg +1
                        if(label==1):
                            pos = pos +1
                            
    # conn = sqlite3.connect('negative_data.db')
    # cursor = conn.execute("SELECT ARTICLE FROM NEWS")
    # negative_data = cursor.fetchall()
            

    return articles

def print_labels_number(writer,articles_train,articles_val,articles_test):
    pos_train = 0
    pos_val = 0
    pos_test = 0
    neg_train = 0
    neg_val = 0
    neg_test = 0

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

    for i in articles_test:
        if i[1]==1: 
            pos_test = pos_test+1
        if i[1]==0:
            neg_test = neg_test+1
     
    msg = f"""
number of articles: {len(articles_train)+len(articles_val)+len(articles_test)}
number of train: {len(articles_train)}
number of train_POS: {(pos_train)}
number of train_NEG: {(neg_train)}
number of val: {len(articles_val)}
number of val_POS: {(pos_val)}
number of val_NEG: {(neg_val)}   
number of test: {len(articles_test)}
number of test_POS: {(pos_test)}
number of test_NEG: {(neg_test)}\n"""
    
    print(msg)
    writer.add_text('DATA', msg)


