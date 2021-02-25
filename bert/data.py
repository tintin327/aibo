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
            article, label = self.data[idx]
            label_tensor = torch.tensor(label)
        elif self.mode == "test":
            article, label = articles[idx]
            label_tensor = None
        encoded_input = self.tokenizer(article,return_tensors="pt",max_length=512,truncation=True)
        return (encoded_input['input_ids'], encoded_input['token_type_ids'], encoded_input["attention_mask"], label_tensor)
    
    def __len__(self):
        return self.len

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

def data_preprocessing():
    conn = sqlite3.connect('aibo.db')
    cursor = conn.execute("SELECT url TEXT FROM collect_log")
    url_in_collect_log = cursor.fetchall()
    cursor = conn.execute("SELECT checked FROM collect_log")
    checked_in_collect_log = cursor.fetchall()
    cursor = conn.execute("SELECT url TEXT FROM parsed_news")
    url_in_parsed_news = cursor.fetchall()
    cursor = conn.execute("SELECT article TEXT FROM parsed_news")
    article_in_parsed_news = cursor.fetchall() 
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
                if(article_len>50):
                    articles.append([article_text,label])
        
    conn = sqlite3.connect('negative_data.db')
    cursor = conn.execute("SELECT ARTICLE FROM NEWS")
    negative_data = cursor.fetchall()
    
    ooo = 0
    for negative_article in negative_data:
        if(negative_article!=None):
            articles.append([negative_article,0])
            ooo = ooo+1
    print(ooo)
            

    return articles

def print_labels_number(articles_train,articles_val):
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

    print(f'number of articles: {len(articles_train)+len(articles_val)}')
    print(f'number of train: {len(articles_train)}')
    print(f'number of train_POS: {(pos_train)}')
    print(f'number of train_NEG: {(neg_train)}')
    print(f'number of val: {len(articles_val)}')
    print(f'number of val_POS: {(pos_val)}')
    print(f'number of val_NEG: {(neg_val)}\n')




