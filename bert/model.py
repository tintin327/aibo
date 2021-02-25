import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from sklearn.metrics import confusion_matrix

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size,2)
    def forward(self, input_ids, attention_mask,bert_output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = (torch.tensor([])).to(device)
        if(bert_output == 'pooler_output'):
            _ , output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        elif(bert_output == 'last_hidden_state'):
            last_hidden_state, _ = self.bert(input_ids=input_ids,attention_mask=attention_mask)
            for l in last_hidden_state:
                output = torch.cat([output,l[0]],dim = 0)
            output = torch.reshape(output, (-1, 768))
            
        return self.out(output)

def train_epoch(model,data_loader,loss_function,optimizer,device,bert_output,n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    cfs_matrix = np.array([[0,0],[0,0]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data in data_loader:
        input_ids, _ , attention_mask, targets = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask,bert_output=bert_output)
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

def eval_model(model, data_loader, loss_function, device,bert_output, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    cfs_matrix = np.array([[0,0],[0,0]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in data_loader:
            input_ids, _ , attention_mask, targets = data
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,bert_output = bert_output)
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

            
    return correct_predictions.double() / n_examples, np.mean(losses) , cfs_matrix