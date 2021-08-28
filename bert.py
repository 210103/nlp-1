from torch.utils.data import Dataset,DataLoader,random_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from transformers import BertTokenizer,BertModel,AutoModel,AutoTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import time

class BertTextClassficationModel(nn.Module):
    def __init__(self):
        super(BertTextClassficationModel,self).__init__()
        self.bert=AutoModel.from_pretrained('bert-base-uncased',mirror='tuna')
        self.classification_head=nn.Sequential(nn.Linear(768,2))
        
    def forward(self,ids,mask):
        out=self.bert(input_ids=ids,attention_mask=mask).last_hidden_state
        out=self.classification_head(out[:,0,:])
        return out

class DataToDataset(Dataset):
    def __init__(self,sentences,labels):
        tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased',mirror='tuna')
        max_length=600
        self.encoding=tokenizer(sentences,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
        self.labels=torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self,index):
        return self.encoding['input_ids'][index],self.encoding['attention_mask'][index],self.labels[index]


train_df = pd.read_csv('data/train_dataset.csv',usecols=['text','label'])
val_df = pd.read_csv('data/val_dataset.csv',usecols=['text','label'])
train_df = pd.concat([train_df,val_df])
train_df = train_df[3000:7000]
print(train_df.shape)
sentences = list(train_df['text'])
labels =train_df['label'].values

datasets=DataToDataset(sentences,labels)
train_size=int(len(datasets)*0.7)
test_size=len(datasets)-train_size
train_dataset,val_dataset=random_split(dataset=datasets,lengths=[train_size,test_size])
BATCH_SIZE=32
train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(dataset=val_dataset,batch_size=BATCH_SIZE,shuffle=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_func=nn.CrossEntropyLoss()
model=BertTextClassficationModel()
model = model.to(device)
optimizer=optim.Adam(model.parameters(),lr=0.0001)
writer=SummaryWriter('./tb_image')

epochs = 100
val_acc_best = 0
for epoch in range(epochs):
    
    print(f"epoch {epoch}: starting")
    train_loss = 0.0
    train_acc=0.0
    val_loss=0
    val_acc=0.0
    start_time = time.time()

    model.train()
    for i,data in enumerate(train_loader):
        input_ids,attention_mask,labels=[elem.to(device) for elem in data]
        #优化器置零
        optimizer.zero_grad()
        #得到模型的结果
        out=model(input_ids,attention_mask)
        #计算误差
        loss=loss_func(out,labels)
        writer.add_scalar('train_loss',loss,epoch)
        train_loss += loss.item()
        #误差反向传播
        loss.backward()
        #更新模型参数
        optimizer.step()
        #计算acc 
        train_acc+=(out.argmax(1)==labels).float().mean()
        
    train_acc/=len(train_loader)
    train_loss/=len(train_loader)
    print("train %d/%d epochs Loss:%f, Acc:%f, used time:%fs" %(epoch,epochs,train_loss,train_acc,time.time()-start_time))
    
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for j,batch in enumerate(val_loader):
            val_input_ids,val_attention_mask,val_labels=[elem.to(device) for elem in batch]
            pred=model(val_input_ids,val_attention_mask)
            loss=loss_func(pred,val_labels)
            writer.add_scalar('val_loss',loss,epoch)
            val_acc += (pred.argmax(1)==val_labels).float().mean()
            val_loss += loss.item()
            
    val_acc/=len(val_loader)
    val_loss/=len(val_loader)
    print("val   %d/%d epochs Loss:%f, Acc:%f, used time:%fs" %(epoch,epochs,val_loss,val_acc,time.time()-start_time))

    if val_acc > val_acc_best:
        val_acc_best = val_acc
        best_state_dict = model.state_dict()

    print('Best accuracy on validation set: %3.6f' % val_acc_best)
    
writer.close()

test_df = pd.read_csv('data/test_dataset.csv',usecols=['text','label'])
test_sentences = list(test_df['text'])
test_labels = test_df['label'].values
test_dataset = DataToDataset(test_sentences,test_labels)
test_loader = DataLoader(dataset = test_dataset,batch_size=BATCH_SIZE,shuffle=True)

model = BertTextClassficationModel()
model = model.to(device)
model.load_state_dict(best_state_dict)
model.eval()
test_acc = 0

with torch.no_grad():
    for j,batch in enumerate(test_loader):
        test_input_ids,test_attention_mask,test_labels=[elem.to(device) for elem in batch]
        pred=model(test_input_ids,test_attention_mask)
        test_acc += (pred.argmax(1)==test_labels).float().mean()

print('Accuracy on test set: %3.6f' % (test_acc/len(test_loader)))