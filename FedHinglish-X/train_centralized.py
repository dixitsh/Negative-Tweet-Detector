import os, json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
from data import load_dataframe, split_dataframe, TextDataset, seed_everything
from model import build_model, tokenizer
from config import CFG


def main():
    seed_everything(CFG.seed); os.makedirs(CFG.output_dir, exist_ok=True)
    df=load_dataframe(); train,val,test=split_dataframe(df); tok=tokenizer()
    model=build_model(False).to(CFG.device)
    tr=DataLoader(TextDataset(train,tok),batch_size=CFG.batch_size,shuffle=True)
    va=DataLoader(TextDataset(val,tok),batch_size=CFG.batch_size)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG.lr,weight_decay=0.01)
    steps=max(1,len(tr)*CFG.local_epochs); sched=get_linear_schedule_with_warmup(opt,0,int(0.1*steps),steps)
    best=0
    for epoch in range(CFG.local_epochs):
        model.train()
        for batch in tr:
            batch={k:v.to(CFG.device) for k,v in batch.items()}; opt.zero_grad(); loss=model(**batch).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        m=evaluate(model,va,tok); print({"epoch":epoch+1,**m})
        if m["macro_f1"]>best: best=m["macro_f1"]; model.save_pretrained(f"{CFG.output_dir}/centralized")
    print("test",evaluate(model,DataLoader(TextDataset(test,tok),batch_size=CFG.batch_size),None))

def evaluate(model,data,tok):
    loader=data if hasattr(data,'__iter__') and not isinstance(data,DataLoader) else data
    if not isinstance(loader,DataLoader): loader=DataLoader(data,batch_size=CFG.batch_size)
    yt=[]; yp=[]; model.eval()
    with torch.no_grad():
        for b in loader:
            yt += b["labels"].numpy().tolist(); x={k:v.to(CFG.device) for k,v in b.items() if k!="labels"}
            yp += model(**x).logits.argmax(-1).cpu().numpy().tolist()
    p,r,f,_=precision_recall_fscore_support(yt,yp,average="macro",zero_division=0)
    return {"accuracy":accuracy_score(yt,yp),"macro_f1":f,"macro_precision":p,"macro_recall":r}

if __name__=='__main__': main()
