import copy, time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data import TextDataset
from model import build_model, adapter_state, load_adapter_state, trainable_parameters
from privacy import clip_state_update, add_gaussian_noise
from config import CFG


def train_client(global_state, frame, tokenizer, client_id, personalized=True):
    model = build_model(True).to(CFG.device)
    load_adapter_state(model, global_state)
    loader = DataLoader(TextDataset(frame, tokenizer), batch_size=CFG.batch_size, shuffle=True)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=CFG.lr)
    model.train(); start = time.time()
    for _ in range(CFG.local_epochs):
        for batch in loader:
            batch = {k:v.to(CFG.device) for k,v in batch.items()}
            opt.zero_grad(); out = model(**batch); out.loss.backward(); opt.step()
    new_state = adapter_state(model)
    clipped, raw_norm, scale = clip_state_update(global_state, new_state, CFG.clip_norm)
    private = add_gaussian_noise(clipped, CFG.dp_sigma, CFG.clip_norm)
    return private, {"client":client_id, "samples":len(frame), "raw_update_norm":raw_norm,
                      "clip_scale":scale, "seconds":time.time()-start}


def weighted_aggregate(states, weights):
    total = float(sum(weights)); keys = states[0].keys(); out = {}
    for k in keys:
        out[k] = sum((s[k].float() * (w/total) for s,w in zip(states,weights)))
    return out


def evaluate_model(model, frame, tokenizer):
    loader = DataLoader(TextDataset(frame, tokenizer), batch_size=CFG.batch_size)
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for batch in loader:
            labels=batch["labels"].numpy().tolist()
            x={k:v.to(CFG.device) for k,v in batch.items() if k != "labels"}
            pred=model(**x).logits.argmax(-1).cpu().numpy().tolist()
            y_true.extend(labels); y_pred.extend(pred)
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    return {"accuracy":accuracy_score(y_true,y_pred),"macro_precision":p,"macro_recall":r,"macro_f1":f}
