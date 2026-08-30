import os, json, random
import torch
from data import load_dataframe, split_dataframe, dirichlet_partition, seed_everything
from model import build_model, tokenizer, adapter_state, load_adapter_state
from federated import train_client, evaluate_model
from config import CFG


def blend(a,b,alpha):
    return {k:(1-alpha)*a[k].float()+alpha*b[k].float() for k in a}


def main():
    seed_everything(CFG.seed); os.makedirs(CFG.output_dir,exist_ok=True)
    df=load_dataframe(); train,val,test=split_dataframe(df); tok=tokenizer()
    clients=dirichlet_partition(train,CFG.num_clients,CFG.dirichlet_alpha,CFG.seed)
    print("client sizes",[len(x) for x in clients])
    template=build_model(True).to(CFG.device); global_state=adapter_state(template)
    client_states=[{k:v.clone() for k,v in global_state.items()} for _ in clients]
    history=[]; client_logs=[]
    for rnd in range(1,CFG.rounds+1):
        order=list(range(len(clients))); random.Random(CFG.seed+rnd).shuffle(order)
        round_logs=[]
        # Sequential updates intentionally model asynchronous arrival: each client updates the server
        # using its current local state and the server does not wait for a synchronized barrier.
        for cid in order:
            private_state,info=train_client(client_states[cid],clients[cid],tok,cid)
            # Staleness-aware server step. Larger clients get more weight, capped for stability.
            alpha=min(0.35,max(0.05,len(clients[cid])/max(1,len(train))))
            global_state=blend(global_state,private_state,alpha)
            # Personalized local state retains part of its own update plus the new global knowledge.
            client_states[cid]=blend(client_states[cid],global_state,0.25)
            client_states[cid]=blend(client_states[cid],private_state,0.75)
            info["server_alpha"]=alpha; round_logs.append(info)
        global_model=build_model(True).to(CFG.device); load_adapter_state(global_model,global_state)
        gm=evaluate_model(global_model,val,tok)
        history.append({"round":rnd,**gm}); client_logs.extend([{**x,"round":rnd} for x in round_logs])
        print({"round":rnd,**gm})
        torch.save(global_state,f"{CFG.output_dir}/global_adapter_round_{rnd}.pt")
    # Final global and personalized client evaluation
    global_model=build_model(True).to(CFG.device); load_adapter_state(global_model,global_state)
    global_test=evaluate_model(global_model,test,tok)
    per_client=[]
    for cid,frame in enumerate(clients):
        m=build_model(True).to(CFG.device); load_adapter_state(m,client_states[cid]); met=evaluate_model(m,frame,tok)
        per_client.append({"client":cid,"samples":len(frame),**met})
    metrics={"global_test":global_test,"per_client":per_client,
             "client_macro_f1_mean":sum(x["macro_f1"] for x in per_client)/len(per_client),
             "client_macro_f1_variance":float(torch.tensor([x["macro_f1"] for x in per_client]).var(unbiased=False)),
             "dp_sigma":CFG.dp_sigma,"clip_norm":CFG.clip_norm,"rounds":CFG.rounds}
    with open(f"{CFG.output_dir}/metrics.json","w",encoding="utf8") as f: json.dump(metrics,f,indent=2)
    with open(f"{CFG.output_dir}/history.json","w",encoding="utf8") as f: json.dump(history,f,indent=2)
    with open(f"{CFG.output_dir}/client_logs.json","w",encoding="utf8") as f: json.dump(client_logs,f,indent=2)
    torch.save(global_state,f"{CFG.output_dir}/global_adapter.pt")
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
