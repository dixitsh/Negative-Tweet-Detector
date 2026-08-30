from flask import Flask, render_template_string, request, jsonify
import torch
from model import build_model, tokenizer, load_adapter_state
from config import CFG, ID2LABEL
from data import normalize_text

app=Flask(__name__)
tok=tokenizer(); model=build_model(True).to(CFG.device)
try: load_adapter_state(model,torch.load(f"{CFG.output_dir}/global_adapter.pt",map_location=CFG.device))
except FileNotFoundError: pass
model.eval()

HTML='''<!doctype html><html><head><title>FedHinglish-X</title><style>body{font-family:Arial;max-width:800px;margin:40px auto;padding:20px}textarea{width:100%;height:120px}button{padding:10px 18px;margin-top:10px}.result{margin-top:25px;padding:20px;background:#f3f3f3}</style></head><body><h1>FedHinglish-X</h1><p>Privacy-preserving personalized federated transformer sentiment demo.</p><form method="post"><textarea name="text" placeholder="Type Hinglish text...">{{text}}</textarea><br><button>Predict</button></form>{% if result %}<div class="result"><h2>{{result.label}}</h2>{% for k,v in result.probabilities.items() %}<p>{{k}}: {{'%.2f'|format(v*100)}}%</p>{% endfor %}</div>{% endif %}</body></html>'''

def predict(text):
    enc=tok(normalize_text(text),return_tensors="pt",truncation=True,padding=True,max_length=CFG.max_length)
    enc={k:v.to(CFG.device) for k,v in enc.items()}
    with torch.no_grad(): probs=torch.softmax(model(**enc).logits,dim=-1)[0].cpu().tolist()
    idx=int(torch.tensor(probs).argmax()); return {"label":ID2LABEL[idx],"probabilities":{ID2LABEL[i]:p for i,p in enumerate(probs)}}

@app.route('/',methods=['GET','POST'])
def home():
    result=None; text=''
    if request.method=='POST': text=request.form.get('text',''); result=predict(text) if text else None
    return render_template_string(HTML,text=text,result=result)

@app.post('/api/predict')
def api_predict():
    payload=request.get_json(force=True); text=payload.get('text','')
    if not text: return jsonify({'error':'text is required'}),400
    return jsonify(predict(text))

if __name__=='__main__': app.run(debug=False)
