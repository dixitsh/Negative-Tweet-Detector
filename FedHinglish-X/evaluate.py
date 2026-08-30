import json, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import load_dataframe, split_dataframe
from model import build_model, tokenizer, load_adapter_state
from federated import evaluate_model
from config import CFG, ID2LABEL


def main():
    metrics_path=f"{CFG.output_dir}/metrics.json"
    if not os.path.exists(metrics_path): raise FileNotFoundError("Run train_federated.py first")
    with open(metrics_path,encoding="utf8") as f: metrics=json.load(f)
    print(json.dumps(metrics,indent=2))
    vals=[x["macro_f1"] for x in metrics["per_client"]]
    print(f"personalization gap (mean client F1 - global F1): {np.mean(vals)-metrics['global_test']['macro_f1']:.4f}")
    hist=json.load(open(f"{CFG.output_dir}/history.json",encoding="utf8"))
    plt.figure(); plt.plot([x["round"] for x in hist],[x["macro_f1"] for x in hist],marker="o"); plt.xlabel("Federated round"); plt.ylabel("Validation Macro-F1"); plt.tight_layout(); plt.savefig(f"{CFG.output_dir}/training_curve.png"); plt.close()

if __name__=='__main__': main()
