import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from config import CFG, ID2LABEL, LABEL2ID


def build_model(trainable_adapter=True):
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name, num_labels=CFG.num_labels,
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
    )
    if trainable_adapter:
        target = ["query", "key", "value"]
        config = LoraConfig(task_type=TaskType.SEQ_CLS, r=CFG.lora_r, lora_alpha=CFG.lora_alpha,
                            lora_dropout=CFG.lora_dropout, target_modules=target,
                            bias="none", modules_to_save=["classifier"])
        model = get_peft_model(model, config)
    return model


def tokenizer():
    return AutoTokenizer.from_pretrained(CFG.model_name)


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items() if "lora_" in k or "classifier" in k}


def load_adapter_state(model, state):
    current = model.state_dict()
    for k,v in state.items():
        if k in current: current[k].copy_(v.to(current[k].device))


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
