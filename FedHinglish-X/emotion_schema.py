"""Fine-grained Hinglish emotion taxonomy and label utilities.

The hierarchy separates intensity (e.g. happy -> very_happy) from emotion family.
Use the fine-grained labels only when the training dataset actually contains them;
do not synthesize labels from sentiment scores for evaluation.
"""
from typing import Dict, List

EMOTION_LABELS: List[str] = [
    "very_happy", "happy", "sad", "very_sad", "angry", "very_angry",
    "fear", "disgust", "surprise", "love", "admiration", "contempt",
    "neutral", "mixed"
]

EMOTION_TO_SENTIMENT: Dict[str, str] = {
    "very_happy": "positive", "happy": "positive", "love": "positive",
    "admiration": "positive", "surprise": "neutral",
    "neutral": "neutral", "mixed": "mixed",
    "sad": "negative", "very_sad": "negative", "angry": "negative",
    "very_angry": "negative", "fear": "negative", "disgust": "negative",
    "contempt": "negative",
}

# Optional weak-label aliases for dataset files that use different names.
ALIASES = {
    "joy": "happy", "happiness": "happy", "sadness": "sad",
    "anger": "angry", "rage": "very_angry", "fury": "very_angry",
    "excited": "very_happy", "excitement": "very_happy",
    "love": "love", "loved": "love", "afraid": "fear",
}


def normalize_emotion_label(label: str) -> str:
    value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(value, value)


def emotion_description(label: str) -> str:
    descriptions = {
        "very_happy": "very strong joy, excitement or celebration",
        "happy": "joy, satisfaction or positive mood",
        "sad": "sadness, disappointment or regret",
        "very_sad": "intense sadness, grief or despair",
        "angry": "anger, frustration or irritation",
        "very_angry": "intense anger, rage or strong outrage",
        "fear": "fear, worry or anxiety",
        "disgust": "disgust or strong aversion",
        "surprise": "unexpectedness, shock or astonishment",
        "love": "affection or love",
        "admiration": "respect, praise or admiration",
        "contempt": "disrespect or contempt",
        "neutral": "factual or emotionally neutral language",
        "mixed": "more than one strong emotion expressed together",
    }
    return descriptions.get(label, "unknown")
