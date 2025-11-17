import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================================

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

EMOTION_LABELS_28 = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# ============================================================================

class EmotionDetector:
    def __init__(self, model_name=MODEL_NAME):
        print(f"🚀 Loading model {model_name}...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.device == 0:
            self.model = self.model.cuda()
            self.device_str = "CUDA"
        else:
            self.device_str = "CPU"
        self.model.eval()
        print(f"✅ Model loaded on {self.device_str}")

    def predict_single(self, text: str, threshold: float = 0.3):
        inputs = self.tokenizer(
            text, max_length=512, padding='max_length', truncation=True, return_tensors='pt'
        )
        if self.device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        emotions = {EMOTION_LABELS_28[i]: float(probs[i]) for i in range(len(EMOTION_LABELS_28)) if probs[i] >= threshold}
        emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))

        top_emotion = list(emotions.keys())[0] if emotions else "neutral"
        confidence = list(emotions.values())[0] if emotions else 0.0

        return {
            'text': text,
            'top_emotion': top_emotion,
            'confidence': confidence,
            'all_emotions': emotions
        }

# ============================================================================

if __name__ == "__main__":
    detector = EmotionDetector()

    print("\n🎭 ENTER TEXT TO DETECT EMOTIONS (type 'exit' to quit)")

    while True:
        user_input = input("\nYour text: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting... Bye!")
            break

        result = detector.predict_single(user_input, threshold=0.3)

        print(f"\n📝 Text: {result['text']}")
        print(f"🎯 Top Emotion: {result['top_emotion']} ({result['confidence']:.1%})")
        print(f"📊 All Emotions: {result['all_emotions']}")
