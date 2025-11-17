"""
Best Emotion Detection Model - Using Fine-tuned Transformers
Uses: RoBERTa-base fine-tuned on GoEmotions (SOTA approach)
"""

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import json
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'base_model': 'roberta-base',  # Best for emotion detection
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'model_save_path': 'X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/',
    'checkpoint_dir': 'X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/checkpoints/',
}

# GoEmotions 28 emotion labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    preds = (preds >= 0.5).astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    subset_accuracy = accuracy_score(labels, preds)
    
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'subset_accuracy': subset_accuracy,
        'precision': precision,
        'recall': recall
    }

# ============================================================================
# CUSTOM MODEL WITH MULTI-LABEL CLASSIFICATION
# ============================================================================

class MultiLabelModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_emotion_detector(csv_file):
    print("="*70)
    print(f"🚀 TRAINING BEST EMOTION DETECTOR")
    print(f"📊 Using: RoBERTa-base (State-of-the-art)")
    print("="*70)
    
    # Create directories
    os.makedirs(CONFIG['model_save_path'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print(f"\n📁 Loading {"X:/file/FAST_API/Isabella/backend/datasets/emotion_detection_dataset/data/full_dataset/goemotions_1.csv"}...")
    df = pd.read_csv("X:/file/FAST_API/Isabella/backend/datasets/emotion_detection_dataset/data/full_dataset/goemotions_1.csv")
    print(f"✅ Loaded {len(df)} samples")
    
    # Prepare labels (assuming columns are emotion names)
    label_columns = EMOTION_LABELS
    if not all(col in df.columns for col in label_columns):
        # If labels are in different format, adjust here
        print("⚠️  Adjusting label format...")
        # Add your label processing logic
    
    texts = df['text'].values
    labels = df[label_columns].values.astype(float)
    
    # Train-validation-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"   Train: {len(X_train)}")
    print(f"   Validation: {len(X_val)}")
    print(f"   Test: {len(X_test)}")
    
    # ========================================================================
    # 2. INITIALIZE TOKENIZER AND MODEL
    # ========================================================================
    print(f"\n🤖 Loading {CONFIG['base_model']} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
    model = MultiLabelModel(CONFIG['base_model'], num_labels=len(EMOTION_LABELS))
    
    print("✅ Model loaded!")
    
    # ========================================================================
    # 3. CREATE DATASETS
    # ========================================================================
    print("\n📦 Creating datasets...")
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset = EmotionDataset(X_val, y_val, tokenizer, CONFIG['max_length'])
    test_dataset = EmotionDataset(X_test, y_test, tokenizer, CONFIG['max_length'])
    print("✅ Datasets created!")
    
    # ========================================================================
    # 4. TRAINING ARGUMENTS
    # ========================================================================
    training_args = TrainingArguments(
        output_dir=CONFIG['checkpoint_dir'],
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        warmup_steps=CONFIG['warmup_steps'],
        weight_decay=CONFIG['weight_decay'],
        learning_rate=CONFIG['learning_rate'],
        logging_dir=os.path.join(CONFIG['checkpoint_dir'], 'logs'),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=4,
        report_to="none"
    )
    
    # ========================================================================
    # 5. TRAINER
    # ========================================================================
    print("\n🎓 Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # ========================================================================
    # 6. TRAIN
    # ========================================================================
    print("\n🔥 Starting training...")
    print("   (This will take 30-60 minutes depending on your hardware)")
    trainer.train()
    print("✅ Training completed!")
    
    # ========================================================================
    # 7. EVALUATE ON TEST SET
    # ========================================================================
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    print("\n" + "="*70)
    print("📊 FINAL TEST RESULTS")
    print("="*70)
    print(f"🎯 Subset Accuracy: {test_results['eval_subset_accuracy']:.4f}")
    print(f"📈 Macro F1: {test_results['eval_macro_f1']:.4f}")
    print(f"📊 Micro F1: {test_results['eval_micro_f1']:.4f}")
    print(f"⚖️  Weighted F1: {test_results['eval_weighted_f1']:.4f}")
    print(f"🎯 Precision: {test_results['eval_precision']:.4f}")
    print(f"🎯 Recall: {test_results['eval_recall']:.4f}")
    print("="*70)
    
    # ========================================================================
    # 8. SAVE MODEL
    # ========================================================================
    print("\n💾 Saving model...")
    final_model_path = os.path.join(CONFIG['model_save_path'], 'Isabella_emotion_roberta')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save training history
    history = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': csv_file,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'results': test_results,
        'config': CONFIG
    }
    
    history_path = os.path.join(CONFIG['model_save_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Model saved to: {final_model_path}")
    print(f"✅ History saved to: {history_path}")
    
    print("\n" + "="*70)
    print("✅ ALL DONE! Your model is ready to use.")
    print("="*70)
    
    return trainer, test_results

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_emotions(texts, model_path=None):
    """
    Predict emotions for new texts
    
    Args:
        texts: List of texts or single text
        model_path: Path to saved model
    
    Returns:
        List of dictionaries with emotion predictions
    """
    if model_path is None:
        model_path = os.path.join(CONFIG['model_save_path'], 'Isabella_emotion_roberta')
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Handle single text
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                max_length=CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Predict
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).numpy()[0]
            
            # Get emotions above threshold
            threshold = 0.3  # Adjustable
            detected_emotions = {
                EMOTION_LABELS[i]: float(probs[i])
                for i in range(len(EMOTION_LABELS))
                if probs[i] >= threshold
            }
            
            # Sort by confidence
            detected_emotions = dict(sorted(
                detected_emotions.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            results.append({
                'text': text,
                'emotions': detected_emotions,
                'top_emotion': max(detected_emotions.items(), key=lambda x: x[1])[0] if detected_emotions else 'neutral'
            })
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Train on your dataset
    csv_file = "X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/goemotions_2.csv"
    
    trainer, results = train_emotion_detector(csv_file)
    
    # Test inference
    print("\n🧪 Testing inference...")
    test_texts = [
        "I'm so happy and excited about this!",
        "This makes me really angry and frustrated.",
        "I'm feeling a bit nervous about tomorrow."
    ]
    
    predictions = predict_emotions(test_texts)
    
    print("\n📝 Sample Predictions:")
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print(f"Top Emotion: {pred['top_emotion']}")
        print("All Emotions:", pred['emotions'])