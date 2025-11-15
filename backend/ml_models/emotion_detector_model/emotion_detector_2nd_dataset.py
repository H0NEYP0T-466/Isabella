"""
Continue training on additional datasets without losing previous knowledge
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
import torch
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - CHANGE THESE FOR EACH NEW DATASET
# ============================================================================

PREVIOUS_MODEL_PATH = "X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/Isabella_emotion_distilbert_svm.pkl"
NEW_DATASET_PATH = "X:/file/FAST_API/Isabella/backend/datasets/emotion_detection_dataset/data/full_dataset/goemotions_2.csv"  # ← CHANGE THIS
DATASET_NAME = "goemotions_2.csv"  # ← CHANGE THIS
OUTPUT_DIR = "X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model"

emotion_cols = ['admiration','amusement','anger','annoyance','approval','caring',
                'confusion','curiosity','desire','disappointment','disapproval',
                'disgust','embarrassment','excitement','fear','gratitude','grief',
                'joy','love','nervousness','optimism','pride','realization',
                'relief','remorse','sadness','surprise','neutral']

# ============================================================================

print("="*70)
print(f"🔄 CONTINUING TRAINING ON: {DATASET_NAME}")
print("="*70)

# 1. Load previous model
print("\n📦 Loading previous model...")
model_package = joblib.load(PREVIOUS_MODEL_PATH)
tokenizer = model_package['tokenizer']
bert_model = model_package['bert_model']
previous_svm = model_package['svm_model']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

print("✅ Previous model loaded!")
print(f"   Previous training samples: {model_package.get('total_training_samples', 'Unknown')}")

# 2. Load new dataset
print(f"\n📁 Loading {DATASET_NAME}...")
new_data = pd.read_csv(NEW_DATASET_PATH)
X_new = new_data['text']
y_new = new_data[emotion_cols].values

print(f"✅ Loaded {len(X_new)} new samples")

# 3. Split new data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

print(f"   Train: {len(X_train_new)}")
print(f"   Test: {len(X_test_new)}")

# 4. Convert new data to embeddings
def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    texts_list = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
    
    for i in tqdm(range(0, len(texts_list), batch_size), desc="Creating embeddings"):
        batch = texts_list[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=128)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

print("\n🔄 Converting new dataset to embeddings...")
X_train_new_emb = get_embeddings(X_train_new.reset_index(drop=True))
X_test_new_emb = get_embeddings(X_test_new.reset_index(drop=True))

print(f"✅ New train embeddings shape: {X_train_new_emb.shape}")
print(f"✅ New test embeddings shape: {X_test_new_emb.shape}")

# 5. Load previous training data if saved
cumulative_data_path = f"{OUTPUT_DIR}/cumulative_training_data.npz"
try:
    print("\n🔗 Loading previous training data...")
    prev_data = np.load(cumulative_data_path)
    X_train_prev = prev_data['embeddings']
    y_train_prev = prev_data['labels']
    print(f"✅ Loaded {len(X_train_prev)} previous training samples")
except Exception as e:
    print(f"⚠️  No previous training data found: {e}")
    print("   Using only current model knowledge.")
    X_train_prev = np.empty((0, 768))
    y_train_prev = np.empty((0, 28))

# 6. Combine old + new data
print("\n🔗 Combining previous + new training data...")
if len(X_train_prev) > 0:
    X_train_combined = np.vstack([X_train_prev, X_train_new_emb])
    y_train_combined = np.vstack([y_train_prev, y_train_new])  # ← FIXED: was y_train_new_emb
else:
    X_train_combined = X_train_new_emb
    y_train_combined = y_train_new

print(f"✅ Combined training set: {len(X_train_combined)} samples")
print(f"   Previous: {len(X_train_prev)}")
print(f"   New: {len(X_train_new_emb)}")

# 7. Retrain model on combined data
print("\n🎓 Retraining model on combined data...")
print("   (This may take several minutes...)")

new_svm = OneVsRestClassifier(
    LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=42),
    n_jobs=-1
)

new_svm.fit(X_train_combined, y_train_combined)
print("✅ Training completed!")

# 8. Evaluate
print("\n📊 Evaluating on new dataset test set...")
y_pred = new_svm.predict(X_test_new_emb)

metrics = {
    'subset_accuracy': accuracy_score(y_test_new, y_pred),
    'hamming_loss': hamming_loss(y_test_new, y_pred),
    'macro_f1': f1_score(y_test_new, y_pred, average='macro', zero_division=0),
    'micro_f1': f1_score(y_test_new, y_pred, average='micro', zero_division=0),
    'weighted_f1': f1_score(y_test_new, y_pred, average='weighted', zero_division=0),
    'macro_precision': precision_score(y_test_new, y_pred, average='macro', zero_division=0),
    'macro_recall': recall_score(y_test_new, y_pred, average='macro', zero_division=0)
}

print("="*70)
print("📊 RESULTS")
print("="*70)
print(f"🎯 Subset Accuracy: {metrics['subset_accuracy']:.4f}")
print(f"📉 Hamming Loss: {metrics['hamming_loss']:.4f}")
print(f"📈 Macro F1: {metrics['macro_f1']:.4f}")
print(f"📊 Micro F1: {metrics['micro_f1']:.4f}")
print(f"⚖️  Weighted F1: {metrics['weighted_f1']:.4f}")
print(f"🎯 Macro Precision: {metrics['macro_precision']:.4f}")
print(f"🎯 Macro Recall: {metrics['macro_recall']:.4f}")
print("="*70)

# 9. Save updated model
print("\n💾 Saving updated model...")

# Save cumulative training data
np.savez_compressed(
    cumulative_data_path,
    embeddings=X_train_combined,
    labels=y_train_combined
)
print(f"✅ Saved cumulative training data ({len(X_train_combined)} samples)")

# Load previous history
history_path = f"{OUTPUT_DIR}/training_history.json"
try:
    with open(history_path, 'r') as f:
        training_history = json.load(f)
except:
    training_history = []

# Add new record
training_history.append({
    'dataset_name': DATASET_NAME,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_samples': len(X_train_new),
    'cumulative_samples': len(X_train_combined),
    'metrics': {k: float(v) for k, v in metrics.items()}
})

# Save updated package
model_package = {
    'tokenizer': tokenizer,
    'svm_model': new_svm,
    'bert_model': bert_model,
    'emotion_labels': emotion_cols,
    'device': str(device),
    'training_history': training_history,
    'total_training_samples': len(X_train_combined)
}

joblib.dump(model_package, PREVIOUS_MODEL_PATH, compress=3)
print(f"✅ Saved main model: {PREVIOUS_MODEL_PATH}")

# Save checkpoint
checkpoint_path = f"{OUTPUT_DIR}/Isabella_emotion_checkpoint_{DATASET_NAME}.pkl"
joblib.dump(model_package, checkpoint_path, compress=3)
print(f"✅ Saved checkpoint: {checkpoint_path}")

# Save history
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"✅ Saved training history: {history_path}")

# 10. Print training history
print("\n" + "="*70)
print("📜 TRAINING HISTORY")
print("="*70)
for i, record in enumerate(training_history, 1):
    print(f"\n{i}. {record['dataset_name']} ({record['timestamp']})")
    print(f"   Train samples: {record['train_samples']}")
    print(f"   Cumulative samples: {record['cumulative_samples']}")
    print(f"   Macro F1: {record['metrics']['macro_f1']:.4f}")
    print(f"   Micro F1: {record['metrics']['micro_f1']:.4f}")

print("\n" + "="*70)
print("✅ Training completed! Ready for next dataset.")
print("="*70)