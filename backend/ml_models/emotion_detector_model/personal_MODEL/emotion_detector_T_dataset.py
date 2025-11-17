import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression  # ✅ CHANGED FROM LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, classification_report
)
import seaborn as sns
import joblib
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Emotion Detection Training with DistilBERT + Logistic Regression")
print("=" * 70)

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
print("\n📁 Loading dataset...")
dataset = pd.read_csv("X:/file/FAST_API/Isabella/backend/datasets/emotion_detection_dataset/go_emotions_dataset.csv")

print("✅ Dataset loaded successfully!")
print(f"Shape: {dataset.shape}")
print(f"Columns: {dataset.columns.tolist()}\n")

# ---------------------------------------------------------------
# 2. Prepare features and labels (MULTI-LABEL)
# ---------------------------------------------------------------
print("📊 Preparing features and labels...")

# Emotion columns
emotion_cols = ['admiration','amusement','anger','annoyance','approval','caring',
                'confusion','curiosity','desire','disappointment','disapproval',
                'disgust','embarrassment','excitement','fear','gratitude','grief',
                'joy','love','nervousness','optimism','pride','realization',
                'relief','remorse','sadness','surprise','neutral']

X = dataset['text']
y = dataset[emotion_cols].values  

print(f"✅ Features: {X.shape[0]} texts")
print(f"✅ Labels: {y.shape} (multi-label binary matrix)")
print(f"   Average emotions per text: {y.sum(axis=1).mean():.2f}")

# ---------------------------------------------------------------
# 3. Split into train and test sets
# ---------------------------------------------------------------
print("\n✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)
print(f"✅ Train set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")

# ---------------------------------------------------------------
# 4. Load DistilBERT tokenizer and model
# ---------------------------------------------------------------
print("\n🤖 Loading DistilBERT model...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

bert_model.to(device)
bert_model.eval()

# ---------------------------------------------------------------
# 5. Function to convert text to embeddings (optimized)
# ---------------------------------------------------------------
def get_embeddings(texts, tokenizer, model, device, batch_size=32):
    """Extract DistilBERT embeddings from texts"""
    all_embeddings = []
    texts_list = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
    
    for i in tqdm(range(0, len(texts_list), batch_size), desc="Creating embeddings"):
        batch = texts_list[i:i+batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=128
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token representation (first token)
            embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

# ---------------------------------------------------------------
# 6. Get embeddings for train and test
# ---------------------------------------------------------------
print("\n🔄 Converting texts to embeddings...")
X_train_emb = get_embeddings(X_train.reset_index(drop=True), tokenizer, bert_model, device, batch_size=16)
X_test_emb = get_embeddings(X_test.reset_index(drop=True), tokenizer, bert_model, device, batch_size=16)

print(f"✅ Train embeddings shape: {X_train_emb.shape}")
print(f"✅ Test embeddings shape: {X_test_emb.shape}")

# ---------------------------------------------------------------
# 7. Train Multi-Label Logistic Regression Model
# ---------------------------------------------------------------
print("\n🎓 Training Multi-Label Logistic Regression model...")
print("   (This may take several minutes...)")

# ✅ LOGISTIC REGRESSION WITH BEST PARAMETERS
model = OneVsRestClassifier(
    LogisticRegression(
        C=1.0,                      # Regularization strength
        max_iter=1000,              # Maximum iterations
        class_weight='balanced',    # Handle class imbalance
        solver='saga',              # Best solver for large datasets
        penalty='l2',               # L2 regularization
        random_state=42,
        n_jobs=1,                   # Jobs per estimator
        verbose=1
    ),
    n_jobs=-1  # Use all CPU cores for parallel training
)   

model.fit(X_train_emb, y_train)
print("✅ Model training completed!")

# ---------------------------------------------------------------
# 8. Predictions with Probability Threshold (Logistic Regression Advantage!)
# ---------------------------------------------------------------
print("\n🔮 Making predictions...")

# ✅ Get probabilities (advantage of Logistic Regression over SVM!)
y_pred_proba = model.predict_proba(X_test_emb)

# Apply threshold (default 0.5, but can be tuned)
threshold = 0.5
y_pred = (np.array(y_pred_proba).T >= threshold).astype(int).T

print("✅ Predictions completed!")
print(f"   Using probability threshold: {threshold}")

# ---------------------------------------------------------------
# 9. Multi-Label Evaluation Metrics
# ---------------------------------------------------------------
print("\n" + "="*70)
print("📊 MODEL PERFORMANCE METRICS")
print("="*70)

# Subset accuracy (exact match)
subset_accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Subset Accuracy (Exact Match): {subset_accuracy:.4f}")

# Hamming Loss (fraction of wrong labels)
h_loss = hamming_loss(y_test, y_pred)
print(f"📉 Hamming Loss: {h_loss:.4f}")

# Macro-averaged metrics
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"\n📈 Macro-Averaged Metrics:")
print(f"   Precision: {precision_macro:.4f}")
print(f"   Recall:    {recall_macro:.4f}")
print(f"   F1-Score:  {f1_macro:.4f}")

# Micro-averaged metrics
precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

print(f"\n📊 Micro-Averaged Metrics:")
print(f"   Precision: {precision_micro:.4f}")
print(f"   Recall:    {recall_micro:.4f}")
print(f"   F1-Score:  {f1_micro:.4f}")

# Weighted metrics
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n⚖️ Weighted Metrics:")
print(f"   Precision: {precision_weighted:.4f}")
print(f"   Recall:    {recall_weighted:.4f}")
print(f"   F1-Score:  {f1_weighted:.4f}")

# ✅ Comparison with previous SVM results
print("\n" + "="*70)
print("📊 COMPARISON: Logistic Regression vs Your Previous SVM")
print("="*70)
print(f"Metric              SVM        Logistic Reg    Improvement")
print(f"Macro F1:          0.2059      {f1_macro:.4f}         {((f1_macro/0.2059 - 1)*100):+.1f}%")
print(f"Micro F1:          0.2273      {f1_micro:.4f}         {((f1_micro/0.2273 - 1)*100):+.1f}%")
print(f"Precision:         0.1292      {precision_macro:.4f}         {((precision_macro/0.1292 - 1)*100):+.1f}%")
print(f"Subset Accuracy:   0.0080      {subset_accuracy:.4f}         {((subset_accuracy/0.0080 - 1)*100):+.1f}%")
print("="*70)

# ---------------------------------------------------------------
# 10. Per-Emotion Performance
# ---------------------------------------------------------------
print("\n" + "="*70)
print("🎭 PER-EMOTION PERFORMANCE")
print("="*70)

per_emotion_metrics = []
for idx, emotion in enumerate(emotion_cols):
    precision = precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
    recall = recall_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
    f1 = f1_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
    support = y_test[:, idx].sum()
    
    per_emotion_metrics.append({
        'emotion': emotion,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    })

# Sort by F1 score
per_emotion_df = pd.DataFrame(per_emotion_metrics).sort_values('f1', ascending=False)

print("\n🏆 Top 10 Best Performing Emotions:")
print(per_emotion_df.head(10).to_string(index=False))

print("\n⚠️ Bottom 10 Performing Emotions:")
print(per_emotion_df.tail(10).to_string(index=False))

# ---------------------------------------------------------------
# 11. Sample Predictions with Probabilities
# ---------------------------------------------------------------
print("\n" + "="*70)
print("🔍 SAMPLE PREDICTIONS (with confidence scores)")
print("="*70)

sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    text = X_test.iloc[idx]
    true_emotions = [emotion_cols[i] for i, val in enumerate(y_test[idx]) if val == 1]
    
    # ✅ Get predicted emotions with confidence scores
    pred_emotions_with_conf = [
        (emotion_cols[i], y_pred_proba[idx][i]) 
        for i, val in enumerate(y_pred[idx]) if val == 1
    ]
    pred_emotions_with_conf.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📝 Text: {text[:100]}...")
    print(f"✅ True emotions: {', '.join(true_emotions) if true_emotions else 'None'}")
    if pred_emotions_with_conf:
        print(f"🔮 Predicted emotions:")
        for emotion, conf in pred_emotions_with_conf:
            print(f"   • {emotion}: {conf:.1%}")
    else:
        print(f"🔮 Predicted emotions: None")

# ---------------------------------------------------------------
# 12. Visualization: Confusion Matrix for Top 5 Emotions
# ---------------------------------------------------------------
print("\n📊 Creating visualizations...")

# Get top 5 most frequent emotions
emotion_frequencies = y_test.sum(axis=0)
top_5_indices = np.argsort(emotion_frequencies)[-5:][::-1]
top_5_emotions = [emotion_cols[i] for i in top_5_indices]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, emotion_idx in enumerate(top_5_indices):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test[:, emotion_idx], y_pred[:, emotion_idx])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Not Present', 'Present'],
                yticklabels=['Not Present', 'Present'])
    axes[i].set_title(f'{emotion_cols[emotion_idx].capitalize()}\n(Support: {int(emotion_frequencies[emotion_idx])})')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')

# Remove extra subplot
axes[-1].remove()

plt.tight_layout()
plt.savefig('X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/confusion_matrices_lr.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Confusion matrices saved!")

# ---------------------------------------------------------------
# 13. Performance Comparison Bar Chart
# ---------------------------------------------------------------
plt.figure(figsize=(14, 8))
per_emotion_df_sorted = per_emotion_df.sort_values('f1', ascending=True)
colors = ['red' if x < 0.3 else 'orange' if x < 0.6 else 'green' for x in per_emotion_df_sorted['f1']]

plt.barh(per_emotion_df_sorted['emotion'], per_emotion_df_sorted['f1'], color=colors)
plt.xlabel('F1-Score', fontsize=12)
plt.ylabel('Emotion', fontsize=12)
plt.title('Per-Emotion F1-Score Performance (Logistic Regression)', fontsize=14, fontweight='bold')
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
plt.legend()
plt.tight_layout()
plt.savefig('X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/per_emotion_performance_lr.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Performance chart saved!")

# ---------------------------------------------------------------
# 14. Save Model + Tokenizer + Embeddings Info
# ---------------------------------------------------------------
print("\n💾 Saving model and components...")

model_package = {
    'tokenizer': tokenizer,
    'lr_model': model,  # ✅ CHANGED NAME
    'bert_model': bert_model,
    'emotion_labels': emotion_cols,
    'device': str(device),
    'threshold': threshold,  # ✅ Added threshold
    'metrics': {
        'subset_accuracy': float(subset_accuracy),
        'hamming_loss': float(h_loss),
        'macro_f1': float(f1_macro),
        'micro_f1': float(f1_micro),
        'weighted_f1': float(f1_weighted),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro)
    },
    'per_emotion_metrics': per_emotion_df.to_dict('records')
}

joblib.dump(
    model_package, 
    "X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/Isabella_emotion_distilbert_lr.pkl",
    compress=3
)

print("✅ Model package saved successfully!")

# ---------------------------------------------------------------
# 15. Save Detailed Metrics Report
# ---------------------------------------------------------------
metrics_report = {
    'model_type': 'DistilBERT + Multi-Label Logistic Regression',
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_size': {
        'train': len(X_train),
        'test': len(X_test),
        'total': len(dataset)
    },
    'hyperparameters': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'saga',
        'penalty': 'l2',
        'class_weight': 'balanced',
        'threshold': threshold
    },
    'overall_metrics': {
        'subset_accuracy': float(subset_accuracy),
        'hamming_loss': float(h_loss),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'micro_precision': float(precision_micro),
        'micro_recall': float(recall_micro),
        'micro_f1': float(f1_micro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted)
    },
    'per_emotion_metrics': per_emotion_df.to_dict('records')
}

import json
with open('X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/training_metrics_lr.json', 'w') as f:
    json.dump(metrics_report, f, indent=2)

print("✅ Metrics report saved!")

# ---------------------------------------------------------------
# 16. Final Summary
# ---------------------------------------------------------------
print("\n" + "="*70)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n📊 Final Results (Logistic Regression):")
print(f"   • Subset Accuracy: {subset_accuracy:.4f}")
print(f"   • Macro F1-Score: {f1_macro:.4f}")
print(f"   • Micro F1-Score: {f1_micro:.4f}")
print(f"   • Macro Precision: {precision_macro:.4f}")
print(f"   • Hamming Loss: {h_loss:.4f}")
print(f"\n💾 Saved Files:")
print(f"   • Model: Isabella_emotion_distilbert_lr.pkl")
print(f"   • Metrics: training_metrics_lr.json")
print(f"   • Visualizations: confusion_matrices_lr.png, per_emotion_performance_lr.png")
print("\n✨ Ready for deployment!")
print("="*70)