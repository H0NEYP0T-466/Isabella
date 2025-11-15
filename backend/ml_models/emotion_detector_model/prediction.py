"""
Interactive Emotion Detection with Confidence Scores
Shows which emotions have higher confidence
"""

import joblib
import torch
import numpy as np
from transformers import DistilBertTokenizerFast
import sys

class EmotionPredictor:
    def __init__(self, model_path):
        """Initialize the emotion predictor"""
        print("🔄 Loading emotion detection model...")
        print("="*70)
        
        try:
            # Load model package
            self.model_package = joblib.load(model_path)
            self.tokenizer = self.model_package['tokenizer']
            self.bert_model = self.model_package['bert_model']
            self.svm_model = self.model_package['svm_model']
            self.emotion_labels = self.model_package['emotion_labels']
            
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Device: {self.device}")
            print(f"   Total training samples: {self.model_package.get('total_training_samples', 'Unknown')}")
            
            # Print training history if available
            if 'training_history' in self.model_package:
                print(f"   Trained on {len(self.model_package['training_history'])} datasets")
                for record in self.model_package['training_history']:
                    print(f"   - {record['dataset_name']}: Macro F1 = {record['metrics']['macro_f1']:.4f}")
            
            print("="*70)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def get_embedding(self, text):
        """Convert text to DistilBERT embedding"""
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get embedding
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:,0,:].cpu().numpy()
        
        return embedding
    
    def predict(self, text, threshold=0.0):
        """
        Predict emotions for input text with confidence scores
        
        Args:
            text: Input text string
            threshold: Minimum confidence score to show emotion (default: 0.0)
        
        Returns:
            Dictionary with predicted emotions and confidence scores
        """
        # Get embedding
        embedding = self.get_embedding(text)
        
        # Get binary predictions
        prediction = self.svm_model.predict(embedding)[0]
        
        # Get decision function scores (confidence/distance from hyperplane)
        # Higher positive values = stronger confidence
        decision_scores = self.svm_model.decision_function(embedding)[0]
        
        # Normalize scores to 0-1 range using sigmoid
        confidence_scores = 1 / (1 + np.exp(-decision_scores))
        
        # Create emotion results with scores
        emotion_results = []
        for i, emotion in enumerate(self.emotion_labels):
            emotion_results.append({
                'emotion': emotion,
                'predicted': int(prediction[i]),
                'confidence': float(confidence_scores[i]),
                'raw_score': float(decision_scores[i])
            })
        
        # Sort by confidence (highest first)
        emotion_results_sorted = sorted(emotion_results, key=lambda x: x['confidence'], reverse=True)
        
        # Get only detected emotions (predicted = 1)
        detected_emotions = [
            e for e in emotion_results_sorted 
            if e['predicted'] == 1 and e['confidence'] >= threshold
        ]
        
        # Get top emotions by confidence even if not predicted
        top_emotions = emotion_results_sorted[:10]  # Top 10 by confidence
        
        return {
            'text': text,
            'detected_emotions': detected_emotions,
            'top_emotions': top_emotions,
            'all_emotions': emotion_results_sorted,
            'num_detected': len(detected_emotions)
        }
    
    def print_prediction(self, text, show_all=False, show_top=5):
        """Print prediction in a nice format with confidence scores"""
        result = self.predict(text)
        
        print("\n" + "="*70)
        print(f"📝 Text: {text}")
        print("-"*70)
        
        if result['detected_emotions']:
            print(f"🎭 Detected Emotions ({result['num_detected']}) - Sorted by Confidence:")
            print()
            for i, emotion_data in enumerate(result['detected_emotions'], 1):
                emotion = emotion_data['emotion']
                confidence = emotion_data['confidence']
                
                # Create visual bar (0-100% scale)
                bar_length = int(confidence * 40)
                bar = "█" * bar_length
                
                # Color coding
                if confidence >= 0.7:
                    indicator = "🟢"  # High confidence
                elif confidence >= 0.5:
                    indicator = "🟡"  # Medium confidence
                else:
                    indicator = "🟠"  # Low confidence
                
                print(f"   {i}. {indicator} {emotion.upper():15s} [{confidence:5.1%}] {bar}")
        else:
            print("🎭 No emotions detected")
        
        # Show top emotions by confidence
        if show_top > 0:
            print(f"\n📊 Top {show_top} Emotions by Confidence (even if not predicted):")
            print()
            for i, emotion_data in enumerate(result['top_emotions'][:show_top], 1):
                emotion = emotion_data['emotion']
                confidence = emotion_data['confidence']
                predicted = "✓" if emotion_data['predicted'] == 1 else "✗"
                
                bar_length = int(confidence * 30)
                bar = "▓" * bar_length + "░" * (30 - bar_length)
                
                print(f"   {i}. [{predicted}] {emotion:15s} {confidence:5.1%} {bar}")
        
        if show_all:
            print("\n📋 All 28 Emotions (Sorted by Confidence):")
            print()
            for i, emotion_data in enumerate(result['all_emotions'], 1):
                emotion = emotion_data['emotion']
                confidence = emotion_data['confidence']
                predicted = "✓ Detected" if emotion_data['predicted'] == 1 else "✗ Not detected"
                print(f"   {i:2d}. {emotion:15s} {confidence:5.1%} - {predicted}")
        
        print("="*70)
        
        return result
    
    def batch_predict(self, texts):
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def compare_texts(self, texts):
        """Compare emotions across multiple texts"""
        print("\n" + "="*70)
        print(f"📊 EMOTION COMPARISON ({len(texts)} texts)")
        print("="*70)
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n{i}. {text[:60]}...")
            result = self.predict(text)
            results.append(result)
            
            if result['detected_emotions']:
                top_3 = result['detected_emotions'][:3]
                for emotion_data in top_3:
                    print(f"   • {emotion_data['emotion']:12s} ({emotion_data['confidence']:.1%})")
            else:
                print("   • No emotions detected")
        
        return results


def interactive_mode(predictor):
    """Interactive mode - user enters text and gets predictions"""
    print("\n" + "="*70)
    print("🎭 INTERACTIVE EMOTION DETECTION WITH CONFIDENCE SCORES")
    print("="*70)
    print("\nCommands:")
    print("  • Type any text to detect emotions")
    print("  • Type 'examples' to see example predictions")
    print("  • Type 'compare' to compare multiple texts")
    print("  • Type 'all' to show all 28 emotions")
    print("  • Type 'quit' or 'exit' to quit")
    print("="*70)
    
    show_all_emotions = False
    
    while True:
        try:
            # Get user input
            text = input("\n💬 Enter text (or command): ").strip()
            
            # Check for exit
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Toggle show all
            if text.lower() == 'all':
                show_all_emotions = not show_all_emotions
                status = "ON" if show_all_emotions else "OFF"
                print(f"✅ Show all emotions: {status}")
                continue
            
            # Check for examples
            if text.lower() == 'examples':
                show_examples(predictor)
                continue
            
            # Check for compare mode
            if text.lower() == 'compare':
                compare_mode(predictor)
                continue
            
            # Check for empty input
            if not text:
                print("⚠️  Please enter some text!")
                continue
            
            # Predict and display
            predictor.print_prediction(text, show_all=show_all_emotions, show_top=5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def compare_mode(predictor):
    """Compare emotions in multiple texts"""
    print("\n" + "="*70)
    print("📊 COMPARISON MODE")
    print("="*70)
    print("Enter multiple texts to compare (empty line to finish):")
    
    texts = []
    i = 1
    while True:
        text = input(f"{i}. ").strip()
        if not text:
            break
        texts.append(text)
        i += 1
    
    if len(texts) < 2:
        print("⚠️  Need at least 2 texts to compare!")
        return
    
    predictor.compare_texts(texts)


def show_examples(predictor):
    """Show example predictions with confidence scores"""
    examples = [
        "I'm so excited about this amazing opportunity! This is the best day ever!",
        "I can't believe you did that. I'm really disappointed and angry.",
        "Thank you so much for your help. I really appreciate your kindness.",
        "I'm not sure what to do next. This is all very confusing to me.",
        "I'm so sorry for what happened. I feel terrible about it.",
        "Wow! I never expected this to happen. What a surprise!",
        "I'm worried something bad might happen. This makes me nervous.",
        "That's actually pretty funny! You made me laugh.",
        "I love spending time with you. You mean so much to me.",
        "This is just a normal day. Nothing special happening."
    ]
    
    print("\n" + "="*70)
    print("📚 EXAMPLE PREDICTIONS WITH CONFIDENCE SCORES")
    print("="*70)
    
    for i, text in enumerate(examples, 1):
        print(f"\n{'─'*70}")
        print(f"{i}. {text}")
        print(f"{'─'*70}")
        
        result = predictor.predict(text)
        
        if result['detected_emotions']:
            print("   Top Detected Emotions:")
            for emotion_data in result['detected_emotions'][:3]:
                confidence = emotion_data['confidence']
                bar = "█" * int(confidence * 20)
                print(f"   • {emotion_data['emotion']:12s} {confidence:5.1%} {bar}")
        else:
            print("   No emotions detected")


def main():
    """Main function"""
    print("\n" + "🎭"*35)
    print("   EMOTION DETECTION SYSTEM - With Confidence Scores")
    print("🎭"*35)
    
    # Model path
    model_path = "X:/file/FAST_API/Isabella/backend/ml_models/emotion_detector_model/Isabella_emotion_distilbert_svm.pkl"
    
    # Load model
    predictor = EmotionPredictor(model_path)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # If text provided as argument
        text = " ".join(sys.argv[1:])
        predictor.print_prediction(text, show_all=False, show_top=10)
    else:
        # Interactive mode
        interactive_mode(predictor)


if __name__ == "__main__":
    main()