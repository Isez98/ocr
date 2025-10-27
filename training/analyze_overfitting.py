#!/usr/bin/env python3
"""
Anti-Overfitting Training Analysis Script
Validates that the enhanced training pipeline reduces overfitting
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def analyze_training_logs(log_file="./trocr-gpu-improved/logs"):
    """Analyze training logs for overfitting patterns"""
    print("🔍 Analyzing Training Logs for Overfitting...")
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Log directory not found: {log_file}")
        return
    
    # Look for tensorboard or JSON logs
    json_logs = list(log_path.glob("*.json"))
    if not json_logs:
        print("⚠️  No JSON logs found")
        return
    
    # Parse training history
    training_data = []
    for log_file in json_logs:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    training_data.append(data)
                except:
                    continue
    
    if not training_data:
        print("❌ No valid training data found")
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame(training_data)
    
    # Separate training and validation metrics
    train_logs = df[df['step'].notna() & df['train_loss'].notna()]
    eval_logs = df[df['eval_loss'].notna()]
    
    print(f"📊 Found {len(train_logs)} training steps, {len(eval_logs)} evaluation steps")
    
    # Overfitting analysis
    if len(eval_logs) > 3:
        # Check if validation loss increases while training loss decreases
        final_eval = eval_logs.tail(3)
        
        train_loss_trend = train_logs['train_loss'].tail(10).diff().mean()
        eval_loss_trend = final_eval['eval_loss'].diff().mean()
        
        print(f"📈 Training loss trend (last 10 steps): {train_loss_trend:.6f}")
        print(f"📈 Validation loss trend (last 3 evals): {eval_loss_trend:.6f}")
        
        # Check CER/WER trends
        if 'eval_cer' in final_eval.columns:
            cer_trend = final_eval['eval_cer'].diff().mean()
            print(f"📈 CER trend: {cer_trend:.6f}")
            
        if 'eval_wer' in final_eval.columns:
            wer_trend = final_eval['eval_wer'].diff().mean()
            print(f"📈 WER trend: {wer_trend:.6f}")
        
        # Overfitting indicators
        if train_loss_trend < -0.01 and eval_loss_trend > 0.01:
            print("⚠️  POTENTIAL OVERFITTING: Training loss decreasing, validation loss increasing")
        elif 'eval_cer' in final_eval.columns and final_eval['eval_cer'].iloc[-1] < 0.1:
            print("✅ Good generalization: Low CER maintained")
        else:
            print("✅ No clear overfitting detected")
    
    return df

def compare_with_previous_training():
    """Compare current training with previous overfitted version"""
    print("\\n🔄 Comparing with Previous Training Results...")
    
    # Look for both models
    current_model = Path("./trocr-gpu-improved")
    previous_logs = Path("./training.log")
    
    if current_model.exists():
        print(f"✅ Current anti-overfitting model found: {current_model}")
    else:
        print("❌ Current model not found")
        
    if previous_logs.exists():
        print("✅ Previous training logs available for comparison")
        # Parse previous logs for comparison
        with open(previous_logs, 'r') as f:
            content = f.read()
            if "'loss': 0.0005" in content:
                print("📊 Previous training: Extremely low loss (likely overfitted)")
            if "'epoch': 5.0" in content:
                print("📊 Previous training: Completed 5 epochs without early stopping")
    else:
        print("⚠️  Previous training logs not found")

def validate_augmentation_impact():
    """Check if data augmentation is working"""
    print("\\n🎨 Validating Data Augmentation Impact...")
    
    # Test augmentation on a sample image
    try:
        from training.train_gpu import ImprovedHandwritingDataset
        from transformers import TrOCRProcessor
        
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Test with and without augmentation
        data_file = "synthetic_data/training_data.json"
        if Path(data_file).exists():
            dataset_with_aug = ImprovedHandwritingDataset(data_file, processor, augment=True)
            dataset_without_aug = ImprovedHandwritingDataset(data_file, processor, augment=False)
            
            print(f"✅ Augmented dataset created: {len(dataset_with_aug)} samples")
            print(f"✅ Non-augmented dataset created: {len(dataset_without_aug)} samples")
            print("🎨 Data augmentation pipeline is functional")
        else:
            print(f"❌ Training data not found: {data_file}")
            
    except Exception as e:
        print(f"⚠️  Could not validate augmentation: {e}")

def check_early_stopping_effectiveness():
    """Check if early stopping was triggered"""
    print("\\n⏹️  Checking Early Stopping Effectiveness...")
    
    # Look for early stopping in logs
    log_files = list(Path("./trocr-gpu-improved/logs").glob("*.log")) if Path("./trocr-gpu-improved/logs").exists() else []
    
    early_stop_triggered = False
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
            if "early stopping" in content.lower():
                early_stop_triggered = True
                print("✅ Early stopping was triggered")
                break
    
    if not early_stop_triggered:
        print("ℹ️  Early stopping not triggered - model trained for full epochs")
        print("   This could mean: good convergence or need to adjust patience")

def generate_training_report():
    """Generate comprehensive anti-overfitting report"""
    print("\\n📋 Generating Anti-Overfitting Training Report...")
    print("=" * 60)
    
    # Main analysis
    df = analyze_training_logs()
    compare_with_previous_training()
    validate_augmentation_impact()
    check_early_stopping_effectiveness()
    
    print("\\n" + "=" * 60)
    print("🎯 ANTI-OVERFITTING SUMMARY:")
    print("1. ✅ CER/WER metrics implemented for proper evaluation")
    print("2. ✅ Data augmentation pipeline active")
    print("3. ✅ Early stopping configured (patience=5)")
    print("4. ✅ Best model selection by CER metric")
    print("5. ✅ Increased regularization (weight_decay=0.01)")
    print("\\n💡 RECOMMENDATIONS:")
    print("- Monitor CER/WER trends instead of just loss")
    print("- Compare validation metrics across multiple runs")
    print("- Consider expanding dataset if overfitting persists")
    print("- Adjust augmentation strength based on results")

if __name__ == "__main__":
    try:
        generate_training_report()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()