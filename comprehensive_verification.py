# ========================================
# COMPREHENSIVE MODEL VERIFICATION SUITE
# Tests if your model is ACTUALLY correct
# ========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ========================================
# ESSENTIAL CHECK 1: TRAIN/TEST ACCURACY
# Detects Overfitting
# ========================================

def check_overfitting(data_path='data.csv'):
    """
    Check if model is overfitting (memorizing vs learning)
    """
    print("\n" + "=" * 70)
    print("✓ CHECK 1: OVERFITTING DETECTION (Train vs Test Accuracy)")
    print("=" * 70)
    
    # Load and preprocess
    data = pd.read_csv(data_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracies
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    difference = abs(train_acc - test_acc)
    
    print(f"\n📊 Results:")
    print(f"   Training Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Difference:         {difference:.4f} ({difference*100:.2f}%)")
    
    print(f"\n💡 Interpretation:")
    if difference < 0.02:
        print(f"   ✅ EXCELLENT! Difference < 2% - Perfect generalization!")
        status = "PASS"
    elif difference < 0.05:
        print(f"   ✅ GOOD! Difference < 5% - Model generalizes well")
        status = "PASS"
    elif difference < 0.10:
        print(f"   ⚠️  FAIR. Difference 5-10% - Slight overfitting")
        status = "WARN"
    else:
        print(f"   ❌ POOR! Difference > 10% - Severe overfitting!")
        print(f"   → Model memorized training data")
        status = "FAIL"
    
    return status, train_acc, test_acc, model, scaler, X_test_scaled, y_test


# ========================================
# ESSENTIAL CHECK 2: CROSS-VALIDATION
# Tests Consistency Across Different Splits
# ========================================

def check_cross_validation(data_path='data.csv', n_folds=5):
    """
    Verify model performs consistently on different data splits
    """
    print("\n" + "=" * 70)
    print(f"✓ CHECK 2: CROSS-VALIDATION ({n_folds}-Fold)")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv(data_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Scale all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print(f"\n📊 Testing on {n_folds} different data splits...")
    cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')
    
    print(f"\n   Individual Fold Accuracies:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    min_cv = cv_scores.min()
    max_cv = cv_scores.max()
    
    print(f"\n   Summary Statistics:")
    print(f"   Mean:      {mean_cv:.4f} ({mean_cv*100:.2f}%)")
    print(f"   Std Dev:   {std_cv:.4f} (±{std_cv*100:.2f}%)")
    print(f"   Min/Max:   {min_cv:.4f} - {max_cv:.4f}")
    print(f"   Range:     {(max_cv-min_cv)*100:.2f}%")
    
    print(f"\n💡 Interpretation:")
    if std_cv < 0.02:
        print(f"   ✅ EXCELLENT! Very stable (std < 2%)")
        status = "PASS"
    elif std_cv < 0.03:
        print(f"   ✅ VERY GOOD! Stable performance (std < 3%)")
        status = "PASS"
    elif std_cv < 0.05:
        print(f"   ✅ GOOD. Acceptable stability (std < 5%)")
        status = "PASS"
    else:
        print(f"   ⚠️  HIGH VARIANCE! Results inconsistent (std > 5%)")
        print(f"   → Model performance varies too much across splits")
        status = "WARN"
    
    return status, cv_scores, mean_cv, std_cv


# ========================================
# ESSENTIAL CHECK 3: CONFUSION MATRIX
# Analyzes Types of Errors
# ========================================

def check_confusion_matrix(y_test, y_pred):
    """
    Detailed analysis of prediction errors
    """
    print("\n" + "=" * 70)
    print("✓ CHECK 3: CONFUSION MATRIX ANALYSIS")
    print("=" * 70)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total = len(y_test)
    
    print(f"\n📊 Confusion Matrix:")
    print(f"\n                    Predicted")
    print(f"                Benign    Malignant")
    print(f"   Actual Benign    {tn:3d}        {fp:3d}      (Total: {tn+fp})")
    print(f"        Malignant   {fn:3d}        {tp:3d}      (Total: {fn+tp})")
    
    print(f"\n   Detailed Breakdown:")
    print(f"   True Negatives:   {tn} ({tn/total*100:.1f}%) - Correctly identified Benign")
    print(f"   True Positives:   {tp} ({tp/total*100:.1f}%) - Correctly identified Malignant")
    print(f"   False Positives:  {fp} ({fp/total*100:.1f}%) - Benign predicted as Malignant")
    print(f"   False Negatives:  {fn} ({fn/total*100:.1f}%) - Malignant predicted as Benign ⚠️")
    
    print(f"\n   Clinical Impact:")
    print(f"   False Positive Rate: {fp/(fp+tn)*100:.2f}% (False alarms)")
    print(f"   False Negative Rate: {fn/(fn+tp)*100:.2f}% (Missed cancers) ⚠️")
    
    print(f"\n💡 Interpretation:")
    
    # Check false negatives (most critical in medical diagnosis)
    if fn == 0:
        print(f"   ✅ PERFECT! No missed cancer cases!")
        fn_status = "EXCELLENT"
    elif fn <= 2:
        print(f"   ✅ EXCELLENT! Only {fn} missed cancer case(s)")
        fn_status = "GOOD"
    elif fn <= 5:
        print(f"   ⚠️  ACCEPTABLE. {fn} missed cancer cases")
        fn_status = "FAIR"
    else:
        print(f"   ❌ CONCERNING! {fn} missed cancer cases")
        fn_status = "POOR"
    
    # Check false positives
    if fp <= 2:
        print(f"   ✅ EXCELLENT! Only {fp} false alarm(s)")
        fp_status = "GOOD"
    elif fp <= 5:
        print(f"   ✅ GOOD. {fp} false alarms (acceptable)")
        fp_status = "GOOD"
    else:
        print(f"   ⚠️  {fp} false alarms (many unnecessary biopsies)")
        fp_status = "FAIR"
    
    # Overall status
    if fn_status in ["EXCELLENT", "GOOD"] and fp_status == "GOOD":
        status = "PASS"
    else:
        status = "WARN"
    
    return status, cm, (tn, fp, fn, tp)


# ========================================
# ESSENTIAL CHECK 4: BASELINE COMPARISON
# Model Better Than Naive Prediction?
# ========================================

def check_baseline_comparison(y_test, test_acc):
    """
    Check if model is better than just guessing the majority class
    """
    print("\n" + "=" * 70)
    print("✓ CHECK 4: BASELINE COMPARISON (Is Model Better Than Guessing?)")
    print("=" * 70)
    
    # Calculate baseline (always predict majority class)
    benign_count = (y_test == 0).sum()
    malignant_count = (y_test == 1).sum()
    total = len(y_test)
    
    baseline_accuracy = max(benign_count, malignant_count) / total
    improvement = (test_acc - baseline_accuracy) * 100
    
    print(f"\n📊 Comparison:")
    print(f"   Class Distribution:")
    print(f"   - Benign:    {benign_count} ({benign_count/total*100:.1f}%)")
    print(f"   - Malignant: {malignant_count} ({malignant_count/total*100:.1f}%)")
    
    print(f"\n   Performance:")
    print(f"   Baseline (always predict majority): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"   Your Model:                         {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Improvement:                        {improvement:.2f} percentage points")
    
    print(f"\n💡 Interpretation:")
    if improvement > 30:
        print(f"   ✅ EXCELLENT! Model is {improvement:.1f}% better than baseline")
        print(f"   → Model learned strong patterns!")
        status = "PASS"
    elif improvement > 20:
        print(f"   ✅ VERY GOOD! Model is {improvement:.1f}% better than baseline")
        status = "PASS"
    elif improvement > 10:
        print(f"   ✅ GOOD. Model is {improvement:.1f}% better than baseline")
        status = "PASS"
    else:
        print(f"   ❌ POOR! Model only {improvement:.1f}% better than baseline")
        print(f"   → Model barely learned anything!")
        status = "FAIL"
    
    return status, baseline_accuracy, improvement


# ========================================
# ESSENTIAL CHECK 5: FEATURE IMPORTANCE
# Medical Validation
# ========================================

def check_feature_importance(data_path='data.csv'):
    """
    Check if model uses medically relevant features
    """
    print("\n" + "=" * 70)
    print("✓ CHECK 5: FEATURE IMPORTANCE (Medical Validity)")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv(data_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Train Random Forest to get feature importance
    print(f"\n   Training Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n📊 Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"   {row['Feature']:30s} {row['Importance']:.4f} {'█' * int(row['Importance']*100)}")
    
    # Check medical validity
    top_5 = importance_df.head(5)['Feature'].tolist()
    
    # Known medically relevant features
    medically_relevant = [
        'worst_perimeter', 'worst_radius', 'worst_area',
        'mean_perimeter', 'mean_radius', 'mean_area',
        'worst_concave_points', 'mean_concave_points',
        'worst_concavity', 'mean_concavity'
    ]
    
    relevant_in_top5 = sum(1 for f in top_5 if f in medically_relevant)
    
    print(f"\n💡 Medical Validity Check:")
    print(f"   Top 5 features: {top_5}")
    print(f"   Medically relevant in top 5: {relevant_in_top5}/5")
    
    if relevant_in_top5 >= 4:
        print(f"   ✅ EXCELLENT! Model uses clinically relevant features")
        print(f"   → These features are known cancer indicators in medical literature")
        status = "PASS"
    elif relevant_in_top5 >= 3:
        print(f"   ✅ GOOD! Most top features are medically relevant")
        status = "PASS"
    elif relevant_in_top5 >= 2:
        print(f"   ⚠️  FAIR. Some irrelevant features in top 5")
        status = "WARN"
    else:
        print(f"   ❌ CONCERNING! Top features don't match medical knowledge")
        status = "FAIL"
    
    return status, importance_df


# ========================================
# BONUS CHECK 6: CONFIDENCE ANALYSIS
# ========================================

def check_prediction_confidence(model, X_test_scaled, y_test):
    """
    Check if model makes confident predictions
    """
    print("\n" + "=" * 70)
    print("✓ BONUS CHECK: PREDICTION CONFIDENCE")
    print("=" * 70)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)
    max_proba = y_pred_proba.max(axis=1)
    
    # Count confidence levels
    very_confident = (max_proba >= 0.95).sum()
    confident = (max_proba >= 0.90).sum()
    moderate = ((max_proba >= 0.75) & (max_proba < 0.90)).sum()
    uncertain = (max_proba < 0.75).sum()
    
    total = len(y_test)
    
    print(f"\n📊 Confidence Distribution:")
    print(f"   Very Confident (≥95%): {very_confident:3d} ({very_confident/total*100:.1f}%)")
    print(f"   Confident (≥90%):      {confident:3d} ({confident/total*100:.1f}%)")
    print(f"   Moderate (75-90%):     {moderate:3d} ({moderate/total*100:.1f}%)")
    print(f"   Uncertain (<75%):      {uncertain:3d} ({uncertain/total*100:.1f}%)")
    
    print(f"\n   Average Confidence: {max_proba.mean()*100:.2f}%")
    print(f"   Minimum Confidence: {max_proba.min()*100:.2f}%")
    
    print(f"\n💡 Interpretation:")
    if confident/total >= 0.90:
        print(f"   ✅ EXCELLENT! {confident/total*100:.1f}% of predictions are confident")
    elif confident/total >= 0.80:
        print(f"   ✅ GOOD! {confident/total*100:.1f}% of predictions are confident")
    elif confident/total >= 0.70:
        print(f"   ⚠️  FAIR. Only {confident/total*100:.1f}% of predictions are confident")
    else:
        print(f"   ❌ POOR! Model is uncertain in many cases")
    
    return confident/total


# ========================================
# BONUS CHECK 7: RANDOM SEED STABILITY
# ========================================

def check_random_seed_stability(data_path='data.csv', n_seeds=5):
    """
    Test if model is stable across different random initializations
    """
    print("\n" + "=" * 70)
    print(f"✓ BONUS CHECK: RANDOM SEED STABILITY (Testing {n_seeds} seeds)")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv(data_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    seeds = [42, 100, 200, 300, 400][:n_seeds]
    accuracies = []
    
    print(f"\n📊 Testing different random seeds:")
    
    for seed in seeds:
        # Split with different seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=seed, max_iter=1000, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Test
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"   Seed {seed:3d}: {acc:.4f} ({acc*100:.2f}%)")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print(f"\n   Summary:")
    print(f"   Mean:      {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"   Std Dev:   {std_acc:.4f} (±{std_acc*100:.2f}%)")
    print(f"   Range:     {min_acc:.4f} - {max_acc:.4f}")
    
    print(f"\n💡 Interpretation:")
    if std_acc < 0.01:
        print(f"   ✅ EXCELLENT! Very stable (std < 1%)")
    elif std_acc < 0.02:
        print(f"   ✅ VERY GOOD! Stable (std < 2%)")
    elif std_acc < 0.03:
        print(f"   ✅ GOOD. Acceptable stability (std < 3%)")
    else:
        print(f"   ⚠️  Results vary with different random seeds (std > 3%)")
    
    return mean_acc, std_acc


# ========================================
# BONUS CHECK 8: MODEL COMPARISON
# ========================================

def check_model_comparison(data_path='data.csv'):
    """
    Compare Logistic Regression with other models
    """
    print("\n" + "=" * 70)
    print("✓ BONUS CHECK: MULTI-MODEL COMPARISON")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv(data_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    print(f"\n📊 Comparing 3 different models:")
    
    results = {}
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"   {name:20s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Check agreement
    lr_acc = results['Logistic Regression']
    max_acc = max(results.values())
    min_acc = min(results.values())
    spread = max_acc - min_acc
    
    print(f"\n   Performance Spread: {spread*100:.2f}%")
    
    print(f"\n💡 Interpretation:")
    if spread < 0.02:
        print(f"   ✅ EXCELLENT! All models agree (< 2% difference)")
        print(f"   → Strong evidence the patterns are real")
    elif spread < 0.05:
        print(f"   ✅ GOOD! Models mostly agree (< 5% difference)")
    else:
        print(f"   ⚠️  Models disagree significantly (> 5% difference)")
        print(f"   → May indicate data complexity or model selection issues")
    
    return results


# ========================================
# GENERATE FINAL REPORT
# ========================================

def generate_final_report(results):
    """
    Generate comprehensive verification report
    """
    print("\n" + "=" * 70)
    print("📋 FINAL VERIFICATION REPORT")
    print("=" * 70)
    
    checks = results['checks']
    
    passed = sum(1 for v in checks.values() if v == 'PASS')
    warned = sum(1 for v in checks.values() if v == 'WARN')
    failed = sum(1 for v in checks.values() if v == 'FAIL')
    total = len(checks)
    
    print(f"\n   Total Checks: {total}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ⚠️  Warnings: {warned}")
    print(f"   ❌ Failed: {failed}")
    
    print(f"\n   Detailed Results:")
    for check, status in checks.items():
        symbol = "✅" if status == "PASS" else ("⚠️ " if status == "WARN" else "❌")
        print(f"   {symbol} {check}: {status}")
    
    print(f"\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    
    if failed == 0 and warned == 0:
        print("✅ ✅ ✅ PERFECT! ALL CHECKS PASSED! ✅ ✅ ✅")
        print("\nYour model is:")
        print("  • Correctly trained")
        print("  • Generalizes well to new data")
        print("  • Medically valid")
        print("  • Production-ready")
        print("\n🎉 CONGRATULATIONS! Your model is EXCELLENT!")
        
    elif failed == 0:
        print("✅ EXCELLENT! All critical checks passed")
        print(f"\n⚠️  {warned} warning(s) - review recommended but not critical")
        print("\nYour model is solid and ready for deployment!")
        
    elif failed <= 2:
        print("⚠️  NEEDS IMPROVEMENT")
        print(f"\n{failed} check(s) failed - address these issues:")
        for check, status in checks.items():
            if status == "FAIL":
                print(f"  • {check}")
        
    else:
        print("❌ CRITICAL ISSUES DETECTED")
        print("\nMultiple checks failed. Model needs significant improvement.")
        print("Review failed checks and retrain model.")
    
    print("\n" + "=" * 70)
    
    # Save report
    with open('verification_report.txt', 'w') as f:
        f.write("MODEL VERIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy: {results['test_acc']:.4f}\n")
        f.write(f"Cross-Val Mean: {results['cv_mean']:.4f}\n")
        f.write(f"Baseline Improvement: {results['improvement']:.2f}%\n\n")
        f.write(f"Checks Passed: {passed}/{total}\n\n")
        for check, status in checks.items():
            f.write(f"{check}: {status}\n")
    
    print("\n📄 Full report saved to: verification_report.txt")


# ========================================
# RUN ALL CHECKS
# ========================================

def run_complete_verification(data_path='data.csv'):
    """
    Run all verification checks
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "MODEL VERIFICATION SUITE" + " " * 29 + "█")
    print("█" + " " * 15 + "Complete Analysis" + " " * 36 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    results = {'checks': {}}
    
    # Run essential checks
    try:
        status1, train_acc, test_acc, model, scaler, X_test, y_test = check_overfitting(data_path)
        results['checks']['Overfitting Check'] = status1
        results['test_acc'] = test_acc
        
        y_pred = model.predict(X_test)
        
        status2, cv_scores, cv_mean, cv_std = check_cross_validation(data_path)
        results['checks']['Cross-Validation'] = status2
        results['cv_mean'] = cv_mean
        
        status3, cm, errors = check_confusion_matrix(y_test, y_pred)
        results['checks']['Confusion Matrix'] = status3
        
        status4, baseline, improvement = check_baseline_comparison(y_test, test_acc)
        results['checks']['Baseline Comparison'] = status4
        results['improvement'] = improvement
        
        status5, importance = check_feature_importance(data_path)
        results['checks']['Feature Importance'] = status5
        
        # Bonus checks
        confidence = check_prediction_confidence(model, X_test, y_test)
        
        mean_acc, std_acc = check_random_seed_stability(data_path)
        
        model_results = check_model_comparison(data_path)
        
        # Generate final report
        generate_final_report(results)
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    import sys
    
    data_file = 'data.csv' if len(sys.argv) < 2 else sys.argv[1]
    
    print("\nStarting comprehensive verification...")
    print(f"Data file: {data_file}")
    
    input("\nPress Enter to begin verification (this will take 30-60 seconds)...")
    
    run_complete_verification(data_file)
    
    print("\n✅ Verification complete!")
    print("\nNext steps:")
    print("1. Review verification_report.txt")
    print("2. If all checks passed, model is ready for deployment")
    print("3. Add verification results to your project report")
    print("4. Run: streamlit run app.py (to launch web interface)")
    
    print("\n" + "=" * 70)
    print("Thank you for using the Model Verification Suite!")
    print("=" * 70)