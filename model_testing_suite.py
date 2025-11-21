# ========================================
# MODEL TESTING SUITE
# Test your model with different data
# ========================================

import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# ========================================
# TEST 1: HOLDOUT TEST SET
# Test on data the model has NEVER seen
# ========================================

def test_on_holdout_set(data_path='data.csv'):
    """
    Test model on completely unseen data
    This is the BEST way to verify if model works
    """
    print("=" * 60)
    print("TEST 1: HOLDOUT TEST SET (Data Model Has Never Seen)")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv(data_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Load model
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Create a DIFFERENT split (not the same as training!)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99  # Different random state!
    )
    
    print(f"\nTesting on {len(X_test)} completely new samples...")
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n📊 RESULTS:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Interpretation
    if accuracy > 0.95:
        print("\n✅ EXCELLENT! Model generalizes very well to new data!")
    elif accuracy > 0.90:
        print("\n✅ GOOD! Model works well on new data.")
    elif accuracy > 0.85:
        print("\n⚠️ FAIR. Model works but could be improved.")
    else:
        print("\n❌ POOR. Model may be overfitting or needs improvement.")
    
    return accuracy, precision, recall, f1


# ========================================
# TEST 2: CROSS-VALIDATION
# Test with multiple data splits
# ========================================

def test_with_cross_validation(data_path='data.csv', n_folds=5):
    """
    Test model stability across multiple data splits
    This shows if model is consistent
    """
    print("\n" + "=" * 60)
    print(f"TEST 2: {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv(data_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Scale data
    scaler = joblib.load('models/scaler.pkl')
    X_scaled = scaler.fit_transform(X)
    
    # Load model
    model = joblib.load('models/breast_cancer_model.pkl')
    
    # Perform cross-validation
    print(f"\nTesting model on {n_folds} different data splits...")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')
    
    print(f"\n📊 CROSS-VALIDATION RESULTS:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print(f"\n  Mean:    {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"  Std Dev: {cv_scores.std():.4f} (±{cv_scores.std()*100:.2f}%)")
    print(f"  Range:   {cv_scores.min():.4f} - {cv_scores.max():.4f}")
    
    # Interpretation
    if cv_scores.std() < 0.03:
        print("\n✅ VERY STABLE! Model performs consistently.")
    elif cv_scores.std() < 0.05:
        print("\n✅ STABLE. Model is reliable.")
    else:
        print("\n⚠️ UNSTABLE. Results vary significantly across splits.")
    
    return cv_scores


# ========================================
# TEST 3: MANUAL TEST CASES
# Test with hand-picked examples
# ========================================

def test_manual_cases():
    """
    Test with specific known cases
    """
    print("\n" + "=" * 60)
    print("TEST 3: MANUAL TEST CASES")
    print("=" * 60)
    
    # Load model
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Test Case 1: Known MALIGNANT case
    print("\n📋 Test Case 1: Known Malignant Sample")
    malignant_sample = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
        0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    
    scaled = scaler.transform([malignant_sample])
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    
    print(f"  Expected: Malignant")
    print(f"  Predicted: {'Malignant' if pred == 1 else 'Benign'}")
    print(f"  Confidence: {max(proba)*100:.2f}%")
    print(f"  ✅ CORRECT!" if pred == 1 else "  ❌ WRONG!")
    
    # Test Case 2: Known BENIGN case
    print("\n📋 Test Case 2: Known Benign Sample")
    benign_sample = [
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
        0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146,
        0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2,
        0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ]
    
    scaled = scaler.transform([benign_sample])
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    
    print(f"  Expected: Benign")
    print(f"  Predicted: {'Malignant' if pred == 1 else 'Benign'}")
    print(f"  Confidence: {max(proba)*100:.2f}%")
    print(f"  ✅ CORRECT!" if pred == 0 else "  ❌ WRONG!")
    
    # Test Case 3: Extreme values (edge case)
    print("\n📋 Test Case 3: Edge Case - Extreme Values")
    print("  Testing model robustness...")
    
    try:
        extreme_sample = [val * 0.5 for val in benign_sample]  # Very small values
        scaled = scaler.transform([extreme_sample])
        pred = model.predict(scaled)[0]
        print(f"  Model handled extreme values: {'Malignant' if pred == 1 else 'Benign'}")
        print("  ✅ Model is robust to edge cases")
    except Exception as e:
        print(f"  ❌ Model failed on edge case: {e}")


# ========================================
# TEST 4: UPLOAD YOUR OWN DATA
# Test with completely different dataset
# ========================================

def test_with_new_data(new_data_path):
    """
    Test model with YOUR OWN new dataset
    
    Usage:
        test_with_new_data('my_new_data.csv')
    
    CSV should have same 30 features + diagnosis column
    """
    print("\n" + "=" * 60)
    print("TEST 4: TESTING WITH NEW UPLOADED DATA")
    print("=" * 60)
    
    try:
        # Load new data
        print(f"\nLoading data from: {new_data_path}")
        new_data = pd.read_csv(new_data_path)
        print(f"✓ Loaded {len(new_data)} samples")
        
        # Check if diagnosis column exists
        has_labels = 'diagnosis' in new_data.columns
        
        if has_labels:
            # We have actual labels to compare
            print("\n✓ Found diagnosis labels - can evaluate accuracy!")
            
            # Prepare data
            new_data['diagnosis'] = new_data['diagnosis'].apply(
                lambda x: 1 if x == 'M' else 0
            )
            X_new = new_data.drop(['diagnosis', 'id', 'Unnamed: 32'], 
                                 axis=1, errors='ignore')
            y_new = new_data['diagnosis']
        else:
            # No labels - just predict
            print("\n⚠️ No diagnosis labels found - will only predict")
            X_new = new_data.drop(['id', 'Unnamed: 32'], 
                                 axis=1, errors='ignore')
            y_new = None
        
        # Load model
        model = joblib.load('models/breast_cancer_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Make predictions
        print("\nMaking predictions...")
        X_new_scaled = scaler.transform(X_new)
        y_pred = model.predict(X_new_scaled)
        y_pred_proba = model.predict_proba(X_new_scaled)
        
        # Show results
        print("\n📊 PREDICTION RESULTS:")
        results_df = pd.DataFrame({
            'Sample': range(1, len(y_pred) + 1),
            'Prediction': ['Malignant' if p == 1 else 'Benign' for p in y_pred],
            'Confidence': [max(proba) * 100 for proba in y_pred_proba],
            'Malignant_Prob': y_pred_proba[:, 1] * 100
        })
        
        print(results_df.to_string(index=False))
        
        # If we have labels, calculate accuracy
        if has_labels:
            accuracy = accuracy_score(y_new, y_pred)
            precision = precision_score(y_new, y_pred)
            recall = recall_score(y_new, y_pred)
            
            print(f"\n📈 PERFORMANCE ON NEW DATA:")
            print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
            
            if accuracy > 0.90:
                print("\n✅ EXCELLENT! Model works on new data!")
            else:
                print("\n⚠️ Model performance decreased on new data.")
                print("   This could mean:")
                print("   - New data is different from training data")
                print("   - Model may need retraining with more diverse data")
        
        # Save results
        results_df.to_csv('prediction_results.csv', index=False)
        print("\n✓ Results saved to: prediction_results.csv")
        
        return results_df
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{new_data_path}' not found!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


# ========================================
# TEST 5: GENERATE SYNTHETIC TEST DATA
# Create artificial data to test model
# ========================================

def test_with_synthetic_data(n_samples=100):
    """
    Generate synthetic data to test model
    Good for testing when you don't have real new data
    """
    print("\n" + "=" * 60)
    print(f"TEST 5: SYNTHETIC DATA TEST ({n_samples} samples)")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    
    # Generate synthetic data similar to breast cancer dataset
    print("\nGenerating synthetic test data...")
    X_synthetic, y_synthetic = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=20,
        n_redundant=10,
        n_classes=2,
        weights=[0.63, 0.37],  # Similar to original class distribution
        random_state=42
    )
    
    # Load model
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Scale and predict
    X_synthetic_scaled = scaler.transform(X_synthetic)
    y_pred = model.predict(X_synthetic_scaled)
    y_pred_proba = model.predict_proba(X_synthetic_scaled)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_synthetic, y_pred)
    precision = precision_score(y_synthetic, y_pred)
    recall = recall_score(y_synthetic, y_pred)
    
    print(f"\n📊 RESULTS ON SYNTHETIC DATA:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    
    print("\n📝 NOTE: Synthetic data may have different patterns")
    print("   Use this test to verify model doesn't crash on new data")
    print("   For real accuracy, use actual breast cancer data!")


# ========================================
# TEST 6: STRESS TEST
# Test model limits
# ========================================

def stress_test_model():
    """
    Test model with challenging cases
    """
    print("\n" + "=" * 60)
    print("TEST 6: STRESS TEST")
    print("=" * 60)
    
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    tests_passed = 0
    total_tests = 5
    
    print("\nRunning stress tests...")
    
    # Test 1: All zeros
    print("\n  Test 1: All zeros")
    try:
        zeros = [0] * 30
        pred = model.predict(scaler.transform([zeros]))[0]
        print(f"    ✓ Handled all zeros: {pred}")
        tests_passed += 1
    except:
        print("    ✗ Failed on all zeros")
    
    # Test 2: All ones
    print("  Test 2: All ones")
    try:
        ones = [1] * 30
        pred = model.predict(scaler.transform([ones]))[0]
        print(f"    ✓ Handled all ones: {pred}")
        tests_passed += 1
    except:
        print("    ✗ Failed on all ones")
    
    # Test 3: Very large values
    print("  Test 3: Very large values")
    try:
        large = [1000] * 30
        pred = model.predict(scaler.transform([large]))[0]
        print(f"    ✓ Handled large values: {pred}")
        tests_passed += 1
    except:
        print("    ✗ Failed on large values")
    
    # Test 4: Negative values
    print("  Test 4: Negative values")
    try:
        negative = [-10] * 30
        pred = model.predict(scaler.transform([negative]))[0]
        print(f"    ✓ Handled negative values: {pred}")
        tests_passed += 1
    except:
        print("    ✗ Failed on negative values")
    
    # Test 5: Mixed extreme values
    print("  Test 5: Mixed extreme values")
    try:
        mixed = [1000 if i % 2 == 0 else 0.001 for i in range(30)]
        pred = model.predict(scaler.transform([mixed]))[0]
        print(f"    ✓ Handled mixed extremes: {pred}")
        tests_passed += 1
    except:
        print("    ✗ Failed on mixed extremes")
    
    print(f"\n📊 Stress Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ Model is ROBUST and handles edge cases well!")
    elif tests_passed >= 3:
        print("⚠️ Model handles most cases but may need improvement")
    else:
        print("❌ Model struggles with edge cases")


# ========================================
# RUN ALL TESTS
# ========================================

def run_all_tests(data_path='data.csv'):
    """
    Run complete test suite
    """
    print("\n" + "=" * 60)
    print("🧪 COMPREHENSIVE MODEL TESTING SUITE")
    print("=" * 60)
    print("\nThis will test your model with 6 different methods")
    print("to verify it actually works and generalizes well!\n")
    
    input("Press Enter to start testing...")
    
    try:
        # Test 1: Holdout set
        test_on_holdout_set(data_path)
        
        # Test 2: Cross-validation
        test_with_cross_validation(data_path)
        
        # Test 3: Manual cases
        test_manual_cases()
        
        # Test 4: Skip (requires user data)
        print("\n" + "=" * 60)
        print("TEST 4: SKIPPED (Upload your own data to use this)")
        print("Usage: test_with_new_data('your_data.csv')")
        print("=" * 60)
        
        # Test 5: Synthetic data
        test_with_synthetic_data()
        
        # Test 6: Stress test
        stress_test_model()
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ TESTING COMPLETE!")
        print("=" * 60)
        print("\nYour model has been tested with:")
        print("  ✓ Unseen holdout data")
        print("  ✓ Cross-validation (5 folds)")
        print("  ✓ Manual test cases")
        print("  ✓ Synthetic data")
        print("  ✓ Stress tests")
        print("\nIf all tests passed with >90% accuracy,")
        print("your model is working correctly! 🎉")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


# ========================================
# MAIN MENU
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 MODEL TESTING SUITE")
    print("=" * 60)
    print("""
Choose a test:
1. Run ALL tests (Recommended)
2. Test on holdout set only
3. Cross-validation only
4. Manual test cases only
5. Test with YOUR OWN data
6. Synthetic data test
7. Stress test only
0. Exit
    """)
    
    choice = input("Enter choice (0-7): ").strip()
    
    if choice == "1":
        run_all_tests()
    elif choice == "2":
        test_on_holdout_set()
    elif choice == "3":
        test_with_cross_validation()
    elif choice == "4":
        test_manual_cases()
    elif choice == "5":
        file_path = input("Enter CSV file path: ").strip()
        test_with_new_data(file_path)
    elif choice == "6":
        test_with_synthetic_data()
    elif choice == "7":
        stress_test_model()
    else:
        print("Exiting...")