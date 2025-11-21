# 🎗️ Breast Cancer Prediction System

An AI-powered web application for breast cancer diagnosis prediction using **Logistic Regression** machine learning model. This project provides an interactive interface for medical diagnostic assistance based on cell nucleus characteristics from the Wisconsin Breast Cancer Dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Demo](#-demo)
- [Model Overview](#-model-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Testing & Verification](#-testing--verification)
- [Disclaimer](#%EF%B8%8F-medical-disclaimer)

## 🎬 Demo

### Quick Start
Once installed, run the application with:
```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501` with three prediction modes:

**1. Manual Input** - Enter cell measurements through an intuitive form interface  
**2. CSV Upload** - Batch process multiple predictions from CSV files  
**3. Sample Data** - Test with pre-loaded benign and malignant examples

### Key Features
- 📊 **Interactive Visualizations**: Real-time gauge charts and probability distributions
- 🎯 **Risk Assessment**: Automatic classification into Low/Medium/High risk categories
- 💯 **Confidence Scores**: Transparent probability percentages for each prediction
- 📥 **Export Results**: Download batch predictions as CSV files

> **Note**: Screenshots/GIFs can be added to the `assets` folder and linked here once the app is running

## 🤖 Model Overview

### Model Type: **Logistic Regression**

This project uses a **Logistic Regression** classifier, a statistical model that's particularly well-suited for binary classification problems like cancer diagnosis.

**Why Logistic Regression?**

- ✅ **High Interpretability**: Easy to understand which features contribute to predictions
- ✅ **Fast Training**: Trains in seconds even on standard hardware
- ✅ **Excellent Performance**: Achieves 97%+ accuracy on breast cancer data
- ✅ **Probabilistic Output**: Provides confidence scores for predictions
- ✅ **Low Computational Cost**: Efficient for real-time predictions
- ✅ **Well-suited for Medical Data**: Works exceptionally well with structured medical measurements

**Model Configuration:**
```python
LogisticRegression(
    solver='lbfgs',      # Efficient optimizer for small-to-medium datasets
    max_iter=1000,       # Maximum iterations for convergence
    C=1.0,               # Regularization strength
    random_state=42,     # Reproducibility
    n_jobs=-1            # Use all CPU cores
)
```

**How It Works:**
1. Takes 30 numerical features from cell nucleus measurements
2. Applies feature standardization (StandardScaler)
3. Uses logistic function to calculate malignancy probability
4. Outputs binary classification: Benign (0) or Malignant (1)
5. Provides confidence scores as probability percentages

## ✨ Features

### 🎯 Multiple Input Methods
- **Manual Input**: Enter individual cell measurements through an intuitive interface
- **CSV Upload**: Batch predictions from uploaded CSV files
- **Sample Data**: Test with pre-loaded malignant and benign examples

### 📊 Rich Visualizations
- Interactive gauge charts for malignancy risk
- Probability breakdown bar charts
- Confidence scores and risk level indicators
- Color-coded results for easy interpretation

### 🔬 Comprehensive Testing Suite
- Holdout set validation
- Cross-validation testing
- Synthetic data generation
- Stress testing capabilities
- Model comparison tools

### ⚡ Performance Optimized
- Fast training (2-10 seconds)
- Real-time predictions
- Efficient memory usage
- Multi-core CPU support

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/breast_cancer_project.git
cd breast_cancer_project
```

> **Note**: Replace `yourusername` with your actual GitHub username after uploading

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you encounter numpy compatibility issues:
```bash
pip install 'numpy<2.0'
```

### Step 4: Train the Model (First Time Setup)
```bash
python deployment_pipeline.py
```

This will:
- Load and preprocess the data
- Train the Logistic Regression model
- Save model artifacts to `models/` directory
- Display performance metrics
- Run quick validation tests

## 💻 Usage

### Running the Web Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Using the Application

#### Method 1: Manual Input
1. Select "Manual Input" from the sidebar
2. Navigate through three tabs (Mean, Standard Error, Worst Values)
3. Enter measurements for all 30 features
4. Click "Predict" to see results

#### Method 2: CSV Upload
1. Select "Upload CSV" from the sidebar
2. Upload a CSV file with 30 feature columns
3. Click "Predict All" to process batch predictions
4. Download results as CSV

#### Method 3: Sample Data
1. Select "Use Sample Data" from the sidebar
2. Choose "Malignant Sample" or "Benign Sample"
3. Click "Predict Sample" to see example predictions

### Testing the Model
```bash
# Run comprehensive verification
python comprehensive_verification.py

# Run testing suite
python model_testing_suite.py
```

## 📈 Model Performance

### Metrics on Test Set
| Metric | Score |
|--------|-------|
| **Accuracy** | 97.37% |
| **Precision** | 97.62% |
| **Recall** | 95.35% |
| **F1-Score** | 96.47% |
| **Training Time** | ~2-5 seconds |

### Confusion Matrix Analysis
- **True Positives**: High detection rate for malignant cases
- **True Negatives**: Excellent benign case identification
- **False Positives**: Minimal misclassification of benign as malignant
- **False Negatives**: Very low missed malignant cases

### Cross-Validation
- **5-Fold CV Accuracy**: 96.8% ± 1.2%
- **Consistent Performance**: Low variance across folds
- **No Overfitting**: Train and test accuracies closely aligned

## 📁 Project Structure

```
breast_cancer_project/
│
├── app.py                           # Main Streamlit web application
├── deployment_pipeline.py           # Model training and deployment
├── comprehensive_verification.py    # Model verification suite
├── model_testing_suite.py          # Testing utilities
├── requirements.txt                # Python dependencies
├── data.csv                        # Wisconsin Breast Cancer Dataset
│
├── models/                         # Trained model artifacts
│   ├── breast_cancer_model.pkl    # Trained Logistic Regression model
│   ├── scaler.pkl                 # Feature StandardScaler
│   ├── feature_names.pkl          # Feature column names
│   └── model_metadata.pkl         # Training metadata
│
└── README.md                       # This file
```

## 📊 Dataset

**Source**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

**Features**: 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast masses

**Feature Categories**:
1. **Mean Values** (10 features): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard Error** (10 features): SE of the above measurements
3. **Worst Values** (10 features): Mean of the three largest values

**Target**:
- `M` (Malignant): 212 samples (37%)
- `B` (Benign): 357 samples (63%)

**Total Samples**: 569

## 🛠️ Technology Stack

### Core Framework
- **Python 3.11**: Primary programming language
- **Streamlit 1.40.0**: Web application framework

### Machine Learning
- **scikit-learn 1.5.2**: ML model implementation
  - `LogisticRegression`: Main classifier
  - `StandardScaler`: Feature normalization
  - `train_test_split`: Data splitting
  - Metrics: accuracy, precision, recall, f1-score

### Data Processing
- **pandas 2.3.3**: Data manipulation and analysis
- **numpy 2.0.0**: Numerical computations

### Visualization
- **Plotly 5.24.0**: Interactive charts and graphs
  - Gauge charts for risk visualization
  - Bar charts for probability breakdown

### Model Persistence
- **joblib 1.4.2**: Efficient model serialization

## 🧪 Testing & Verification

### Available Test Suites

1. **Overfitting Check**: Compares train vs test accuracy
2. **Cross-Validation**: Tests consistency across data splits
3. **Confusion Matrix Analysis**: Detailed error analysis
4. **Baseline Comparison**: Validates improvement over naive guessing
5. **Feature Importance**: Medical relevance validation
6. **Confidence Analysis**: Prediction certainty assessment
7. **Random Seed Stability**: Reproducibility verification
8. **Model Comparison**: Benchmarking against other algorithms

### Running Tests
```bash
# Full verification suite
python comprehensive_verification.py

# Interactive testing menu
python model_testing_suite.py
```

## 🔄 Retraining the Model

If you want to retrain with different parameters:

```bash
# Standard training
python deployment_pipeline.py

# Ultra-fast training (reduced features)
# Uncomment lines 441-442 in deployment_pipeline.py
```

### Optimization Options

**Faster Training**:
- Reduce `max_iter` (e.g., 500 instead of 1000)
- Use solver `'saga'` instead of `'lbfgs'`
- Increase tolerance `tol=1e-3`

**Better Accuracy**:
- Increase `max_iter` (e.g., 5000)
- Tune regularization parameter `C`
- Use cross-validation for hyperparameter tuning

## 📝 Feature Engineering

All 30 features are standardized using `StandardScaler`:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This ensures:
- Mean = 0, Standard Deviation = 1 for each feature
- Equal feature importance weights
- Faster convergence during training
- Better numerical stability

## ⚕️ Medical Disclaimer

> **IMPORTANT**: This application is for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical diagnosis or treatment.
>
> - Always consult qualified healthcare professionals for medical decisions
> - This tool is not FDA approved or clinically validated
> - Predictions are based on limited dataset and may not generalize to all cases
> - False negatives and false positives are possible

## 🎯 Future Enhancements

Potential improvements:
- [ ] Add more advanced models (Random Forest, XGBoost, Neural Networks)
- [ ] Implement SHAP values for better interpretability
- [ ] Add user authentication and history tracking
- [ ] Create REST API for integration
- [ ] Mobile-responsive design improvements
- [ ] Multi-language support
- [ ] Export detailed PDF reports

## 👨‍💻 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- How to report bugs
- How to suggest features
- Code style guidelines
- Pull request process

**Key Areas for Contribution:**
- [ ] Bug fixes and testing
- [ ] Documentation improvements
- [ ] New visualization features
- [ ] Model performance optimization
- [ ] UI/UX enhancements
- [ ] Additional ML models
- [ ] API development

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Wisconsin Diagnostic Breast Cancer Dataset providers
- scikit-learn development team
- Streamlit community
- Medical professionals who validated feature importance

## 📞 Support

For questions or issues:
1. Check existing documentation
2. Review test suite outputs
3. Verify model training completed successfully
4. Ensure all dependencies are correctly installed

## 🔗 References

- [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Logistic Regression - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Built with ❤️ for advancing medical AI research and education**
