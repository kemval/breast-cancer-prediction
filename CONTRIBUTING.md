# Contributing to Breast Cancer Prediction System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs
1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Features
1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Explain why this would be valuable

### Code Contributions

#### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/breast_cancer_project.git
cd breast_cancer_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python deployment_pipeline.py
```

#### Making Changes
1. **Fork the repository**
2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add comments for complex logic
   - Update documentation if needed
4. **Test your changes**
   ```bash
   # Run verification
   python comprehensive_verification.py
   
   # Test the app
   streamlit run app.py
   ```
5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

#### Commit Message Guidelines
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Start with a verb (Add, Update, Fix, Remove, etc.)
- Keep first line under 50 characters
- Add detailed description if needed

Examples:
```
Add Random Forest model option
Fix numpy version compatibility issue
Update README with installation steps
Remove deprecated function calls
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional ML models (Random Forest, XGBoost, Neural Networks)
- [ ] Model explainability (SHAP values, LIME)
- [ ] Improved UI/UX design
- [ ] Mobile responsiveness
- [ ] API development (REST or GraphQL)

### Medium Priority
- [ ] User authentication system
- [ ] Prediction history tracking
- [ ] Export results as PDF
- [ ] Multi-language support
- [ ] Dark mode

### Low Priority
- [ ] Additional visualizations
- [ ] Performance optimizations
- [ ] Code refactoring
- [ ] Additional test cases
- [ ] Documentation improvements

## 📋 Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints where applicable
- Maximum line length: 100 characters
- Use meaningful variable names
- Add docstrings for functions and classes

Example:
```python
def predict_cancer(features: List[float], model, scaler) -> Tuple[int, np.ndarray]:
    """
    Predict cancer diagnosis from cell features.
    
    Args:
        features: List of 30 numerical feature values
        model: Trained sklearn model
        scaler: Fitted StandardScaler
        
    Returns:
        Tuple of (prediction, probabilities)
    """
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    return prediction, probabilities
```

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Include docstrings for all functions
- Update requirements.txt if adding dependencies

## 🧪 Testing

Before submitting a PR, ensure:
- [ ] Code runs without errors
- [ ] Model training completes successfully
- [ ] Web app loads and functions properly
- [ ] All verification tests pass
- [ ] No new warnings introduced
- [ ] Documentation is updated

Run tests:
```bash
# Comprehensive verification
python comprehensive_verification.py

# Testing suite
python model_testing_suite.py

# Manual testing
streamlit run app.py
```

## 📝 Pull Request Process

1. **Update documentation** - README, docstrings, comments
2. **Follow code style** - PEP 8, clear naming
3. **Test thoroughly** - Verify nothing breaks
4. **Describe changes** - Clear PR description
5. **Link issues** - Reference related issues
6. **Be responsive** - Address review feedback

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe testing done

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass
```

## ⚕️ Medical AI Considerations

This project deals with medical predictions. Please ensure:
- **Accuracy**: Verify changes don't reduce model performance
- **Safety**: Consider edge cases and failure modes
- **Transparency**: Document model decisions and limitations
- **Disclaimer**: Maintain medical disclaimers
- **Ethics**: Consider bias and fairness implications

## 📞 Questions?

- Open an issue for questions
- Check existing documentation
- Review closed issues for similar questions

## 🙏 Recognition

Contributors will be acknowledged in the README. Thank you for helping improve this project!

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
