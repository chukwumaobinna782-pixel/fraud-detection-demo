# Real-Time Credit Card Fraud Detection 🛡️

**XGBoost-powered fraud detection system** with live transaction simulation and real-time risk assessment.

## 🚀 Live Demo
[Deploy this app to see live fraud detection in action!]

## 📊 Model Performance
- **Algorithm**: XGBoost Classifier
- **ROC-AUC**: ~0.98+
- **Features**: 43 engineered features including PCA components
- **Training Dataset**: 284,807 credit card transactions

## 🎯 Key Features
- ⚡ **Real-time fraud detection** every 2 seconds
- 📈 **Live transaction simulation** 
- 🎯 **Automated decision pipeline** (APPROVE/REVIEW/BLOCK)
- 💰 **Risk-based transaction scoring**

## 🛠️ Technical Stack
- **ML Framework**: XGBoost + Scikit-learn
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud
- **Dependencies**: pandas, numpy, joblib

## 📋 Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
