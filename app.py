
### **Step 4: Upload Your App.py (Fixed for Deployment)**
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
import traceback


# Page config first
st.set_page_config(page_title="Live Fraud Detection Demo", page_icon="🛡️", layout="wide")

# Check for model files
if not os.path.exists('xgboost_fraud_with_features.pkl') or not os.path.exists('scaler_with_features.pkl'):
    st.error("❌ Model files missing! Please ensure 'xgboost_fraud_with_features.pkl' and 'scaler_with_features.pkl' are uploaded.")
    st.stop()

@st.cache_resource
def load_models():
    try:
        model = joblib.load('xgboost_fraud_with_features.pkl')
        scaler = joblib.load('scaler_with_features.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

model, scaler = load_models()

st.write("### 🔍 Debug Console")
st.write("Testing your fraud detection system...")

# Test 1: Model Files Exist?
col1, col2 = st.columns(2)
with col1:
    st.write("📁 Checking model files:")
    if os.path.exists('xgboost_fraud_with_features.pkl'):
        st.success("✅ Model file found")
    else:
        st.error("❌ Model file missing")

with col2:
    if os.path.exists('scaler_with_features.pkl'):
        st.success("✅ Scaler file found") 
    else:
        st.error("❌ Scaler file missing")

# Test 2: Model Loading
with st.expander("🧪 Test Model Loading"):
    try:
        model, scaler = load_models()
        st.success(f"✅ Model loaded: {type(model).__name__}")
        st.success(f"✅ Scaler loaded: {type(scaler).__name__}")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.code(traceback.format_exc())

# Test 3: Transaction Generation
with st.expander("🧪 Test Transaction"):
    try:
        txn = generate_random_transaction()
        st.write("📊 Raw transaction:", txn)
        
        # Show a few key values
        st.write(f"💰 Amount: ${txn['Amount']:.2f}")
        st.write(f"⏰ Time: {txn['Time']:.0f}")
        st.write(f"V14: {txn.get('V14', 'Not set')}")
        
    except Exception as e:
        st.error(f"❌ Transaction generation failed: {str(e)}")

# Test 4: Prediction
with st.expander("🧪 Test Prediction"):
    try:
        txn = generate_random_transaction()
        pred = predict_fraud_live(txn)
        
        st.success(f"🎯 Fraud Probability: {pred['fraud_probability']:.1%}")
        st.success(f"🏷️ Decision: {pred['decision']}")
        st.success(f"⚡ Action: {pred['action']}")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.code(traceback.format_exc())

# Test 5: Manual Simulation
with st.expander("🧪 Manual Simulation"):
    if st.button("Generate ONE Transaction"):
        try:
            txn = generate_random_transaction()
            pred = predict_fraud_live(txn)
            
            result = {
                'Amount': f"${txn['Amount']:.2f}",
                'Fraud Probability': f"{pred['fraud_probability']:.1%}",
                'Decision': pred['decision'],
                'Action': pred['action']
            }
            
            st.success("✅ Transaction processed!")
            st.write(result)
            
        except Exception as e:
            st.error(f"❌ Manual simulation failed: {str(e)}")

# Feature columns from your training
FEATURE_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
    'Hour', 'Is_night', 'Amount_log', 'Is_Zero_Amount', 'V_sum', 'V_std', 'V_max', 'V_min', 'V3_x_V10', 'V3_x_V12', 'V10_x_V12', 'V11_x_V14', 'V14_x_V17'
]

def predict_fraud_live(transaction_dict):
    df_live = pd.DataFrame([transaction_dict])

    # Feature engineering
    df_live['Hour'] = (df_live['Time'] // 3600) % 24
    df_live['Is_night'] = df_live['Hour'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    df_live['Amount_log'] = np.log1p(df_live['Amount'])
    df_live['Is_Zero_Amount'] = (df_live['Amount'] == 0).astype(int)

    v_cols = [f'V{i}' for i in range(1, 29)]
    df_live['V_sum'] = df_live[v_cols].sum(axis=1)
    df_live['V_std'] = df_live[v_cols].std(axis=1)
    df_live['V_max'] = df_live[v_cols].max(axis=1)
    df_live['V_min'] = df_live[v_cols].min(axis=1)

    df_live['V3_x_V10'] = df_live['V3'] * df_live['V10']
    df_live['V3_x_V12'] = df_live['V3'] * df_live['V12']
    df_live['V10_x_V12'] = df_live['V10'] * df_live['V12']
    df_live['V11_x_V14'] = df_live['V11'] * df_live['V14']
    df_live['V14_x_V17'] = df_live['V14'] * df_live['V17']

    # Add missing columns
    for col in FEATURE_COLUMNS:
        if col not in df_live.columns:
            df_live[col] = 0

    X_live = df_live[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X_live)

    prob = model.predict_proba(X_scaled)[0, 1]

    return {
        "fraud_probability": round(float(prob), 5),
        "decision": "🚨 FRAUD - BLOCK" if prob > 0.5 else "✅ APPROVED",
        "action": "🔒 BLOCK" if prob > 0.8 else ("⚠️ REVIEW" if prob > 0.3 else "✅ APPROVE")
    }

def generate_random_transaction():
    txn = {
        'Time': np.random.uniform(0, 172800),
        'Amount': np.random.exponential(88.35),
    }
    for i in range(1, 29):
        txn[f'V{i}'] = np.random.normal(0, 1)
    
    # 1% fraud chance
    if np.random.rand() < 0.01:
        txn['V14'] = np.random.uniform(-10, -5)
        txn['Amount'] = np.random.uniform(100, 1000)

    return txn

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# UI
st.title("🛡️ Real-Time Credit Card Fraud Detection")
st.markdown("**XGBoost Model** • ROC-AUC: ~0.98+ • Live Transaction Simulation")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶️ Start Simulation", type="primary"):
        st.session_state.sim_running = True
        st.session_state.last_update = time.time()

with col2:
    if st.button("⏸️ Stop Simulation"):
        st.session_state.sim_running = False

with col3:
    st.metric("Status", "🟢 RUNNING" if st.session_state.sim_running else "🔴 STOPPED")

if st.session_state.sim_running:
    try:
        txn = generate_random_transaction()
        pred = predict_fraud_live(txn)
        
        timestamp = time.strftime("%H:%M:%S")
        entry = {
            '🕐 Time': timestamp,
            '💰 Amount': f"${txn['Amount']:.2f}",
            '🎯 Fraud Prob': f"{pred['fraud_probability']:.1%}",
            '🏷️ Decision': pred['decision'],
            '⚡ Action': pred['action']
        }
        st.session_state.transactions.append(entry)
        
        if len(st.session_state.transactions) > 25:
            st.session_state.transactions = st.session_state.transactions[-25:]
        
        # Force rerun to update display
        st.rerun()
        
    except Exception as e:
        st.error(f"Simulation error: {str(e)}")

# Display results
if st.session_state.transactions:
    df_display = pd.DataFrame(st.session_state.transactions)
    
    def highlight_fraud(row):
        if '🚨 FRAUD' in row['🏷️ Decision']:
            return ['background-color: #ff4444; color: white' for _ in row]
        elif '⚠️ REVIEW' in row['⚡ Action']:
            return ['background-color: #ffaa44; color: black' for _ in row]
        return ['' for _ in row]
    
    st.dataframe(df_display.style.apply(highlight_fraud, axis=1), use_container_width=True)
    
    # Summary metrics
    latest = st.session_state.transactions[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Transaction", latest['💰 Amount'])
    col2.metric("Fraud Probability", latest['🎯 Fraud Prob'])
    col3.metric("Decision", latest['🏷️ Decision'].split(' - ')[0])
    
else:
    st.info("🚀 Click 'Start Simulation' to begin real-time fraud detection!")

# Footer
st.markdown("---")
st.markdown("*Built with XGBoost • Streamlit • Real-time ML Pipeline*")
