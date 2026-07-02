import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Page Setup
st.set_page_config(page_title="SEO RankPredict Dashboard", page_icon="🚀", layout="centered")

st.title("🚀 SEO RankPredict: Front-Page Predictor")
st.write("First-Year CSE Project — Algorithmic Marketing Simulator")
st.markdown("---")

# Background AI Training Function
@st.cache_data
def train_model():
    df = pd.read_csv("seo_data.csv")
    df['Is_Page_1'] = np.where(df['rank'] <= 10, 1, 0)
    X = df[['word_count', 'load_speed']]
    y = df['Is_Page_1']
    model = LogisticRegression()
    model.fit(X, y)
    return model

try:
    model = train_model()
    
    st.subheader("📝 Analyze Your Web Content")
    
    # 1. Text Area Input
    user_text = st.text_area("Paste your draft text/article here:", placeholder="In our shoes company we build better shoes...")
    
    # 2. Performance Metric Input (Asked to the user)
    load_speed = st.slider("Expected Website Load Speed (Seconds)", min_value=0.5, max_value=6.0, value=1.5, step=0.1)
    
    st.markdown("---")

    if st.button("Predict Google Ranking Position", type="primary"):
        if not user_text.strip():
            st.warning("⚠️ Please type some text in the box above to analyze word count parameters!")
        else:
            # Code automatically calculates the word count behind the scenes
            calculated_word_count = len(user_text.split())
            st.info(f"📊 Auto-Calculated Metrics -> Word Count: **{calculated_word_count} words** | Load Time Input: **{load_speed}s**")
            
            # Feed numbers into the Machine Learning Equation
            user_input = [[calculated_word_count, load_speed]]
            prediction = model.predict(user_input)
            probability = model.predict_proba(user_input)[0][1] * 100
            
            # Show the Final Verdict UI
            st.markdown("### **Model Verdict:**")
            if prediction[0] == 1:
                st.success(f"🎉 **Page 1 Highly Likely!** This structure fits high-performance ranking parameters with a **{probability:.1f}%** confidence index. Publish your page!")
            else:
                st.error(f"❌ **Page 2+ Visibility Warning!** Low search viability score (**{probability:.1f}%**). Try writing more text or adjusting your technical load metrics.")
except Exception as e:
    st.error("Error setting up internal machine learning engine components.")