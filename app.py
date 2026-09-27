import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

# ====================== إعداد الصفحة ======================
st.set_page_config(
    page_title="كاشف الصور المزيفة",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ كاشف الصور المزيفة (AI Detector)")
st.markdown("---")

# ====================== تحميل الموديل ======================
@st.cache_resource
def load_model():
    model_name = "Organika/sdxl-detector"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

with st.spinner("جاري تحميل الموديل... انتظر شوي"):
    processor, model = load_model()

# ====================== رفع الصورة ======================
uploaded_file = st.file_uploader("ارفع الصورة هنا", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    # ====================== التحليل ======================
    if st.button("🚀 ابدأ التحليل", type="primary"):
        with st.spinner("جاري التحليل..."):
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            # عادة label 0 = Real ، label 1 = AI (تأكد من الموديل)
            labels = model.config.id2label
            real_prob = float(probs[0]) * 100
            ai_prob = float(probs[1]) * 100
            
            # لو الموديل معكوس نصلح
            if "ai" in labels[0].lower() or "fake" in labels[0].lower():
                real_prob, ai_prob = ai_prob, real_prob

        with col2:
            st.subheader("نتيجة التحليل")
            
            if ai_prob > 70:
                st.error(f"🚨 **صورة مزيفة (AI)**")
                st.write(f"**مستوى الثقة:** عالي")
            elif ai_prob > 55:
                st.warning(f"⚠️ **محتملة أنها مزيفة**")
                st.write(f"**مستوى الثقة:** متوسط")
            else:
                st.success(f"✅ **صورة حقيقية على الأرجح**")
                st.write(f"**مستوى الثقة:** عالي")
            
            st.markdown("---")
            st.write(f"**نسبة التزييف (AI):** `{ai_prob:.1f}%`")
            st.progress(ai_prob / 100)
            
            st.write(f"**نسبة الواقعية:** `{real_prob:.1f}%`")
            st.progress(real_prob / 100)

else:
    st.info("ارفع صورة عشان نبدأ التحليل")

st.markdown("---")
st.caption("ملاحظة: النتائج مش 100% مضمونة، استخدمها كمساعدة فقط.")
