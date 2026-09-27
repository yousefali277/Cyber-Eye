import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ====================== إعداد الصفحة ======================
st.set_page_config(
    page_title="Cyber Eye",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Cyber Eye")
st.caption("نظام كشف الصور المزيفة بالذكاء الاصطناعي")
st.markdown("---")

# ====================== تحميل الموديل ======================
@st.cache_resource
def load_model():
    model_name = "Organika/sdxl-detector"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

with st.spinner("جاري تحميل النظام..."):
    processor, model = load_model()

# ====================== رفع الصورة ======================
uploaded_file = st.file_uploader("ارفع الصورة للتحليل", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("🚀 بدء التحليل", type="primary"):
        with st.spinner("جاري التحليل السيبراني..."):
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            labels = model.config.id2label
            real_prob = float(probs[0]) * 100
            ai_prob = float(probs[1]) * 100
            
            # تصحيح لو الموديل معكوس
            if "ai" in str(labels[0]).lower() or "fake" in str(labels[0]).lower():
                real_prob, ai_prob = ai_prob, real_prob

        with col2:
            st.subheader("نتيجة التحليل")
            
            if ai_prob >= 70:
                st.error("🚨 تم كشف تزيف عميق (Deepfake)")
                st.write("**مستوى الخطورة:** عالي")
            elif ai_prob >= 55:
                st.warning("⚠️ صورة مشبوهة")
                st.write("**مستوى الخطورة:** متوسط")
            else:
                st.success("✅ صورة حقيقية على الأرجح")
                st.write("**مستوى الخطورة:** منخفض")
            
            st.markdown("---")
            st.write("**تفاصيل الاحتمالات (Ensemble Result):**")
            
            st.write(f"نسبة التزييف: **{ai_prob:.1f}%**")
            st.progress(ai_prob / 100)
            
            st.write(f"نسبة الواقعية: **{real_prob:.1f}%**")
            st.progress(real_prob / 100)

else:
    st.info("ارفع صورة عشان نبدأ التحليل")

st.markdown("---")
st.caption("Cyber Eye • تطوير: يوسف المرشدي")
