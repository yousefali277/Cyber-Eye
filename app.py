import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# إعدادات الصفحة
st.set_page_config(page_title="CyberEye Platform", page_icon="🛡️", layout="wide")

# تحميل النموذج
@st.cache_resource
def load_model():
    model_name = "Organika/sdxl-detector"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

processor, model = load_model()

# العنوان الرئيسي
st.title("🛡️ منصة CyberEye لكشف التزييف العميق")
st.caption("النظام الذكي للحماية والاستجابة اللحظية | مشارك في مسابقة SAIF 2026")
st.markdown("---")

# إنشاء التبويبات
tab1, tab2, tab3 = st.tabs(["🔍 وحدة الفحص", "📊 لوحة الإحصائيات", "ℹ️ عن المنصة"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("ارفع الصورة هنا للتحليل السيبراني", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="الصورة المرفوعة", use_container_width=True)
            
    with col2:
        if uploaded_file is not None:
            if st.button("🚀 بدء التحليل السيبراني", type="primary"):
                with st.spinner("جاري تحليل الترددات والأنماط..."):
                    image_rgb = image.convert("RGB")
                    inputs = processor(images=image_rgb, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=-1)[0]
                    
                    fake_score = float(probs[0].item())
                    real_score = float(probs[1].item())
                    
                    if fake_score > real_score:
                        st.error(f"🔴 تم كشف تزييف عميق (Deepfake)\n\nمستوى الخطورة: عالي (High Risk)")
                    else:
                        st.success(f"🟢 المحتوى حقيقي (Authentic)\n\nمستوى الخطورة: آمن (Safe)")
                    
                    st.write("### تفاصيل الاحتمالات:")
                    st.progress(fake_score, text=f"نسبة التزييف: {fake_score*100:.1f}%")
                    st.progress(real_score, text=f"نسبة الواقعية: {real_score*100:.1f}%")

with tab2:
    st.subheader("أداء محرك الذكاء الاصطناعي")
    st.metric(label="زمن الاستجابة (Latency)", value="~0.4s", delta="Real-Time")
    st.metric(label="المعمارية المستخدمة", value="Vision Transformer")
    st.metric(label="دقة النموذج المستهدفة", value="+95%")

with tab3:
    st.write("**تطوير:** يوسف علي")
    st.write("**الفئة:** مسابقة SAIF 2026")
    st.write("**الهدف:** كشف الوسائط المزيفة ودعم الأمن السيبراني.")
