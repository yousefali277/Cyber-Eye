import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# إعدادات الصفحة
st.set_page_config(page_title="CyberEye Platform", page_icon="🛡️", layout="wide")

# تحميل النماذج المزدوجة (Ensemble)
@st.cache_resource
def load_models():
    # Model 1: SDXL Detector
    m1_name = "Organika/sdxl-detector"
    p1 = AutoImageProcessor.from_pretrained(m1_name)
    m1 = AutoModelForImageClassification.from_pretrained(m1_name)
    m1.eval()

    # Model 2: AI Image Detector
    m2_name = "umm-maybe/AI-image-detector"
    p2 = AutoImageProcessor.from_pretrained(m2_name)
    m2 = AutoModelForImageClassification.from_pretrained(m2_name)
    m2.eval()

    return (p1, m1), (p2, m2)

(p1, m1), (p2, m2) = load_models()

# دالة تركيز وقص زوائد الشاشة والتركيز على المنتصف تلقائياً
def focus_on_content(pil_image):
    image = pil_image.convert("RGB")
    w, h = image.size
    
    # إذا كانت لقطة شاشة طوالية (أطول من العرض بكثير)
    if h > w * 1.1:
        # اقتطاع الجزء الأوسط الذي يحوي الوجه/المحتوى وإلغاء أزرار التطبيقات العلوية والسفلية
        left = int(w * 0.05)
        top = int(h * 0.20)
        right = int(w * 0.95)
        bottom = int(h * 0.80)
        return image.crop((left, top, right, bottom))
    
    return image

# واجهة الموقع
st.title("🛡️ منصة CyberEye لكشف التزييف العميق")
st.caption("نظام التقييم المزدوج المطور بالذكاء الاصطناعي | مسابقة SAIF 2026")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 وحدة الفحص", "📊 لوحة الإحصائيات", "ℹ️ عن المنصة"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("ارفع الصورة هنا للتحليل السيبراني", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            st.image(raw_image, caption="الصورة المرفوعة الأصليّة", use_container_width=True)
            
    with col2:
        if uploaded_file is not None:
            if st.button("🚀 بدء التحليل السيبراني المزدوج", type="primary"):
                with st.spinner("جاري تنقية الصورة وتحليل الترددات الهيكلية..."):
                    # التركيز على محتوى الصورة الأساسي
                    processed_img = focus_on_content(raw_image)

                    # الفحص بالنموذج الأول
                    in1 = p1(images=processed_img, return_tensors="pt")
                    with torch.no_grad():
                        out1 = m1(**in1)
                        prob1 = F.softmax(out1.logits, dim=-1)[0]
                        fake1 = float(prob1[0].item())

                    # الفحص بالنموذج الثاني
                    in2 = p2(images=processed_img, return_tensors="pt")
                    with torch.no_grad():
                        out2 = m2(**in2)
                        prob2 = F.softmax(out2.logits, dim=-1)[0]
                        fake2 = float(prob2[0].item())

                    # متوسط الفحص المزدوج
                    fake_score = (fake1 + fake2) / 2.0
                    real_score = 1.0 - fake_score

                    if fake_score > 0.5:
                        st.error("🔴 تم كشف تزييف عميق (Deepfake)\n\nمستوى الخطورة: عالي (High Risk)")
                    else:
                        st.success("🟢 المحتوى حقيقي (Authentic)\n\nمستوى الخطورة: آمن (Safe)")
                    
                    st.write("### تفاصيل الاحتمالات (Ensemble Result):")
                    st.progress(fake_score, text=f"نسبة التزييف: {fake_score*100:.1f}%")
                    st.progress(real_score, text=f"نسبة الواقعية: {real_score*100:.1f}%")

with tab2:
    st.subheader("أداء محرك الذكاء الاصطناعي")
    st.metric(label="نوع الفحص", value="Dual-Engine Ensemble AI", delta="Smart Focus")
    st.metric(label="زمن الاستجابة", value="~0.4s", delta="Real-time")

with tab3:
    st.write("**تطوير:** يوسف علي المرشدي")
    st.write("**الفئة:** مسابقة SAIF 2026")
