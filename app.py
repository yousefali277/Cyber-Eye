import streamlit as st
from PIL import Image
import torch
import math
import cv2
import numpy as np
import tempfile
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor, AutoModelForAudioClassification
import librosa
import soundfile as sf

# ====================== إعداد الصفحة ======================
st.set_page_config(
    page_title="Cyber Eye",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Cyber Eye")
st.caption("نظام كشف التزييف العميق • صور • فيديو • صوت")
st.markdown("---")

# ====================== دالة المعايرة ======================
def calibrate_score(prob_fake, strength=4.3):
    x = (prob_fake - 0.5) * strength
    return 1.0 / (1.0 + math.exp(-x))

# ====================== تحميل موديل الصور ======================
@st.cache_resource
def load_image_model():
    model_name = "Organika/sdxl-detector"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

# ====================== تحميل موديل الصوت ======================
@st.cache_resource
def load_audio_model():
    model_name = "mo-thecreator/Deepfake-audio-detection"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()
    return feature_extractor, model

with st.spinner("جاري تحميل الأنظمة..."):
    img_processor, img_model = load_image_model()
    try:
        audio_extractor, audio_model = load_audio_model()
        audio_ready = True
    except:
        audio_ready = False

# ====================== التبويبات ======================
tab1, tab2, tab3 = st.tabs(["🖼️ صور", "🎬 فيديو", "🔊 صوت"])

# ====================== تبويب الصور ======================
with tab1:
    st.subheader("كشف الصور المزيفة")
    uploaded_img = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png", "webp"], key="img")

    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="الصورة المرفوعة", use_container_width=True)

        if st.button("🚀 بدء تحليل الصورة", key="btn_img"):
            with st.spinner("جاري التحليل..."):
                inputs = img_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = img_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

                labels = img_model.config.id2label
                raw_fake = float(probs[0])
                raw_real = float(probs[1])

                if "ai" not in str(labels[0]).lower() and "fake" not in str(labels[0]).lower():
                    raw_fake, raw_real = raw_real, raw_fake

                calibrated = calibrate_score(raw_fake)
                ai_prob = calibrated * 100
                real_prob = (1 - calibrated) * 100

            with col2:
                st.subheader("النتيجة")
                if ai_prob >= 65:
                    st.error("🚨 تم كشف تزييف عميق")
                    st.write("**مستوى الخطورة:** عالي")
                elif ai_prob >= 45:
                    st.warning("⚠️ صورة مشبوهة")
                    st.write("**مستوى الخطورة:** متوسط")
                else:
                    st.success("✅ صورة حقيقية على الأرجح")
                    st.write("**مستوى الخطورة:** منخفض")

                st.write(f"**نسبة التزييف:** {ai_prob:.1f}%")
                st.progress(ai_prob / 100)
                st.write(f"**نسبة الواقعية:** {real_prob:.1f}%")
                st.progress(real_prob / 100)

# ====================== تبويب الفيديو ======================
with tab2:
    st.subheader("كشف الفيديوهات المزيفة")
    st.info("يتم استخراج فريمات من الفيديو وتحليلها ثم حساب المتوسط")

    uploaded_video = st.file_uploader("ارفع فيديو", type=["mp4", "mov", "avi", "mkv"], key="vid")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(uploaded_video)

        if st.button("🚀 بدء تحليل الفيديو", key="btn_vid"):
            with st.spinner("جاري استخراج الفريمات والتحليل... قد يأخذ وقت"):
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # نأخذ فريم كل ثانية تقريباً (أو كل 15 فريم)
                step = max(1, int(fps))
                scores = []
                frames_analyzed = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frames_analyzed % step == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        inputs = img_processor(images=pil_img, return_tensors="pt")
                        with torch.no_grad():
                            outputs = img_model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                        
                        raw_fake = float(probs[0])
                        if "ai" not in str(img_model.config.id2label[0]).lower():
                            raw_fake = float(probs[1])
                        
                        scores.append(calibrate_score(raw_fake))
                    frames_analyzed += 1

                cap.release()
                os.unlink(video_path)

                if scores:
                    avg_fake = np.mean(scores) * 100
                    max_fake = np.max(scores) * 100

                    st.subheader("نتيجة تحليل الفيديو")
                    st.write(f"تم تحليل **{len(scores)}** فريم")

                    if avg_fake >= 65:
                        st.error("🚨 تم كشف تزييف عميق في الفيديو")
                    elif avg_fake >= 45:
                        st.warning("⚠️ الفيديو مشبوه")
                    else:
                        st.success("✅ الفيديو حقيقي على الأرجح")

                    st.write(f"**متوسط نسبة التزييف:** {avg_fake:.1f}%")
                    st.progress(avg_fake / 100)
                    st.write(f"**أعلى نسبة تزييف في فريم:** {max_fake:.1f}%")
                else:
                    st.error("لم يتم استخراج فريمات صالحة")

# ====================== تبويب الصوت ======================
with tab3:
    st.subheader("كشف الأصوات المزيفة")
    
    if not audio_ready:
        st.warning("موديل الصوت غير متوفر حالياً. جرب لاحقاً أو ثبت المكتبات المطلوبة.")
    else:
        uploaded_audio = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "ogg", "flac", "m4a"], key="aud")

        if uploaded_audio is not None:
            st.audio(uploaded_audio)

            if st.button("🚀 بدء تحليل الصوت", key="btn_aud"):
                with st.spinner("جاري تحليل الصوت..."):
                    # حفظ مؤقت
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tfile.write(uploaded_audio.read())
                    audio_path = tfile.name

                    try:
                        speech, sr = librosa.load(audio_path, sr=16000)
                        inputs = audio_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                        
                        with torch.no_grad():
                            outputs = audio_model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

                        # غالباً 0 = Real ، 1 = Fake (حسب الموديل)
                        fake_prob = float(probs[1]) * 100
                        real_prob = float(probs[0]) * 100

                        st.subheader("نتيجة تحليل الصوت")
                        if fake_prob >= 65:
                            st.error("🚨 تم كشف صوت مزيف (Deepfake Audio)")
                        elif fake_prob >= 45:
                            st.warning("⚠️ الصوت مشبوه")
                        else:
                            st.success("✅ الصوت حقيقي على الأرجح")

                        st.write(f"**نسبة التزييف:** {fake_prob:.1f}%")
                        st.progress(fake_prob / 100)
                        st.write(f"**نسبة الواقعية:** {real_prob:.1f}%")
                        st.progress(real_prob / 100)

                    except Exception as e:
                        st.error(f"حدث خطأ أثناء التحليل: {e}")
                    finally:
                        os.unlink(audio_path)

st.markdown("---")
st.caption("Cyber Eye • تطوير: يوسف المرشدي")
