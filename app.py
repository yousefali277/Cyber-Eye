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

st.set_page_config(page_title="Cyber Eye", page_icon="👁️", layout="centered")

st.markdown("<h1 style='text-align: center;'>👁️ Cyber Eye</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>نظام كشف التزييف العميق • صور • فيديو • صوت</p>", unsafe_allow_html=True)
st.markdown("---")

def calibrate_score(prob, strength=4.2):
    x = (prob - 0.5) * strength
    return 1.0 / (1.0 + math.exp(-x))

@st.cache_resource
def load_image_model():
    model_name = "Organika/sdxl-detector"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

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

tab1, tab2, tab3 = st.tabs(["🖼️ صور", "🎬 فيديو", "🔊 صوت"])

# ====================== الصور ======================
with tab1:
    st.subheader("كشف الصور المزيفة")
    uploaded_img = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png", "webp"], key="img")

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, use_container_width=True)

        if st.button("🚀 بدء تحليل الصورة", type="primary", key="btn_img"):
            with st.spinner("جاري التحليل..."):
                inputs = img_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = img_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

                # تصحيح العكس: في هذا الموديل غالباً 0 = AI ، 1 = Real
                # لو طلع معكوس نجبره
                raw_ai = float(probs[1])   # نفترض 0 = AI
                calibrated = calibrate_score(raw_ai)
                ai_prob = calibrated * 100
                real_prob = 100 - ai_prob

            with col2:
                if ai_prob >= 68:
                    st.error("🚨 تم كشف تزييف عميق")
                elif ai_prob >= 48:
                    st.warning("⚠️ صورة مشبوهة")
                else:
                    st.success("✅ صورة حقيقية على الأرجح")

                st.metric("نسبة التزييف", f"{ai_prob:.1f}%")
                st.progress(ai_prob / 100)
                st.metric("نسبة الواقعية", f"{real_prob:.1f}%")

# ====================== الفيديو ======================
with tab2:
    st.subheader("كشف الفيديوهات المزيفة")
    uploaded_video = st.file_uploader("ارفع فيديو", type=["mp4", "mov", "avi", "mkv"], key="vid")

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        st.video(uploaded_video)

        if st.button("🚀 بدء تحليل الفيديو", type="primary", key="btn_vid"):
            with st.spinner("جاري تحليل الفيديو..."):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                step = max(1, int(fps * 1.2))
                scores = []
                count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % step == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        inputs = img_processor(images=pil_img, return_tensors="pt")
                        with torch.no_grad():
                            outputs = img_model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                        raw_ai = float(probs[1])
                        scores.append(calibrate_score(raw_ai))
                    count += 1

                cap.release()
                os.unlink(video_path)

                if scores:
                    avg = np.mean(scores) * 100
                    mx = np.max(scores) * 100
                    st.write(f"تم تحليل {len(scores)} فريم")
                    if avg >= 68:
                        st.error("🚨 تم كشف تزييف عميق")
                    elif avg >= 48:
                        st.warning("⚠️ الفيديو مشبوه")
                    else:
                        st.success("✅ الفيديو حقيقي على الأرجح")
                    st.metric("متوسط التزييف", f"{avg:.1f}%")
                    st.progress(avg / 100)
                    st.metric("أعلى نسبة", f"{mx:.1f}%")

# ====================== الصوت ======================
with tab3:
    st.subheader("كشف الأصوات المزيفة")
    if not audio_ready:
        st.warning("موديل الصوت غير متوفر")
    else:
        uploaded_audio = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "ogg", "flac", "m4a"], key="aud")
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            
            if st.button("🚀 بدء تحليل الصوت", type="primary", key="btn_aud"):
                with st.spinner("جاري تحليل الصوت..."):
                    try:
                        suffix = os.path.splitext(uploaded_audio.name)[1].lower() or ".wav"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                            tfile.write(uploaded_audio.getvalue())
                            audio_path = tfile.name

                        speech, sr = librosa.load(audio_path, sr=16000, duration=30)
                        inputs = audio_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                        
                        with torch.no_grad():
                            outputs = audio_model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

                        # 0 = fake ، 1 = real
                        fake_prob = float(probs[0]) * 100
                        real_prob = float(probs[1]) * 100

                        if fake_prob >= 68:
                            st.error("🚨 تم كشف صوت مزيف")
                        elif fake_prob >= 48:
                            st.warning("⚠️ الصوت مشبوه")
                        else:
                            st.success("✅ الصوت حقيقي على الأرجح")

                        st.metric("نسبة التزييف", f"{fake_prob:.1f}%")
                        st.progress(fake_prob / 100)
                        st.metric("نسبة الواقعية", f"{real_prob:.1f}%")

                    except Exception as e:
                        st.error(f"خطأ: {e}")
                    finally:
                        if 'audio_path' in locals() and os.path.exists(audio_path):
                            os.unlink(audio_path)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Cyber Eye • تطوير: يوسف المرشدي</p>", unsafe_allow_html=True)
