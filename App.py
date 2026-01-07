# ================= STREAMLIT IMPLEMENTATION - FINAL REVISION =================
# Status: Dashboard Images Removed, Features Colored, 1-Para, No "AI"
# =============================================================================

import streamlit as st
import numpy as np
import time, json, pickle
from PIL import Image
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="DermaVision",
    page_icon="DermaVision1.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS for Clean UI ----------------
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Headers */
    h1 {
        color: #0277bd;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #0288d1;
        font-weight: 600;
    }

    /* Cards/Containers */
    .stCard {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #f1f3f4;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .stCard:hover {
        transform: translateY(-2px);
    }

    /* Dashboard Feature Cards Styling (Colored) */
    .feature-card {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        color: #333;
    }
    .feature-card:hover {
        transform: scale(1.03);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fbff;
        border-right: 1px solid #eef2f6;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #0277bd;
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #0277bd 0%, #039be5 100%);
        color: white;
        border: none;
        height: 3.2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(2, 119, 189, 0.2);
    }
    .stButton>button:hover {
        box-shadow: 0 6px 8px rgba(2, 119, 189, 0.3);
        transform: translateY(-1px);
    }

    /* Info Boxes */
    .info-box {
        background-color: #e1f5fe;
        border-left: 5px solid #0288d1;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }

    /* Uniform Height Card for Encyclopedia */
    .abc-card {
        background-color: #fcfcfc;
        border: 1px solid #eee;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        min-height: 180px; /* Force equal height */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Radio Button Customization */
    .stRadio > label {
        font-weight: bold;
        color: #444;
        font-size: 1rem;
    }
    div[role="radiogroup"] > label > div:first-child {
        background-color: #e1f5fe;
        border-color: #0288d1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Cancer Information Database ----------------
CANCER_INFO = {
    "melanoma": {
        "nama": "Melanoma",
        "icon": "⚫",
        "deskripsi": "Kanker kulit paling agresif yang berkembang dari sel penghasil pigmen (melanosit). Sering menyerupai tahi lalat namun tumbuh tidak beraturan.",
        "gejala": ["Bentuk asimetris", "Tepi kasar/kabur", "Warna campuran (hitam, coklat, merah)", "Diameter > 6mm"],
        "tingkat_bahaya": "SANGAT TINGGI (Bisa menyebar ke organ lain)",
        "color": "#d32f2f"
    },
    "basal cell carcinoma": {
        "nama": "Karsinoma Sel Basal (BCC)",
        "icon": "🔴",
        "deskripsi": "Jenis kanker kulit paling umum. Biasanya muncul di area yang sering terpapar matahari seperti wajah dan leher.",
        "gejala": ["Benjolan seperti mutiara/lilin", "Lesi datar berwarna coklat/daging", "Luka yang berdarah/berkerak dan tak kunjung sembuh"],
        "tingkat_bahaya": "RENDAH (Jarang menyebar, tapi bisa merusak jaringan sekitar)",
        "color": "#ff9800"
    },
    "squamous cell carcinoma": {
        "nama": "Karsinoma Sel Skuamosa (SCC)",
        "icon": "🟠",
        "deskripsi": "Kanker yang muncul di lapisan skuamosa epidermis. Sering terlihat seperti kulit yang menebal atau bersisik.",
        "gejala": ["Benjolan merah keras", "Bercak datar bersisik", "Luka terbuka yang persisten"],
        "tingkat_bahaya": "SEDANG (Bisa menyebar jika dibiarkan lama)",
        "color": "#ff5722"
    },
    "actinic keratosis": {
        "nama": "Keratosis Aktinik",
        "icon": "🟡",
        "deskripsi": "Bukan kanker, tapi lesi prakanker yang disebabkan kerusakan akibat sinar matahari. Berpotensi berubah menjadi SCC.",
        "gejala": ["Bercak kasar/kering", "Terasa seperti amplas saat diraba", "Warna merah muda/coklat"],
        "tingkat_bahaya": "PRA-KANKER (Perlu dipantau)",
        "color": "#ffc107"
    }
}

# ---------------- Load Artifacts ----------------
@st.cache_resource
def load_artifacts():
    try:
        cnn_model = tf.keras.models.load_model("artifacts_isic/final_model.h5")
        
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="avg")
        base.trainable = False
        feature_extractor = models.Model(base.input, base.output)

        with open("artifacts_cnn_ml/naive_bayes_model.pkl", "rb") as f:
            nb_model = pickle.load(f)
        with open("artifacts_cnn_ml/random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("artifacts_cnn_ml/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        class_names = json.load(open("artifacts_isic/class_names.json"))
        return cnn_model, feature_extractor, nb_model, rf_model, scaler, class_names
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        return None, None, None, None, None, None

cnn_model, feature_extractor, nb_model, rf_model, scaler, CLASS_NAMES = load_artifacts()

# ---------------- Helper Functions ----------------
def is_cancer(label):
    if isinstance(label, (int, float, np.number)): return int(label) == 1
    str_label = str(label).strip().lower()
    if str_label == "1": return True
    if str_label == "0": return False
    cancer_keywords = ["melanoma", "basal cell", "squamous cell", "actinic", "malignant", "ganas", "carcinoma", "cancer"]
    return any(keyword in str_label for keyword in cancer_keywords)

def preprocess_image(img: Image.Image):
    img = img.resize((224,224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_single_image(img):
    if cnn_model is None: return {}
    
    # CNN
    start = time.time()
    cnn_input = preprocess_image(img)
    cnn_pred = cnn_model.predict(cnn_input, verbose=0)
    cnn_time = time.time() - start
    
    cnn_idx = np.argmax(cnn_pred)
    cnn_label = CLASS_NAMES[cnn_idx]
    cnn_conf = np.max(cnn_pred)
    
    # Feature Extraction
    start = time.time()
    features = feature_extractor.predict(cnn_input, verbose=0)
    features_scaled = scaler.transform(features)
    
    # ML Models
    nb_pred_idx = nb_model.predict(features_scaled)[0]
    nb_label = CLASS_NAMES[nb_pred_idx] if isinstance(nb_pred_idx, (int, np.integer)) and nb_pred_idx < len(CLASS_NAMES) else str(nb_pred_idx)
    nb_conf = np.max(nb_model.predict_proba(features_scaled))
    nb_time = time.time() - start
    
    start = time.time()
    rf_pred_idx = rf_model.predict(features_scaled)[0]
    rf_label = CLASS_NAMES[rf_pred_idx] if isinstance(rf_pred_idx, (int, np.integer)) and rf_pred_idx < len(CLASS_NAMES) else str(rf_pred_idx)
    rf_conf = np.max(rf_model.predict_proba(features_scaled))
    rf_time = time.time() - start
    
    return {
        "cnn": {"label": cnn_label, "idx": cnn_idx, "conf": cnn_conf, "time": cnn_time},
        "nb": {"label": nb_label, "idx": nb_pred_idx, "conf": nb_conf, "time": nb_time},
        "rf": {"label": rf_label, "idx": rf_pred_idx, "conf": rf_conf, "time": rf_time}
    }

def create_pdf_report(results_df, timestamp):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1f77b4'), spaceAfter=30, alignment=TA_CENTER)
        elements.append(Paragraph("Laporan Klasifikasi Kanker Kulit", title_style))
        elements.append(Paragraph(f"Tanggal: {timestamp}", styles['Normal']))
        elements.append(Spacer(1, 30))
        
        data = [["No", "File", "CNN", "NB", "RF", "Hasil"]]
        for i, row in results_df.iterrows():
            data.append([str(i+1), row['Nama File'][:20], str(row['Prediksi CNN']), str(row['Prediksi NB']), str(row['Prediksi RF']), "KANKER" if row['Vote Kanker'] >= 2 else "JINAK"])
            
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(t)
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except: return None

# ---------------- 1. Dashboard Page (UPDATED) ----------------
def dashboard_page():
    # Hero Section
    st.markdown("""
    <div style='background: linear-gradient(120deg, #e3f2fd 0%, #90caf9 100%); padding: 3rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05);'>
        <h1 style='color: #0d47a1; font-size: 3rem; margin-bottom: 0.5rem;'>Selamat Datang di DermaVision</h1>
        <p style='font-size: 1.4rem; color: #1565c0; font-weight: 500;'>Deteksi Dini untuk Kulit yang Lebih Sehat</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 1. FITUR UNGGULAN (Colored Cards - Top) ---
    st.markdown("#### ✨ Fitur Unggulan")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class='feature-card' style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-left: 5px solid #1976d2;'>
            <div style='font-size: 2.5rem; margin-bottom: 5px;'>🚀</div>
            <b style='color: #0d47a1; font-size: 1.1rem;'>Cepat & Akurat</b><br>
            <span style='font-size:0.9rem; color:#444;'>Analisis citra dermoskopi dalam hitungan detik dengan presisi tinggi.</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='feature-card' style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-left: 5px solid #388e3c;'>
            <div style='font-size: 2.5rem; margin-bottom: 5px;'>🧠</div>
            <b style='color: #1b5e20; font-size: 1.1rem;'>Ensemble Learning</b><br>
            <span style='font-size:0.9rem; color:#444;'>Menggabungkan 3 model (CNN+NB+RF) untuk hasil yang lebih valid.</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='feature-card' style='background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-left: 5px solid #f57c00;'>
            <div style='font-size: 2.5rem; margin-bottom: 5px;'>📚</div>
            <b style='color: #e65100; font-size: 1.1rem;'>Edukatif</b><br>
            <span style='font-size:0.9rem; color:#444;'>Menyediakan ensiklopedia lengkap mengenai kesehatan kulit.</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.markdown("---")

    # --- 2. TENTANG DERMAVISION (1 Paragraph, No AI) ---
    st.markdown("### 📱 Tentang DermaVision")
    st.write("""
    DermaVision adalah sistem pendukung keputusan medis (*Clinical Decision Support System*) yang dirancang untuk memfasilitasi skrining awal kelainan pada kulit secara efisien. Dengan menerapkan teknologi komputasi canggih berbasis *Ensemble Learning*, aplikasi ini mengintegrasikan model Deep Learning EfficientNetB0 dengan algoritma Machine Learning klasik seperti Naive Bayes dan Random Forest untuk menganalisis karakteristik visual lesi kulit dan memberikan estimasi risiko yang akurat, sehingga dapat digunakan sebagai alat bantu rujukan awal bagi tenaga medis maupun pengguna umum.
    """)
    
    st.markdown("---")

    # --- 3. PENJELASAN UMUM KANKER KULIT (1 Paragraph, Linear Layout) ---
    st.markdown("### 🩺 Apa itu Kanker Kulit?")
    st.write("""
    Kanker kulit adalah kondisi pertumbuhan sel-sel kulit yang tidak normal dan tidak terkendali, yang umumnya dipicu oleh kerusakan DNA akibat paparan radiasi ultraviolet berlebih dari matahari atau alat *tanning*. Kerusakan ini menyebabkan sel-sel bermutasi dan membelah diri secara cepat membentuk massa tumor, namun dengan deteksi dini dan penanganan yang tepat sebelum sel menyebar ke jaringan lain, peluang kesembuhan bagi penderita sangatlah tinggi.
    """)
    
    
    
    # Fakta Penting (Placed BELOW the text, full width)
    st.info("💡 **Fakta Penting:** Sebagian besar kasus kanker kulit dapat disembuhkan secara efektif jika gejala dikenali dan diobati sejak tahap awal.")

    st.markdown("---")

    # --- 4. KLASIFIKASI KLINIS (Benign vs Malignant) - NO IMAGES ---
    st.markdown("### 🔍 Kategori Deteksi")
    st.markdown("Dalam sistem ini, hasil analisis dikelompokkan menjadi dua kategori utama:")
    
    col_benign, col_malig = st.columns(2, gap="large")

    with col_benign:
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 12px; border-left: 5px solid #2e7d32; height: 100%;'>
            <h2 style='color: #2e7d32; margin-top:0;'>🟢 Benign (Jinak)</h2>
            <p><strong>Definisi:</strong> Pertumbuhan kulit yang <strong>tidak berbahaya</strong> dan bukan kanker. Lesi ini tidak menyebar ke bagian tubuh lain.</p>
            <p><strong>Ciri-ciri Umum:</strong></p>
            <ul>
                <li>Bentuk simetris dan teratur.</li>
                <li>Pinggiran rata dan halus.</li>
                <li>Satu warna dominan (biasanya coklat).</li>
                <li>Tidak berubah bentuk secara drastis.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Images removed as requested

    with col_malig:
        st.markdown("""
        <div style='background-color: #ffebee; padding: 20px; border-radius: 12px; border-left: 5px solid #c62828; height: 100%;'>
            <h2 style='color: #c62828; margin-top:0;'>🔴 Malignant (Ganas)</h2>
            <p><strong>Definisi:</strong> Pertumbuhan kulit yang <strong>bersifat kanker</strong>. Berpotensi merusak jaringan sekitar dan menyebar ke organ lain.</p>
            <p><strong>Ciri-ciri Umum (ABCDE):</strong></p>
            <ul>
                <li><strong>A</strong>simetris (bentuk aneh).</li>
                <li><strong>B</strong>order (tepi kasar/kabur).</li>
                <li><strong>C</strong>olor (warna tidak rata/campuran).</li>
                <li><strong>D</strong>iameter (membesar > 6mm).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Images removed as requested

    st.write("")
    st.write("")

    # --- 5. CALL TO ACTION (NAVIGATION) ---
    st.markdown("""
    <div style='text-align: center; margin-top: 20px; margin-bottom: 20px;'>
        <h3 style='color: #444;'>Siap Melakukan Pengecekan?</h3>
        <p>Unggah foto lesi kulit Anda untuk mendapatkan analisis instan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Centered Button
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("🚀 Mulai Deteksi Citra Sekarang", type="primary", use_container_width=True):
            st.session_state.page = "📸 Deteksi Citra"
            st.rerun()

# ---------------- 2. Classification Page ----------------
def classification_page():
    # Modern Header
    st.markdown("""
    <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;'>
        <div>
            <h1 style='margin-bottom: 5px;'>🔬 Analisis Dermoskopi Digital</h1>
            <p style='color: gray; margin-top: 0;'>Deteksi dini Kanker Kulit menggunakan Ensemble Deep Learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- LEGEND / INFO BOX ---
    st.info("ℹ️ **Keterangan Label Kelas:**\n\n🟢 **0 : Benign (Jinak)** — Tidak berbahaya.\n\n🔴 **1 : Malignant (Ganas)** — Berpotensi kanker (Melanoma/Carcinoma).")
    
    # Upload Section
    with st.container():
        st.markdown("### 📤 Upload Citra")
        uploaded_files = st.file_uploader(
            "",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            help="Supported formats: JPG, PNG, JPEG"
        )

    if uploaded_files:
        st.success(f"✅ **{len(uploaded_files)} berkas** berhasil diunggah. Mulai menganalisis...")
        
        all_results = []
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
            st.markdown("---")
            
            # --- CARD LAYOUT FOR SINGLE IMAGE ---
            with st.container():
                st.markdown(f"<h3 style='color:#333;'>📷 Sampel #{idx+1}: {uploaded_file.name}</h3>", unsafe_allow_html=True)
                
                col_img, col_res = st.columns([1, 2.2], gap="large")
                
                img = Image.open(uploaded_file).convert("RGB")
                
                with col_img:
                    st.image(img, use_container_width=True, caption="Citra Input (Dermoskopi)")
                    with st.expander("📐 Metadata Citra"):
                        st.caption(f"Dimensi: {img.size}")
                        st.caption(f"Size: {uploaded_file.size/1024:.1f} KB")
                
                with col_res:
                    with st.spinner("🧠 Menjalankan model ensemble..."):
                        res = predict_single_image(img)
                    
                    if not res: continue

                    # Logic
                    lbl_cnn, lbl_nb, lbl_rf = res["cnn"]["label"], res["nb"]["label"], res["rf"]["label"]
                    c_cnn, c_nb, c_rf = is_cancer(lbl_cnn), is_cancer(lbl_nb), is_cancer(lbl_rf)
                    votes_cancer = sum([c_cnn, c_nb, c_rf])
                    is_majority_cancer = votes_cancer >= 2
                    avg_conf = (res["cnn"]["conf"] + res["nb"]["conf"] + res["rf"]["conf"]) / 3
                    
                    # HTML Table for Results
                    def badge(text, is_bad):
                        color = "#ffebee" if is_bad else "#e8f5e9"
                        text_col = "#c62828" if is_bad else "#2e7d32"
                        icon = "🔴" if is_bad else "🟢"
                        return f"<div style='background-color:{color}; color:{text_col}; padding:4px 8px; border-radius:4px; font-weight:bold; display:inline-block;'>{icon} {text}</div>"

                    res_html = f"""
                    <table style='width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.9rem;'>
                        <tr style='border-bottom: 2px solid #eee; color: #555;'>
                            <th style='text-align:left; padding:8px;'>Model</th>
                            <th style='text-align:left; padding:8px;'>Prediksi</th>
                            <th style='text-align:right; padding:8px;'>Confidence</th>
                            <th style='text-align:right; padding:8px;'>Waktu</th>
                        </tr>
                        <tr>
                            <td style='padding:8px;'><b>CNN (EfficientNet)</b></td>
                            <td style='padding:8px;'>{badge(lbl_cnn, c_cnn)}</td>
                            <td style='text-align:right; padding:8px;'>{res["cnn"]["conf"]:.1%}</td>
                            <td style='text-align:right; padding:8px; color:gray;'>{res["cnn"]["time"]:.3f}s</td>
                        </tr>
                        <tr style='background-color: #f9f9f9;'>
                            <td style='padding:8px;'><b>Naive Bayes</b></td>
                            <td style='padding:8px;'>{badge(lbl_nb, c_nb)}</td>
                            <td style='text-align:right; padding:8px;'>{res["nb"]["conf"]:.1%}</td>
                            <td style='text-align:right; padding:8px; color:gray;'>{res["nb"]["time"]:.3f}s</td>
                        </tr>
                        <tr>
                            <td style='padding:8px;'><b>Random Forest</b></td>
                            <td style='padding:8px;'>{badge(lbl_rf, c_rf)}</td>
                            <td style='text-align:right; padding:8px;'>{res["rf"]["conf"]:.1%}</td>
                            <td style='text-align:right; padding:8px; color:gray;'>{res["rf"]["time"]:.3f}s</td>
                        </tr>
                    </table>
                    """
                    st.markdown(res_html, unsafe_allow_html=True)
                    
                    st.write("") # Spacer

                    # --- VISUALIZATION CHARTS ---
                    col_chart1, col_chart2 = st.columns(2)
                    models_list = ['CNN', 'Naive Bayes', 'Random Forest']
                    confs = [res["cnn"]["conf"], res["nb"]["conf"], res["rf"]["conf"]]
                    times = [res["cnn"]["time"], res["nb"]["time"], res["rf"]["time"]]
                    
                    with col_chart1:
                        fig1, ax1 = plt.subplots(figsize=(5, 2))
                        bars = ax1.barh(models_list, confs, color=['#1976d2', '#ff9800', '#4caf50'], height=0.6)
                        ax1.set_xlim(0, 1.1)
                        ax1.axis('off')
                        for bar, name, val in zip(bars, models_list, confs):
                            ax1.text(0, bar.get_y() + bar.get_height()/2, f" {name}", va='center', ha='left', color='white', fontsize=8, fontweight='bold')
                            ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.1%}", va='center', fontsize=9, fontweight='bold', color='#333')
                        ax1.set_title("📊 Confidence Score", loc='left', fontsize=10, fontweight='bold')
                        st.pyplot(fig1, use_container_width=True)
                        plt.close(fig1)

                    with col_chart2:
                        fig2, ax2 = plt.subplots(figsize=(5, 2))
                        bars2 = ax2.barh(models_list, times, color='#90a4ae', height=0.6)
                        ax2.axis('off')
                        for bar, val in zip(bars2, times):
                            ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, f"{val:.3f}s", va='center', fontsize=9, color='#333')
                        ax2.set_title("⚡ Waktu Inferensi", loc='left', fontsize=10, fontweight='bold')
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)

                    # Consensus Box
                    if is_majority_cancer:
                        st.error(f"⚠️ KESIMPULAN: TERDETEKSI KANKER ({votes_cancer}/3 Vote). Segera periksa ke dokter!")
                    else:
                        st.success(f"✅ KESIMPULAN: JINAK / BENIGN ({3-votes_cancer}/3 Vote). Kondisi aman.")

                    all_results.append({
                        "Nama File": uploaded_file.name,
                        "Prediksi CNN": lbl_cnn, "Prediksi NB": lbl_nb, "Prediksi RF": lbl_rf,
                        "Vote Kanker": votes_cancer, "Confidence Avg": avg_conf
                    })
        
        progress_bar.empty()
        
        # === SUMMARY TABLE SECTION ===
        st.markdown("---")
        if all_results:
            df = pd.DataFrame(all_results)
            df['Status'] = df['Vote Kanker'].apply(lambda x: "🔴 Periksa" if x >= 2 else "🟢 Aman")
            
            st.subheader("📋 Tabel Rekapitulasi")
            st.dataframe(
                df,
                column_config={
                    "Nama File": st.column_config.TextColumn("Nama File", width="medium"),
                    "Confidence Avg": st.column_config.ProgressColumn(
                        "Rata-rata Akurasi", help="Confidence rata-rata", format="%.2f", min_value=0, max_value=1
                    ),
                    "Vote Kanker": st.column_config.NumberColumn(
                        "Skor Bahaya", help="Jml Model Deteksi Kanker", format="%d / 3"
                    ),
                    "Status": st.column_config.TextColumn("Rekomendasi")
                },
                hide_index=True,
                use_container_width=True
            )
            
            c1, c2 = st.columns(2)
            with c1: st.download_button("💾 Unduh CSV", df.to_csv(index=False), "hasil_scan.csv", "text/csv", use_container_width=True)
            with c2: 
                pdf = create_pdf_report(df, datetime.now().strftime("%Y-%m-%d"))
                if pdf: st.download_button("📄 Unduh PDF", pdf, "laporan_medis.pdf", "application/pdf", use_container_width=True)

# ---------------- 3. Education & Info Page ----------------
def information_page():
    # Hero Section
    st.markdown("""
    <div style='background: linear-gradient(120deg, #e3f2fd 0%, #bbdefb 100%); padding: 3rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #0d47a1; margin-bottom: 0.5rem;'>📚 Ensiklopedia Kesehatan Kulit</h1>
        <p style='font-size: 1.2rem; color: #444; max-width: 800px; margin: 0 auto;'>
            Kenali tanda-tanda awal kanker kulit, pahami risikonya, dan pelajari cara pencegahan yang tepat. 
            <b>Deteksi dini menyelamatkan nyawa.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Tabs
    tabs = st.tabs(["🔍 Metode ABCDE", "🦠 Jenis Kanker", "🛡️ Pencegahan", "❓ Mitos vs Fakta"])
    
    # TAB 1: ABCDE (ALIGNED GRID)
    with tabs[0]:
        st.markdown("### Cara Membedakan Tahi Lalat Normal vs Melanoma")
        st.caption("Gunakan panduan ABCDE ini saat melakukan pemeriksaan kulit mandiri:")
        
        # Grid System for Alignment
        # Row 1: A and B
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("""
            <div class='abc-card' style='border-left: 5px solid #0d47a1;'>
                <div style='font-size: 2.5rem;'>🟦 A - Asymmetry</div>
                <p><b>Asimetris:</b> Tahi lalat normal biasanya simetris. Melanoma bentuknya tidak beraturan jika dibagi dua.</p>
            </div>
            """, unsafe_allow_html=True)
        with r1c2:
            st.markdown("""
            <div class='abc-card' style='border-left: 5px solid #2e7d32;'>
                <div style='font-size: 2.5rem;'>🟩 B - Border</div>
                <p><b>Tepi:</b> Pinggiran melanoma seringkali kasar, kabur, atau bergerigi, tidak seperti tahi lalat normal yang halus.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") # Gap

        # Row 2: C and D
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("""
            <div class='abc-card' style='border-left: 5px solid #ff6f00;'>
                <div style='font-size: 2.5rem;'>🟧 C - Color</div>
                <p><b>Warna:</b> Waspada jika tahi lalat memiliki warna campuran (hitam, coklat, merah, putih, atau biru) dalam satu lesi.</p>
            </div>
            """, unsafe_allow_html=True)
        with r2c2:
            st.markdown("""
            <div class='abc-card' style='border-left: 5px solid #c62828;'>
                <div style='font-size: 2.5rem;'>🟥 D - Diameter</div>
                <p><b>Ukuran:</b> Perhatikan jika diameter tahi lalat lebih besar dari 6mm (kira-kira seukuran penghapus pensil).</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("") # Gap

        # Row 3: E (Centered or Full width)
        r3c1, r3c2, r3c3 = st.columns([1, 2, 1])
        with r3c2:
            st.markdown("""
            <div class='abc-card' style='border-left: 5px solid #7b1fa2; text-align: center;'>
                <div style='font-size: 2.5rem;'>🟪 E - Evolving</div>
                <p><b>Berubah:</b> Tahi lalat yang berubah ukuran, bentuk, warna, atau mulai gatal/berdarah adalah tanda bahaya utama.</p>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: Jenis Kanker
    with tabs[1]:
        st.markdown("### Profil Jenis Kanker Kulit")
        for k, v in CANCER_INFO.items():
            with st.expander(f"{v['icon']} {v['nama']}", expanded=False):
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(f"<div style='font-size:4rem; text-align:center;'>{v['icon']}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**Deskripsi:** {v['deskripsi']}")
                    st.markdown("**Gejala Utama:**")
                    for g in v['gejala']:
                        st.markdown(f"- {g}")
                    st.markdown(f"**Tingkat Bahaya:** <span style='color:{v['color']}; font-weight:bold;'>{v['tingkat_bahaya']}</span>", unsafe_allow_html=True)

    # TAB 3: Pencegahan (Improved)
    with tabs[2]:
        st.markdown("### 🛡️ Langkah Perlindungan Diri")
        
        c_prev1, c_prev2 = st.columns(2)
        with c_prev1:
            st.markdown("""
            <div class='info-box'>
            <h4>✅ WAJIB DILAKUKAN</h4>
            <ul>
                <li>Gunakan <b>Sunscreen SPF 30+</b> (Broad Spectrum) setiap hari, bahkan saat mendung.</li>
                <li>Oleskan ulang sunscreen setiap 2 jam atau setelah berenang/berkeringat.</li>
                <li>Kenakan pakaian pelindung: topi lebar, kacamata UV, dan baju lengan panjang.</li>
                <li>Cari tempat teduh, terutama pada pukul <b>10.00 - 14.00</b> saat sinar UV paling kuat.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with c_prev2:
            st.markdown("""
            <div style='background-color: #ffebee; border-left: 5px solid #c62828; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h4>❌ HINDARI</h4>
            <ul>
                <li>Penggunaan <b>Tanning Bed</b> (Meningkatkan risiko melanoma hingga 75% jika dimulai sebelum usia 35).</li>
                <li>Terbakar matahari (Sunburn) berulang kali.</li>
                <li>Mengabaikan tahi lalat yang berubah bentuk atau berdarah.</li>
                <li>Berada di bawah matahari langsung tanpa perlindungan pada bayi < 6 bulan.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    # TAB 4: Mitos vs Fakta (NEW)
    with tabs[3]:
        st.markdown("### 🧐 Meluruskan Mitos Populer")
        
        myths = [
            ("Kanker kulit hanya menyerang orang berkulit putih.", 
             "Salah. Meski risiko lebih tinggi pada kulit terang, orang berkulit gelap pun bisa terkena kanker kulit, seringkali didiagnosis pada tahap yang lebih lanjut (misal: Bob Marley meninggal karena Melanoma)."),
            ("Saya tidak perlu sunscreen jika hari mendung.", 
             "Salah. Hingga 80% sinar UV matahari dapat menembus awan dan merusak kulit Anda."),
            ("Luka pada kulit hanya berbahaya jika terasa sakit.", 
             "Salah. Banyak kanker kulit (seperti BCC atau Melanoma awal) tidak menimbulkan rasa sakit. Perhatikan visualnya, bukan rasanya."),
            ("Tanning bed lebih aman daripada jemur matahari.",
             "Salah. Tanning bed memancarkan sinar UVA intensitas tinggi yang merusak DNA kulit dan sangat karsinogenik.")
        ]
        
        for m, f in myths:
            with st.expander(f"Mitos: {m}"):
                st.info(f"**Fakta:** {f}")

# ---------------- Sidebar & Navigation ----------------
with st.sidebar:
    # 1. Gambar Medis di Navigasi (LOKAL)
    st.image("DermaVision.png", use_container_width=True)
    
    st.markdown("---")
    
    # 2. UI Navigasi (Updated to include Dashboard & Sync with Session State)
    st.markdown("### 🧭 Menu Utama")
    
    # Init Session State
    if 'page' not in st.session_state:
        st.session_state.page = "📊 Dashboard"
    
    # Define Options
    page_options = ["📊 Dashboard", "📸 Deteksi Citra", "📖 Ensiklopedia Kulit"]
    
    # Get current index safely
    try:
        current_index = page_options.index(st.session_state.page)
    except ValueError:
        current_index = 0
        
    # Render Radio Button (Manual Sync)
    selected_page = st.radio(
        "Pilih Halaman:",
        page_options,
        index=current_index,
        label_visibility="collapsed"
    )
    
    # Update state if radio changes manually
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    st.markdown("---")
    
    # Info Model
    st.markdown("#### 🤖 Spesifikasi Model") 
    st.caption("Ensemble Learning System:")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("• EfficientNet")
        st.caption("• Naive Bayes")
    with col_b:
        st.caption("• Random Forest")
        
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:**\nAplikasi ini adalah alat bantu (CDSS). Hasil prediksi **bukan** diagnosis medis final.")
    st.caption("© 2025 MedSkin Project")

# ---------------- Main Router ----------------
if st.session_state.page == "📊 Dashboard":
    dashboard_page()
elif st.session_state.page == "📸 Deteksi Citra":
    classification_page()
else:
    information_page()
