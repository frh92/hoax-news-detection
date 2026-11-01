import streamlit as st
import time

# --- Layout and Configuration ---
st.set_page_config(
    page_title="Hoax Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Minimalist Blue/White Theme
st.markdown(
    """
<style>
    /* Primary Colors */
    :root {
        --primary-blue: #0056b3;
        --light-blue: #e6f2ff;
        --white: #ffffff;
        --white-text: #ffffff;
    }
    
    /* Background */
    .stApp {
        background-color: var(--white);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--primary-blue);
        font-family: 'Inter', sans-serif;
    }
    
    p, span, div {
        color: var(--gray-text);
        font-family: 'Inter', sans-serif;
    }
    
    /* Cards */
    .metric-card {
        background-color: var(--light-blue);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid var(--primary-blue);
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .metric-card-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--primary-blue);
        margin-bottom: 5px;
    }
    
    .metric-card-value {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary-blue) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button * {
        color: var(--white) !important;
    }
    .stButton>button:hover {
        background-color: #004494 !important;
        color: var(--white) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Import Modules ---
from utils import is_valid_x_url, extract_text_from_x_url
from modules.preprocessing import preprocess_text
from modules.category_validation import validate_category_rule_based
from modules.prediction import predict_hoax
from modules.openai_explainer import generate_explanation

# --- Sidebar ---
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/X_logo_2023.svg/1200px-X_logo_2023.svg.png",
        width=50,
    )  # Placeholder for assets/logo.png
    st.title("Tentang Penelitian")
    st.markdown("""
    **Model yang digunakan:**
    * IndoBERT
    * SVM

    **Kategori penelitian:**
    * Kesehatan
    * Politik dan Pemerintahan
    * Ekonomi
    """)

    st.markdown("---")
    st.markdown("### Pengaturan")
    api_key_input = st.text_input(
        "OpenAI API Key (Untuk fitur GPT)",
        type="password",
        help="Masukkan API Key OpenAI Anda agar fitur penjelasan GPT dapat bekerja.",
    )

# --- Main Content ---
st.markdown("<h1>🛡️ Hoax Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3>Implementasi Sistem Deteksi Berita Hoaks Menggunakan IndoBERT dan Support Vector Machine dengan Validasi Kategori</h3>",
    unsafe_allow_html=True,
)
st.write("---")

# --- Session State ---
if "narasi" not in st.session_state:
    st.session_state.narasi = ""
if "url_input" not in st.session_state:
    st.session_state.url_input = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- Input Section ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Input URL Posting X")
    url_input = st.text_input(
        "Paste URL postingan X", placeholder="https://x.com/username/status/123456789"
    )
    if st.button("Ambil Narasi"):
        if url_input:
            if is_valid_x_url(url_input):
                with st.spinner("Mengambil postingan dari X..."):
                    time.sleep(1)  # Simulated network delay
                    extracted_text = extract_text_from_x_url(url_input)
                    st.session_state.narasi = extracted_text
                    st.session_state.url_input = url_input
            else:
                st.error("Sistem hanya menerima URL dari media sosial X (Twitter).")
        else:
            st.warning("Masukkan URL terlebih dahulu.")

with col2:
    st.markdown("#### Narasi Berita")
    narasi_input = st.text_area(
        "Paste teks berita secara manual atau edit hasil ekstraksi",
        value=st.session_state.narasi,
        height=150,
    )
    st.session_state.narasi = narasi_input

col3, col4 = st.columns([2, 10])
with col3:
    if st.button("Analisis Berita", type="primary"):
        if not st.session_state.narasi.strip():
            st.warning("Narasi berita tidak boleh kosong.")
        else:
            st.session_state.analysis_done = True
with col4:
    if st.button("Reset"):
        st.session_state.narasi = ""
        st.session_state.url_input = ""
        st.session_state.analysis_done = False
        st.rerun()

st.write("---")

# --- Pipeline Execution ---
if st.session_state.analysis_done and st.session_state.narasi.strip():
    text_to_analyze = st.session_state.narasi

    # 1. Preprocessing
    with st.spinner("Melakukan preprocessing..."):
        time.sleep(0.5)
        clean_text = preprocess_text(text_to_analyze)

    # 2. Category Validation Gate
    with st.spinner("Melakukan validasi kategori..."):
        time.sleep(0.5)
        is_valid_cat, category_msg = validate_category_rule_based(clean_text)

    if not is_valid_cat:
        st.error(category_msg)
        st.info(
            "Penelitian ini hanya mendukung: \n- Kesehatan \n- Politik dan Pemerintahan \n- Ekonomi"
        )
        st.stop()  # Halt pipeline

    # 3 & 4. IndoBERT Embedding & SVM Classification
    with st.spinner("Menghasilkan embedding IndoBERT dan Mengklasifikasikan berita..."):
        try:
            prediction, accuracy = predict_hoax(clean_text)
        except Exception as e:
            st.error(f"Gagal melakukan klasifikasi: {e}")
            st.stop()

    # 5. GPT Explanation
    with st.spinner("Menghasilkan penjelasan..."):
        gpt_explanation = generate_explanation(
            clean_text, prediction, category_msg, api_key=api_key_input
        )

    # --- Results Display ---
    st.markdown("## Hasil Analisis")

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-card-title">Kategori Berita</div>
            <div class="metric-card-value">{category_msg}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-card-title">Akurasi Hasil Deteksi</div>
            <div class="metric-card-value">{accuracy:.2f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with r_col2:
        pred_color = "#dc3545" if prediction == "HOAKS" else "#28a745"
        st.markdown(
            f"""
        <div class="metric-card" style="border-left-color: {pred_color};">
            <div class="metric-card-title">Hasil Prediksi</div>
            <div class="metric-card-value" style="color: {pred_color};">{prediction}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-card-title">GPT Explanation</div>
            <div style="color: var(--gray-text); font-size: 14px; margin-top: 10px;">{gpt_explanation}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.session_state.analysis_done = False
