import os
from openai import OpenAI
import streamlit as st

def generate_explanation(text: str, prediction: str, category: str, api_key: str = None) -> str:
    """
    Uses GPT to explain the SVM model's prediction.
    GPT MUST NOT change the prediction or the category.
    It only provides an explanation.
    """
    # Use the passed API key, fallback to secrets if not provided
    if not api_key:
        try:
            if "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
            
    if not api_key:
        return "OpenAI API Key is missing. Silakan masukkan API Key di sidebar sebelah kiri."
        
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return f"Gagal menginisialisasi OpenAI client: {str(e)}"
        
    prompt = (
        f"Berita berikut telah diklasifikasikan sebagai {prediction} dalam kategori {category} "
        f"oleh sistem deteksi berbasis SVM.\n\n"
        f"Berita: \"{text}\"\n\n"
        f"Tugas Anda: Berikan penjelasan singkat, logis, dan profesional mengapa berita tersebut "
        f"dianggap sebagai {prediction}. JANGAN MENGUBAH HASIL KLASIFIKASI ({prediction}) ATAU KATEGORI ({category}). "
        f"Fokus hanya pada analisis elemen teks yang mendukung hasil klasifikasi tersebut."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or whichever model you used in Colab
            messages=[
                {"role": "system", "content": "Anda adalah asisten AI yang bertugas menjelaskan hasil prediksi model machine learning kepada pengguna."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Gagal menghasilkan penjelasan dari GPT: {str(e)}"
