"""
Text Preprocessing -- disamakan dengan HoaxPreprocessor di notebook training
(case folding, hapus URL/angka/tanda baca, stopword removal, stemming
Sastrawi), supaya teks yang masuk ke IndoBERT saat prediksi punya distribusi
yang sama dengan teks yang dipakai saat melatih model SVM.
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

_stemmer = None
_stopwords = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


def _get_stopwords():
    global _stopwords
    if _stopwords is None:
        _stopwords = set(StopWordRemoverFactory().get_stop_words())
    return _stopwords


def preprocess_text(text: str) -> str:
    """
    Membersihkan dan menyiapkan teks input, identik dengan
    HoaxPreprocessor.clean_text() pada notebook training.
    """
    if not isinstance(text, str):
        return ""

    # 1. Case folding
    text = text.lower()

    # 2. Hapus URL, lalu hapus semua karakter selain huruf & spasi
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Tokenisasi sederhana (whitespace) & Stopword Removal
    stopwords = _get_stopwords()
    tokens = [w for w in text.split() if w not in stopwords]

    # 4. Stemming
    cleaned_text = " ".join(tokens)
    return _get_stemmer().stem(cleaned_text)
