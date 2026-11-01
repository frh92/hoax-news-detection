import re
import requests
from urllib.parse import urlparse

def is_valid_x_url(url: str) -> bool:
    """
    Validates if the provided URL is a valid X (Twitter) status URL.
    Accepted domains: x.com, twitter.com
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if domain not in ['x.com', 'www.x.com', 'twitter.com', 'www.twitter.com']:
            return False
        
        # Check if it's a status URL
        if not re.search(r'/status/\d+', parsed_url.path):
            return False
            
        return True
    except Exception:
        return False

def extract_text_from_x_url(url: str) -> str:
    """
    Extracts text from an X (Twitter) post URL using FixTweet API.
    """
    try:
        parsed_url = urlparse(url)
        # Transform the URL to the FixTweet API endpoint
        api_url = url.replace(parsed_url.netloc, "api.fxtwitter.com")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 200 and data.get('tweet'):
                text = data['tweet'].get('text', '')
                if text:
                    return text
            return "Postingan ditemukan, tetapi tidak memiliki teks narasi, atau postingan privat/dihapus."
        else:
            return f"Gagal mengambil data dari X. Status Code: {response.status_code}. Postingan mungkin privat atau telah dihapus."
            
    except Exception as e:
        return f"Terjadi kesalahan saat mengambil narasi: {str(e)}"
