
# Multilayer Hoax Detection System

A Streamlit application to detect hoax news from X (Twitter) using IndoBERT, SVM, and GPT.

## Setup
1. Place your pre-trained `svm_model.pkl` inside the `models/` directory.
2. Provide an OpenAI API Key in your environment or Streamlit secrets if GPT explanation is enabled.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```


