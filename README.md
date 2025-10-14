# Resume-Screening-App

A small Streamlit app that predicts a resume's job category using a TF-IDF vectorizer and a classifier (Logistic Regression by default). This repository contains a dataset, helper scripts, and pretrained pickled artifacts for quick testing.

## Contents

- `app.py` - Streamlit application. Upload a PDF/DOCX/TXT resume and get a predicted category.
- `train_model.py` - Script to train TF-IDF + LogisticRegression from `UpdatedResumeDataSet.csv` and save `tfidf.pkl`, `clf.pkl`, and `encoder.pkl`.
- `extract_and_predict.py` - Helper to extract text from bundled sample PDFs and run predictions (diagnostics).
- `diag.py` - Diagnostic script to inspect model classes and run a few sample predictions.
- `UpdatedResumeDataSet.csv` - Dataset used to train the model (CSV with `Category` and `Resume` columns).
- `clf.pkl`, `tfidf.pkl`, `encoder.pkl` - Pickled artifacts included for convenience (already in the repo).

## Quickstart (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install requirements and extras:

```powershell
C:/Users/DELL/Resume-Screening-App/.venv/Scripts/python.exe -m pip install -r reqiurements.txt streamlit python-docx PyPDF2
```

3. (Optional) Train a fresh model from the dataset. If you want to re-train and overwrite the packaged pickles:

```powershell
C:/Users/DELL/Resume-Screening-App/.venv/Scripts/python.exe train_model.py
```

4. Run the Streamlit app:

```powershell
C:/Users/DELL/Resume-Screening-App/.venv/Scripts/python.exe -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Notes & Troubleshooting

- If uploaded resumes are always predicted as a single class (for example, "HR"), try the following checks:
  - Use `diag.py` to see the model's class mapping and sample predictions.
  - Run `extract_and_predict.py` to confirm the PDF/DOCX text extraction returns non-empty, meaningful text (sometimes PDFs are image-based and need OCR).
  - Retrain with `train_model.py` if your dataset or label distribution changed.

- Large binary files (`clf.pkl`) are included for convenience in this repo. You may want to remove them from git and store them elsewhere for production.

## Next steps / Improvements

- Improve `app.py` to show prediction probabilities and warn when extracted text is too short.
- Add OCR fallback (Tesseract) for image-based PDFs.
- Add unit tests for extraction and prediction functions.

## License

This repository contains example code for demonstration and educational purposes. Update the license to match your needs.
# Resume-Screening-App
Resume Screening App With Python and Machine Learning 
