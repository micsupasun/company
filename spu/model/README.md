# Vaccine Outcome Prediction

Streamlit web application for predicting vaccine-related outcomes from Thai symptom text and patient age using a trained Character CNN model.

## Live Demo

Open the public web app:

https://micsupasun.github.io/vaccine-outcome-prediction/

## Files

- `app.py` - Streamlit web interface.
- `vaccine_predictor.py` - Model loading and prediction logic.
- `best_charcnn_model.pt` - Trained PyTorch checkpoint.
- `requirements.txt` - Python dependencies for local use or Streamlit Cloud deployment.

## Run Locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Deployment

This project can be deployed on Streamlit Cloud. Upload these files to the same GitHub repository:

- `app.py`
- `vaccine_predictor.py`
- `best_charcnn_model.pt`
- `requirements.txt`

Set the Streamlit entry point to:

```text
app.py
```

## Note

The model output is intended as decision support and should be interpreted with clinical judgment.
