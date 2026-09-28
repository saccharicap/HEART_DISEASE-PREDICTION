# Heart Attack Prediction - Logistic Regression + Streamlit

## Files
| File | Purpose |
|---|---|
| `heart_attack_analysis.py` | Full pipeline: preprocessing, EDA, tuning, evaluation, saves the model |
| `app.py` | Streamlit web app (single prediction, batch CSV prediction, performance tab) |
| `Heart_Attack.csv` | Dataset |
| `heart_attack_model.joblib` | Trained sklearn Pipeline (created by the analysis script) |
| `model_metadata.json` | Metrics, best params, input ranges used by the app |
| `figures/` | All EDA and evaluation charts (17 PNGs) |
| `run_log.txt` | Console output of the analysis run |

## How to run
```bash
pip install -r requirements.txt
python heart_attack_analysis.py      # (re)creates model + figures with YOUR library versions
streamlit run app.py                 # opens http://localhost:8501
```
Run the analysis script once on your machine before launching the app. A saved
`.joblib` model is tied to the scikit-learn version that created it.

## Deploy free on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `heart_attack_model.joblib`, `model_metadata.json`,
   `figures/` (and the data + analysis script) to a GitHub repo.
2. Go to share.streamlit.io -> New app -> select the repo -> main file `app.py`.
