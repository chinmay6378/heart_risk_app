# Heart Disease Risk Predictor (Streamlit)
A calibrated Random Forest model served via Streamlit. Enter clinical metrics; get a **percentage** risk estimate with a gauge and feature insights.

## Quickstart

```bash
git clone <your-repo-url>
cd heart-risk-app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Put dataset at data/heart.csv (UCI/Kaggle heart dataset with 'target' column)
python train_model.py
streamlit run app.py
