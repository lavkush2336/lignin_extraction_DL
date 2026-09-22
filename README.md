# Lignin Extraction — Deep Learning

A research project predicting lignin removal efficiency in deep eutectic solvent-based biomass fractionation from a high-dimensional experimental dataset, comparing multiple deep learning architectures against a classical baseline.

## Models
- **DNN** — PyTorch feed-forward network, Optuna-tuned, with SHAP-based interpretability (KernelExplainer on a held-out blind test set)
- **TabNet** — attention-based tabular deep learning model, Optuna-tuned (~R² 0.80 on blind test)
- **NODE** — Neural ODE-based model (best blind R² ≈ 0.86–0.87 in tuned runs)
- Benchmarked against an XGBoost baseline

## Approach
- Strict train/validation contract to avoid data leakage: scalers fit only on the training split; a held-out blind set evaluated once at the end
- Feature engineering and mutual-information-based feature selection
- Hyperparameter search via Optuna; ensembling across multiple seeds
- SHAP analysis (summary, dependence, and force plots) to identify the most influential process variables

## Tech Stack
Python · PyTorch · pytorch-tabnet · torchdiffeq (NODE) · scikit-learn · SHAP · Optuna · pandas/NumPy · MongoDB (data storage)

## My Contribution
This was a collaborative research project. My focus was on data preprocessing, feature analysis, and SHAP-based interpretation of the trained models to identify the most influential input features and explain model predictions.

## Repo Structure
- `lignin_dnn_shap.py` — DNN training + SHAP analysis
- `lignin_tabnet.py` — TabNet training
- `lignin_node.py` — NODE training
- `lignin_hf.py` — data handling/feature engineering
- `dnn_accelerated.ipynb` — notebook version of the DNN pipeline

## Setup
```bash
pip install torch torchdiffeq pytorch-tabnet optuna shap scikit-learn pandas numpy pymongo
python lignin_dnn_shap.py   # or lignin_tabnet.py / lignin_node.py
```
Requires a MongoDB connection with the experimental dataset (connection string configured inside each script).
