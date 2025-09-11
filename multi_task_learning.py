# pip install scikit-learn pandas numpy
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import colorsys
from utils import *

# -----------------------
# Feature engineering
# -----------------------
class ColorFeatures(BaseEstimator, TransformerMixin):
    """
    Input: DataFrame with columns ['mean_R', 'mean_G', 'mean_B'].
    Output features: [R,G,B, r_frac,g_frac,b_frac, hue_deg, sat, val].
    """
    def __init__(self):
        pass
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X, columns=["mean_R","mean_G","mean_B"]) if not isinstance(X, pd.DataFrame) else X.copy()
        R = df["mean_R"].astype(float).to_numpy()
        G = df["mean_G"].astype(float).to_numpy()
        B = df["mean_B"].astype(float).to_numpy()

        S = R + G + B + 1e-8
        r_frac, g_frac, b_frac = R/S, G/S, B/S

        rgb01 = np.stack([R, G, B], axis=1) / 255.0
        hsv = np.array([colorsys.rgb_to_hsv(*row) for row in rgb01])
        hue_deg = hsv[:,0] * 360.0
        sat     = hsv[:,1]
        val     = hsv[:,2]

        feats = np.column_stack([R,G,B, r_frac,g_frac,b_frac, hue_deg,sat,val])
        return feats

# -----------------------
# Model choosers
# -----------------------
def make_classifier(model_type: str):
    if model_type == "svm":
        # scale features for SVM
        return Pipeline([
            ("feat", ColorFeatures()),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))
        ])
    if model_type == "rf":
        return Pipeline([
            ("feat", ColorFeatures()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced"))
        ])
    if model_type == "gb":
        return Pipeline([
            ("feat", ColorFeatures()),
            ("clf", GradientBoostingClassifier())
        ])
    raise ValueError("model_type must be one of {'svm','rf','gb'}")

def make_regressor(model_type: str):
    if model_type == "svm":
        return Pipeline([
            ("feat", ColorFeatures()),
            ("scale", StandardScaler()),
            ("reg", SVR(kernel="rbf"))
        ])
    if model_type == "rf":
        return Pipeline([
            ("feat", ColorFeatures()),
            ("reg", RandomForestRegressor(n_estimators=400, random_state=0))
        ])
    if model_type == "gb":
        return Pipeline([
            ("feat", ColorFeatures()),
            ("reg", GradientBoostingRegressor())
        ])
    raise ValueError("model_type must be one of {'svm','rf','gb'}")

# -----------------------
# Small, safe param grids (tiny, to avoid overfitting on 65 samples)
# -----------------------
def param_grid_classifier(model_type: str):
    if model_type == "svm":
        return {"clf__C":[0.5,1,2], "clf__gamma":["scale","auto"]}
    if model_type == "rf":
        return {"clf__max_depth":[None, 3, 5], "clf__min_samples_leaf":[1, 3, 5]}
    if model_type == "gb":
        return {"clf__n_estimators":[100,200], "clf__learning_rate":[0.05,0.1], "clf__max_depth":[2,3]}
    return {}

def param_grid_regressor(model_type: str):
    if model_type == "svm":
        return {"reg__C":[0.5,1,2], "reg__gamma":["scale","auto"], "reg__epsilon":[0.05,0.1]}
    if model_type == "rf":
        return {"reg__max_depth":[None, 3, 5], "reg__min_samples_leaf":[1, 3, 5]}
    if model_type == "gb":
        return {"reg__n_estimators":[200,400], "reg__learning_rate":[0.05,0.1], "reg__max_depth":[2,3]}
    return {}

# -----------------------
# Training + CV helpers
# -----------------------
@dataclass
class TrainedModels:
    antibiotic_model: any
    strain_model: any
    concentration_model: any
    category_model: any
    le_antibiotic: Optional[LabelEncoder]
    le_strain: Optional[LabelEncoder]
    le_category: Optional[LabelEncoder]

def _encode(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str).values)
    return y_enc, le

def train_with_cv(df: pd.DataFrame,
                  model_type_cls="rf",
                  model_type_reg="gb",
                  antibiotic_col="Antibiotic",
                  category_col="Category",
                  strain_col="Strain",
                  conc_col="Concentration_mgL",
                  k_cls=5,
                  k_reg=5,
                  do_grid=True) -> TrainedModels:
    """
    df must have: ['mean_R','mean_G','mean_B', 'Antibiotic','Strain','Concentration_mgL'] (labels can have NaNs).
    """

    # --- Antibiotic (classification)
    maskA = df[antibiotic_col].notna()
    X_A   = df.loc[maskA, ["mean_R","mean_G","mean_B"]]
    yA    = df.loc[maskA, antibiotic_col]
    yA_enc, leA = _encode(yA)

    skfA = StratifiedKFold(n_splits=min(k_cls, len(np.unique(yA_enc))), shuffle=True, random_state=0)
    clfA = make_classifier(model_type_cls)
    scoring_cls = {"acc": make_scorer(accuracy_score),
                   "f1_macro": make_scorer(f1_score, average="macro", zero_division=0)}
    if do_grid:
        clfA = GridSearchCV(clfA, param_grid_classifier(model_type_cls), cv=skfA, scoring="f1_macro", n_jobs=-1)
    cvA = cross_validate(clfA, X_A, yA_enc, cv=skfA, scoring=scoring_cls, return_train_score=False)
    print(f"[Antibiotic] acc={cvA['test_acc'].mean():.3f}±{cvA['test_acc'].std():.3f} | "
          f"f1_macro={cvA['test_f1_macro'].mean():.3f}±{cvA['test_f1_macro'].std():.3f}")
    clfA.fit(X_A, yA_enc)

    # --- Strain (classification)
    maskB = df[strain_col].notna()
    X_B   = df.loc[maskB, ["mean_R","mean_G","mean_B"]]
    yB    = df.loc[maskB, strain_col]
    yB_enc, leB = _encode(yB)

    skfB = StratifiedKFold(n_splits=min(k_cls, len(np.unique(yB_enc))), shuffle=True, random_state=0)
    clfB = make_classifier(model_type_cls)
    if do_grid:
        clfB = GridSearchCV(clfB, param_grid_classifier(model_type_cls), cv=skfB, scoring="f1_macro", n_jobs=-1)
    cvB = cross_validate(clfB, X_B, yB_enc, cv=skfB, scoring=scoring_cls, return_train_score=False)
    print(f"[Strain]     acc={cvB['test_acc'].mean():.3f}±{cvB['test_acc'].std():.3f} | "
          f"f1_macro={cvB['test_f1_macro'].mean():.3f}±{cvB['test_f1_macro'].std():.3f}")
    clfB.fit(X_B, yB_enc)

    # --- Category (classification)
    maskC = df[category_col].notna()
    X_C   = df.loc[maskC, ["mean_R","mean_G","mean_B"]]
    yC    = df.loc[maskC, category_col]
    yC_enc, leC = _encode(yC)

    skfC = StratifiedKFold(n_splits=min(k_cls, len(np.unique(yC_enc))), shuffle=True, random_state=0)
    clfC = make_classifier(model_type_cls)
    if do_grid:
        clfC = GridSearchCV(clfC, param_grid_classifier(model_type_cls), cv=skfC, scoring="f1_macro", n_jobs=-1)
    cvC = cross_validate(clfC, X_C, yC_enc, cv=skfC, scoring=scoring_cls, return_train_score=False)
    print(f"[Category]   acc={cvC['test_acc'].mean():.3f}±{cvC['test_acc'].std():.3f} | "
          f"f1_macro={cvC['test_f1_macro'].mean():.3f}±{cvC['test_f1_macro'].std():.3f}")
    clfC.fit(X_C, yC_enc)

    # --- Concentration (regression)
    maskD = df[conc_col].notna()
    X_D   = df.loc[maskD, ["mean_R","mean_G","mean_B"]]
    yD    = df.loc[maskD, conc_col].astype(float).values

    kfD = KFold(n_splits=min(k_reg, len(X_D)), shuffle=True, random_state=0)
    regD = make_regressor(model_type_reg)
    scoring_reg = {"MAE": make_scorer(mean_absolute_error, greater_is_better=False),
                   "R2": make_scorer(r2_score)}
    if do_grid:
        regD = GridSearchCV(regD, param_grid_regressor(model_type_reg), cv=kfD, scoring="neg_mean_absolute_error", n_jobs=-1)
    cvD = cross_validate(regD, X_D, yD, cv=kfD, scoring=scoring_reg, return_train_score=False)
    print(f"[Concentration] MAE={-cvD['test_MAE'].mean():.4f}±{cvD['test_MAE'].std():.4f} | "
          f"R2={cvD['test_R2'].mean():.3f}±{cvD['test_R2'].std():.3f}")
    regD.fit(X_D, yD)

    # If you need probability outputs from classifiers, SVM has .predict_proba when probability=True
    return TrainedModels(
        antibiotic_model=clfA, strain_model=clfB, category_model=clfC, concentration_model=regD,
        le_antibiotic=leA, le_strain=leB, le_category=leC
    )

# -----------------------
# Inference helper
# -----------------------
def predict_all(models: TrainedModels, X_df: pd.DataFrame) -> pd.DataFrame:
    X = X_df[["mean_R","mean_G","mean_B"]]
    yA_hat = models.antibiotic_model.predict(X)
    yB_hat = models.strain_model.predict(X)
    conc_hat = models.concentration_model.predict(X)

    # decode labels back to strings
    yA_lbl = models.le_antibiotic.inverse_transform(yA_hat)
    yB_lbl = models.le_strain.inverse_transform(yB_hat)

    return pd.DataFrame({
        "mean_R": X_df["mean_R"],
        "mean_G": X_df["mean_G"],   
        "mean_B": X_df["mean_B"],
        "Antibiotic_true": X_df["Antibiotic"],
        "Strain_true": X_df["Strain"],
        "Concentration_true": X_df["Concentration_mgL"],
        "Antibiotic_pred": yA_lbl,
        "Strain_pred": yB_lbl,
        "Concentration_pred": conc_hat
    })
# -----------------------
# Example usage:
# -----------------------   
if __name__ == "__main__":
    # Example usage
    IN_CSV = "Datasets/s9_wells_rgb_labeled_clinical.csv"
    df = pd.read_csv(IN_CSV)
    df_train = df.iloc[:40]  # use first 40 samples for training (rest can be used for testing)
    df_test  = df.iloc[40:]
    print("Training models with 5-fold CV and grid search...")
    models = train_with_cv(df_train, model_type_cls="rf", model_type_reg="gb", do_grid=True)

    print("\nPredicting on the test data (just as an example)...")
    preds = predict_all(models, df_test)
    preds.to_csv("/content/AgariX/outputs/s9_wells_rgb_predictions.csv", index=False)
    print(preds.head())

    # In Jupyter/Colab: returning the Styler will render the colored table
    # df_with_color_pixel(preds)
    styled = df_with_color_pixel(preds)
    html = styled.to_html()  # pandas Styler -> HTML
    with open("/content/AgariX/outputs/table_with_color_swatches.html", "w", encoding="utf-8") as f:
        f.write(html)
    from IPython.display import HTML, display
    display(HTML(filename="/content/AgariX/outputs/table_with_color_swatches.html"))
