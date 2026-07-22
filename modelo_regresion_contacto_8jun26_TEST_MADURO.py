import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.utils.class_weight import compute_sample_weight

from common_pr_venta_8jun26 import add_common_features, clean_numeric, parse_date


ARCHIVO_IN = "t_pr_venta_update_8jun26.csv"
ARCHIVO_OUT = "todo_con_prob_descuelgue.csv"
TARGET = "target_descuelgue_calc"
MODELO_OUT = "modelo_prob_descuelgue.joblib"
OUT_FIG_DIR = "graficos_modelo"


# ============================================================
# SPLIT TEMPORAL MADURO
# ============================================================
# Por defecto:
# - TEST = carga 2026-04-28
# - TRAIN = todas las cargas anteriores a 2026-04-28
#
# Evitamos 2026-04-29 y 2026-05-05 porque parecen cargas anómalas/inmaduras
# para contacto: baja tasa real y pocos descuelgues por registro.

TEST_DATES = ["2026-04-28"]

# Si prefieres probar 21/04 + 28/04, cambia arriba por:
# TEST_DATES = ["2026-04-21", "2026-04-28"]


def temporal_mature_train_test_split(df_model, target_col):
    if "fe_carga_efectiva" not in df_model.columns:
        raise ValueError("No existe fe_carga_efectiva. No se puede hacer split temporal.")

    df_model = df_model.copy()
    df_model["fe_carga_efectiva_dt"] = parse_date(df_model["fe_carga_efectiva"]).dt.normalize()

    test_dates_dt = [pd.to_datetime(x).normalize() for x in TEST_DATES]

    mask_test = df_model["fe_carga_efectiva_dt"].isin(test_dates_dt)
    if mask_test.sum() == 0:
        raise ValueError(f"No hay filas para TEST_DATES={TEST_DATES}")

    min_test_date = min(test_dates_dt)
    mask_train = df_model["fe_carga_efectiva_dt"] < min_test_date

    if mask_train.sum() == 0:
        raise ValueError("El split temporal deja el train vacío.")

    y_test_tmp = df_model.loc[mask_test, target_col]

    print("\n--- SPLIT TEMPORAL MADURO ---")
    print("Fechas usadas en TEST:")
    for f in sorted(test_dates_dt):
        print("-", pd.Timestamp(f).date())

    print("\nRango TRAIN:")
    print(
        df_model.loc[mask_train, "fe_carga_efectiva_dt"].min(),
        "->",
        df_model.loc[mask_train, "fe_carga_efectiva_dt"].max()
    )

    print("\nRango TEST:")
    print(
        df_model.loc[mask_test, "fe_carga_efectiva_dt"].min(),
        "->",
        df_model.loc[mask_test, "fe_carga_efectiva_dt"].max()
    )

    print("\nFilas TRAIN:", mask_train.sum())
    print("Filas TEST:", mask_test.sum())
    print("Positivos TEST:", int(y_test_tmp.sum()))
    print("Negativos TEST:", int(len(y_test_tmp) - y_test_tmp.sum()))
    print("Tasa TEST:", y_test_tmp.mean())

    print("\nDistribución fechas TEST:")
    print(
        df_model.loc[mask_test, "fe_carga_efectiva_dt"]
        .value_counts()
        .sort_index()
    )

    return mask_train, mask_test



class PriorCorrectedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, eps=1e-15):
        self.base_estimator = base_estimator
        self.eps = eps

    def fit(self, X, y):
        y_arr = np.asarray(y).astype(int)
        self.base_estimator.fit(X, y_arr)
        self.observed_prior_ = float(y_arr.mean())

        class_weight = None
        if hasattr(self.base_estimator, "named_steps") and "clf" in self.base_estimator.named_steps:
            class_weight = getattr(self.base_estimator.named_steps["clf"], "class_weight", None)
        else:
            class_weight = getattr(self.base_estimator, "class_weight", None)

        sample_weight = (
            np.ones_like(y_arr, dtype=float)
            if class_weight is None
            else compute_sample_weight(class_weight=class_weight, y=y_arr)
        )

        self.model_prior_ = float(np.average(y_arr, weights=sample_weight))
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _apply_prior_correction(self, p_model):
        p_model = np.asarray(p_model, dtype=float)
        p_model = np.clip(p_model, self.eps, 1 - self.eps)

        pi_real = np.clip(self.observed_prior_, self.eps, 1 - self.eps)
        pi_model = np.clip(self.model_prior_, self.eps, 1 - self.eps)

        num = p_model * (pi_real / pi_model)
        den = num + (1 - p_model) * ((1 - pi_real) / (1 - pi_model))

        return np.clip(num / den, self.eps, 1 - self.eps)

    def predict_proba_raw(self, X):
        return self.base_estimator.predict_proba(X)

    def predict_proba(self, X):
        p_raw = self.predict_proba_raw(X)[:, 1]
        p_corr = self._apply_prior_correction(p_raw)
        return np.column_stack([1 - p_corr, p_corr])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



print("Leyendo datos...")

df_raw = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas al cargar: {len(df_raw):,}")

df = add_common_features(df_raw)

if TARGET not in df.columns:
    raise ValueError("No existe target_descuelgue_calc. Revisa camp_total_descuelgues.")

df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

features_num = ["dias_desde_crea_reg"]
features_cat = [
    "sin_gmb",
    "movil",
    "ct_merclie",
    "con_web",
    "con_local",
    "cat_contact",
    "ant_empresa",
    "algun_contacto",
    "registro_nuevo_30d",
    "registro_nuevo_90d",
    "bucket_dias_crea_reg",
    "tiene_reactivacion",
]

features_num = [c for c in features_num if c in df.columns]
features_cat = [c for c in features_cat if c in df.columns]

for c in features_num:
    df[c] = clean_numeric(df[c])

for c in features_cat:
    df[c] = df[c].astype(object)

print("\nFeatures num:", features_num)
print("Features cat:", features_cat)
print(f"Filas modelo contacto: {len(df):,}")
print(f"Tasa descuelgue global: {df[TARGET].mean():.6f}")

cols_modelo = features_num + features_cat + [TARGET, "fe_carga_efectiva"]
faltan = [c for c in cols_modelo if c not in df.columns]

if faltan:
    raise ValueError(f"Faltan columnas: {faltan}")

df_model = df[cols_modelo].copy()

X = df_model[features_num + features_cat]
y = df_model[TARGET]

transformers = []

if features_num:
    transformers.append((
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        features_num,
    ))

if features_cat:
    transformers.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]),
        features_cat,
    ))

preprocess = ColumnTransformer(transformers=transformers)

clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="lbfgs",
)

base_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("clf", clf),
])

model = PriorCorrectedClassifier(base_estimator=base_pipeline)

mask_train, mask_test = temporal_mature_train_test_split(df_model, TARGET)

X_train = X.loc[mask_train]
X_test = X.loc[mask_test]
y_train = y.loc[mask_train]
y_test = y.loc[mask_test]

model.fit(X_train, y_train)

proba_test_raw = model.predict_proba_raw(X_test)[:, 1]
proba_test = model.predict_proba(X_test)[:, 1]

print("\n--- MÉTRICAS TEST MADURO CONTACTO ---")
print(f"Media real:               {y_test.mean():.6f}")
print(f"Media predicha raw:       {proba_test_raw.mean():.6f}")
print(f"Media predicha corregida: {proba_test.mean():.6f}")
print(f"AUC raw:                  {roc_auc_score(y_test, proba_test_raw):.6f}")
print(f"AUC corregida:            {roc_auc_score(y_test, proba_test):.6f}")
print(f"AUPRC raw:                {average_precision_score(y_test, proba_test_raw):.6f}")
print(f"AUPRC corregida:          {average_precision_score(y_test, proba_test):.6f}")
print(f"Brier corregida:          {brier_score_loss(y_test, proba_test):.6f}")
print(f"LogLoss corregida:        {log_loss(y_test, proba_test):.6f}")

df_eval = X_test.copy()
df_eval[TARGET] = y_test.values
df_eval["score"] = proba_test

df_eval = df_eval.sort_values("score", ascending=False).reset_index(drop=True)
df_eval["decil"] = pd.qcut(df_eval.index, 10, labels=False, duplicates="drop") + 1

tabla = df_eval.groupby("decil")[TARGET].agg(["count", "mean", "sum"])
tabla = tabla.rename(columns={"mean": "tasa_descuelgue", "sum": "descuelgues"})
tabla["lift_vs_media"] = tabla["tasa_descuelgue"] / y_test.mean()

print("\n--- LIFT POR DECILES CONTACTO - TEST MADURO ---")
print(tabla)

os.makedirs(OUT_FIG_DIR, exist_ok=True)

plt.figure(figsize=(9, 5))
plt.bar(tabla.index.astype(str), tabla["tasa_descuelgue"])
plt.title("Tasa de descuelgue por decil - test maduro (1 = TOP)")
plt.xlabel("Decil")
plt.ylabel("Tasa de descuelgue")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_FIG_DIR, "maduro_tasa_descuelgue_por_decil.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.show()

tabla.to_csv("lift_deciles_contacto_maduro.csv")

X_all = df[features_num + features_cat]

df["prob_descuelgue_modelo_raw"] = model.predict_proba_raw(X_all)[:, 1]
df["prob_descuelgue_modelo"] = model.predict_proba(X_all)[:, 1]

feature_names = model.base_estimator.named_steps["preprocess"].get_feature_names_out()
coefs = model.base_estimator.named_steps["clf"].coef_[0]

df_coefs = (
    pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
    .sort_values("abs_coef", ascending=False)
)

df_coefs.to_csv("coeficientes_modelo_descuelgue.csv", index=False)

print("\n--- TOP COEFICIENTES CONTACTO ---")
print(df_coefs.head(30))

df.to_csv(ARCHIVO_OUT, index=False)
joblib.dump(model, MODELO_OUT)

print(f"\nGuardado CSV: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")
print(f"Guardado modelo: {MODELO_OUT}")
