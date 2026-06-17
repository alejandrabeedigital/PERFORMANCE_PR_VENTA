
import pandas as pd
import numpy as np


def clean_numeric(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NAN": np.nan, "NaN": np.nan}),
        errors="coerce"
    )


def parse_date(s):
    return pd.to_datetime(
        s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NAN": np.nan, "NaN": np.nan}),
        errors="coerce"
    )


def clean_bool_01(s):
    return (
        s.astype(str).str.strip().str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "SI": 1, "SÍ": 1, "NO": 0})
    )


def safe_str_series(s):
    return s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NAN": np.nan, "NaN": np.nan})


def add_common_features(df):
    df = df.copy()

    if "camp_total_descuelgues" in df.columns:
        df["camp_total_descuelgues"] = clean_numeric(df["camp_total_descuelgues"])
        df["target_descuelgue_calc"] = np.where(df["camp_total_descuelgues"].fillna(0) > 0, 1, 0)

    if "fe_carga_efectiva" in df.columns:
        df["fecha_ref_registro_dt"] = parse_date(df["fe_carga_efectiva"])
        df["fecha_ref_registro_origen"] = "fe_carga_efectiva"
    elif "fe_carga" in df.columns:
        df["fecha_ref_registro_dt"] = parse_date(df["fe_carga"])
        df["fecha_ref_registro_origen"] = "fe_carga"
    else:
        df["fecha_ref_registro_dt"] = pd.NaT
        df["fecha_ref_registro_origen"] = "sin_fecha_ref"

    if "fe_crea_reg" in df.columns:
        df["fe_crea_reg_dt"] = parse_date(df["fe_crea_reg"])
        df["dias_desde_crea_reg"] = (df["fecha_ref_registro_dt"] - df["fe_crea_reg_dt"]).dt.days
        df.loc[df["dias_desde_crea_reg"] < 0, "dias_desde_crea_reg"] = np.nan
    else:
        df["dias_desde_crea_reg"] = np.nan

    df["registro_nuevo_30d"] = np.where(df["dias_desde_crea_reg"].le(30), "SI", "NO")
    df["registro_nuevo_90d"] = np.where(df["dias_desde_crea_reg"].le(90), "SI", "NO")

    df["bucket_dias_crea_reg"] = pd.cut(
        df["dias_desde_crea_reg"],
        bins=[-1, 7, 30, 90, 365, 730, np.inf],
        labels=["00_0_7d", "01_8_30d", "02_31_90d", "03_91_365d", "04_1_2_anios", "05_mas_2_anios"]
    ).astype(object)

    if "fe_reactivacion" in df.columns:
        df["fe_reactivacion_dt"] = parse_date(df["fe_reactivacion"])
        df["dias_desde_reactivacion"] = (df["fecha_ref_registro_dt"] - df["fe_reactivacion_dt"]).dt.days
        df.loc[df["dias_desde_reactivacion"] < 0, "dias_desde_reactivacion"] = np.nan
        df["tiene_reactivacion"] = np.where(df["fe_reactivacion_dt"].notna(), "SI", "NO")
    else:
        df["dias_desde_reactivacion"] = np.nan
        df["tiene_reactivacion"] = "NO"

    if "con_web" not in df.columns:
        if "website" in df.columns:
            w = safe_str_series(df["website"])
            df["con_web"] = np.where(w.notna(), "SI", "NO")
        else:
            df["con_web"] = "DESCONOCIDO"

    if "excliente_cat" not in df.columns:
        if "excliente" in df.columns:
            e = clean_bool_01(df["excliente"])
            df["excliente_cat"] = np.where(e.eq(1), "EXCLIENTE", "NO_EXCLIENTE")
            df.loc[e.isna(), "excliente_cat"] = "DESCONOCIDO"
        else:
            df["excliente_cat"] = "DESCONOCIDO"

    if "sin_intentos_recientes" not in df.columns:
        if "intentos_ult6m" in df.columns:
            intentos = clean_numeric(df["intentos_ult6m"])
            df["sin_intentos_recientes"] = np.where(intentos.fillna(0).eq(0), "SI", "NO")
        else:
            df["sin_intentos_recientes"] = "DESCONOCIDO"

    if "algun_contacto" not in df.columns:
        if "n_contactado_ult6m" in df.columns:
            n_cont = clean_numeric(df["n_contactado_ult6m"])
            df["algun_contacto"] = np.where(n_cont.fillna(0).gt(0), "SI", "NO")
        else:
            df["algun_contacto"] = "DESCONOCIDO"

    if "origen_sc_o_no" not in df.columns:
        if "origen" in df.columns:
            origen = safe_str_series(df["origen"]).str.upper()
            df["origen_sc_o_no"] = np.where(origen.str.contains("SC", na=False), "SC", "NO_SC")
            df.loc[origen.isna(), "origen_sc_o_no"] = "DESCONOCIDO"
        else:
            df["origen_sc_o_no"] = "DESCONOCIDO"

    if "ant_empresa" not in df.columns:
        if "fe_creacion_empresa" in df.columns:
            year = clean_numeric(df["fe_creacion_empresa"])
            edad = pd.Timestamp.today().year - year
            df["ant_empresa"] = pd.cut(
                edad,
                bins=[-np.inf, 0, 2, 5, 10, np.inf],
                labels=["00_mismo_anio", "01_1_2_anios", "02_3_5_anios", "03_6_10_anios", "04_mas_10_anios"]
            ).astype(object)
            df.loc[year.isna(), "ant_empresa"] = "DESCONOCIDO"
        else:
            df["ant_empresa"] = "DESCONOCIDO"

    if "sin_gmb" not in df.columns:
        if "claim_business" in df.columns:
            cb = clean_bool_01(df["claim_business"])
            df["sin_gmb"] = np.where(cb.eq(1), "NO", "SI")
            df.loc[cb.isna(), "sin_gmb"] = "DESCONOCIDO"
        else:
            df["sin_gmb"] = "DESCONOCIDO"

    for c in [
        "movil", "ct_merclie", "con_local", "cat_contact", "con_web", "excliente_cat",
        "origen_sc_o_no", "sin_gmb", "sin_intentos_recientes", "ant_empresa", "algun_contacto",
        "registro_nuevo_30d", "registro_nuevo_90d", "bucket_dias_crea_reg", "tiene_reactivacion"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(object)

    return df

import statsmodels.api as sm

ARCHIVO = "todo_con_resultados_17.csv"
TARGET = "ganada"
CONTACT_TARGET = "target_descuelgue_calc"

print("Leyendo datos...")
df = pd.read_csv(ARCHIVO, low_memory=False)
print(f"Filas leídas: {len(df):,}")

df = add_common_features(df)
df[TARGET] = clean_numeric(df[TARGET])
df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

if CONTACT_TARGET not in df.columns:
    raise ValueError(f"No existe {CONTACT_TARGET}")

df = df[df[CONTACT_TARGET].eq(1)].copy()

features_num = ["q_rk_score", "dias_desde_crea_reg"]
features_cat = [
    "ct_merclie", "excliente_cat", "origen_sc_o_no", "con_web", "sin_gmb",
    "movil", "sin_intentos_recientes", "ant_empresa", "registro_nuevo_30d",
    "registro_nuevo_90d", "bucket_dias_crea_reg", "tiene_reactivacion"
]

features_num = [c for c in features_num if c in df.columns]
features_cat = [c for c in features_cat if c in df.columns]
features = features_num + features_cat

for c in features_num:
    df[c] = clean_numeric(df[c])
for c in features_cat:
    df[c] = df[c].astype(object)

df_model = df[features + [TARGET]].dropna().copy()

print(f"Filas usadas para inferencia venta post-contacto: {len(df_model):,}")
print(f"Tasa base venta post-contacto: {df_model[TARGET].mean():.6f}")

X = pd.get_dummies(df_model[features], drop_first=True)
y = df_model[TARGET]
X = X.astype(float)

for col in features_num:
    if col in X.columns:
        sd = X[col].std()
        if sd is not None and sd > 0:
            X[col] = (X[col] - X[col].mean()) / sd

cols_sin_var = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if cols_sin_var:
    print("\nColumnas eliminadas sin variación:")
    print(cols_sin_var)
    X = X.drop(columns=cols_sin_var)

duplicadas = X.columns[X.T.duplicated()].tolist()
if duplicadas:
    print("\nColumnas eliminadas por duplicadas/colineales exactas:")
    print(duplicadas)
    X = X.drop(columns=duplicadas)

X = sm.add_constant(X, has_constant="add")

print("\nShape X:", X.shape)
print("Shape y:", y.shape)

model = sm.Logit(y, X)

try:
    result = model.fit(method="lbfgs", maxiter=5000, disp=True)
except Exception as e:
    print("\nFallo lbfgs, probando fit_regularized...")
    print("Motivo:", e)
    result = model.fit_regularized(method="l1", alpha=0.01, maxiter=5000, disp=True)

print("\n==============================")
print("RESUMEN VENTA POST-CONTACTO")
print("==============================")
try:
    print(result.summary())
except Exception as e:
    print("No se pudo imprimir summary completo:", e)

params = result.params
try:
    pvals = result.pvalues
except Exception:
    pvals = pd.Series(np.nan, index=params.index)

odds_ratios = pd.DataFrame({
    "variable": params.index,
    "coef": params.values,
    "odds_ratio": np.exp(params.values),
    "p_value": pvals.reindex(params.index).values
}).sort_values("coef", ascending=False)

odds_ratios.to_csv("odds_ratios_modelo_venta.csv", index=False)

print("\nODDS RATIOS / COEFICIENTES VENTA")
print(odds_ratios.head(80))

try:
    marginal = result.get_margeff()
    mfx_summary = marginal.summary_frame().reset_index()
    mfx_summary = mfx_summary.rename(columns={"index": "variable", "dy/dx": "dy_dx", "Pr(>|z|)": "p_value"})
    df_excel = mfx_summary[["variable", "dy_dx", "p_value"]].copy()
    df_excel["dy_dx_10000"] = df_excel["dy_dx"] * 10000
    df_excel["significancia"] = np.where(df_excel["p_value"] < 0.05, "S", "NS")
    df_excel = df_excel.sort_values(["p_value", "variable"]).reset_index(drop=True)
    print("\nEFECTOS MARGINALES STATSMODELS")
    print(df_excel.head(80))
except Exception as e:
    print("\nNo se pudieron calcular efectos marginales con covarianza statsmodels.")
    print("Motivo:", e)
    print("Calculando aproximación manual: coef * media(p*(1-p)).")
    pred = result.predict(X)
    scale = np.mean(pred * (1 - pred))
    df_excel = pd.DataFrame({"variable": params.index, "dy_dx": params.values * scale, "p_value": np.nan})
    df_excel = df_excel[df_excel["variable"] != "const"].copy()
    df_excel["dy_dx_10000"] = df_excel["dy_dx"] * 10000
    df_excel["significancia"] = "NA"
    df_excel = df_excel.sort_values("dy_dx_10000", ascending=False).reset_index(drop=True)
    print("\nEFECTOS MARGINALES APROXIMADOS")
    print(df_excel.head(80))

df_excel.to_csv("tabla_efectos_marginales_excel_venta.csv", index=False, sep=";", decimal=",")
# =========================
# GRÁFICO IMPORTANCIA VARIABLES
# =========================

import os
import matplotlib.pyplot as plt

OUT_FIG_DIR = "graficos_modelo"
os.makedirs(OUT_FIG_DIR, exist_ok=True)

df_plot = df_excel.copy()

df_plot = df_plot[
    df_plot["variable"].astype(str) != "const"
].copy()

df_plot["abs_impacto"] = df_plot["dy_dx_10000"].abs()

df_plot = (
    df_plot
    .sort_values("abs_impacto", ascending=False)
    .head(25)
    .sort_values("dy_dx_10000")
)

plt.figure(figsize=(10, 8))

plt.barh(
    df_plot["variable"],
    df_plot["dy_dx_10000"]
)

plt.axvline(0, linestyle="--")

plt.title("Importancia de variables - efecto marginal aproximado")
plt.xlabel("Cambio estimado por cada 10.000 registros")
plt.ylabel("Variable")

plt.tight_layout()

nombre_grafico = (
    "importancia_variables_venta.png"
    if "venta" in __file__.lower()
    else "importancia_variables_contacto.png"
)

plt.savefig(
    os.path.join(OUT_FIG_DIR, nombre_grafico),
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"Gráfico de importancia guardado en: {os.path.join(OUT_FIG_DIR, nombre_grafico)}")
print("\nGuardado: odds_ratios_modelo_venta.csv")
print("Guardado: tabla_efectos_marginales_excel_venta.csv")