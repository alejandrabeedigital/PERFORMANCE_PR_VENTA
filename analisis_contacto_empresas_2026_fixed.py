
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

    if "fe_test" in df.columns:
        df["fe_test_dt"] = parse_date(df["fe_test"])
    elif "fe_carga_efectiva" in df.columns:
        df["fe_test_dt"] = parse_date(df["fe_carga_efectiva"])
    else:
        df["fe_test_dt"] = pd.NaT

    if "fe_crea_reg" in df.columns:
        df["fe_crea_reg_dt"] = parse_date(df["fe_crea_reg"])
        df["dias_desde_crea_reg"] = (df["fe_test_dt"] - df["fe_crea_reg_dt"]).dt.days
        df.loc[df["dias_desde_crea_reg"] < 0, "dias_desde_crea_reg"] = np.nan
    else:
        df["dias_desde_crea_reg"] = np.nan

    df["registro_nuevo_30d"] = np.where(df["dias_desde_crea_reg"].le(30), "SI", "NO")
    df["registro_nuevo_90d"] = np.where(df["dias_desde_crea_reg"].le(90), "SI", "NO")

    df["bucket_dias_crea_reg"] = pd.cut(
        df["dias_desde_crea_reg"],
        bins=[-1, 7, 30, 90, 180, 365, np.inf],
        labels=["00_0_7d", "01_8_30d", "02_31_90d", "03_91_180d", "04_181_365d", "05_mas_365d"]
    ).astype(object)

    if "fe_reactivacion" in df.columns:
        df["fe_reactivacion_dt"] = parse_date(df["fe_reactivacion"])
        df["dias_desde_reactivacion"] = (df["fe_test_dt"] - df["fe_reactivacion_dt"]).dt.days
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

from sklearn.metrics import roc_auc_score, average_precision_score

CSV_INPUT = "todo_con_prob_descuelgue.csv"
OUT_EXCEL = "analisis_contacto_empresas_2026.xlsx"
TARGET_COL = "target_descuelgue_calc"
SCORE_COL = "prob_descuelgue_modelo"


def safe_auc(y, s):
    tmp = pd.DataFrame({"y": y, "s": s}).dropna()
    if len(tmp) == 0 or tmp["y"].nunique() < 2:
        return np.nan
    return roc_auc_score(tmp["y"], tmp["s"])


def safe_auprc(y, s):
    tmp = pd.DataFrame({"y": y, "s": s}).dropna()
    if len(tmp) == 0 or tmp["y"].nunique() < 2:
        return np.nan
    return average_precision_score(tmp["y"], tmp["s"])


def resumen_grupo(df, group_col):
    rows = []
    for grupo, g in df.groupby(group_col, dropna=False):
        g_score = g[[TARGET_COL, SCORE_COL]].dropna().copy()
        if len(g_score) == 0:
            continue

        tasa_real = g_score[TARGET_COL].mean()
        score_medio = g_score[SCORE_COL].mean()

        rows.append({
            "grupo": grupo,
            "n": len(g),
            "n_con_score": len(g_score),
            "descuelgues": g_score[TARGET_COL].sum(),
            "tasa_descuelgue_real": tasa_real,
            "prob_contacto_media": score_medio,
            "error_pred_real": score_medio - tasa_real,
            "ratio_pred_real": score_medio / tasa_real if tasa_real > 0 else np.nan,
            "auc": safe_auc(g_score[TARGET_COL], g_score[SCORE_COL]),
            "auprc": safe_auprc(g_score[TARGET_COL], g_score[SCORE_COL]),
        })
    return pd.DataFrame(rows)


print("Leyendo datos...")
df = pd.read_csv(CSV_INPUT, low_memory=False)
df.columns = df.columns.str.strip()
df = add_common_features(df)

for col in ["fe_creacion_empresa", TARGET_COL, SCORE_COL]:
    if col not in df.columns:
        raise ValueError(f"Falta columna: {col}")

df[TARGET_COL] = clean_numeric(df[TARGET_COL]).fillna(0).astype(int)
df[SCORE_COL] = clean_numeric(df[SCORE_COL])

df["year_creacion_empresa"] = clean_numeric(df["fe_creacion_empresa"])
current_year = pd.Timestamp.now().year
df.loc[(df["year_creacion_empresa"] < 1900) | (df["year_creacion_empresa"] > current_year), "year_creacion_empresa"] = np.nan

df["grupo_creacion_empresa"] = np.select(
    [
        df["year_creacion_empresa"].eq(2026),
        df["year_creacion_empresa"].eq(2025),
        df["year_creacion_empresa"].between(2023, 2024),
        df["year_creacion_empresa"].between(2020, 2022),
        df["year_creacion_empresa"].lt(2020),
        df["year_creacion_empresa"].isna(),
    ],
    ["2026", "2025", "2023_2024", "2020_2022", "antes_2020", "sin_fecha"],
    default="otros"
)

df["empresa_2026_vs_resto"] = np.where(df["year_creacion_empresa"].eq(2026), "EMPRESA_2026", "RESTO")

print("\nDistribución cohortes:")
print(df["grupo_creacion_empresa"].value_counts(dropna=False))

print("\nDescuelgue real por cohorte:")
print(df.groupby("grupo_creacion_empresa")[TARGET_COL].agg(["count", "sum", "mean"]).sort_index())

res_cohorte = resumen_grupo(df, "grupo_creacion_empresa")
res_2026 = resumen_grupo(df, "empresa_2026_vs_resto")

print("\n\n====================================================")
print("1. CONTACTO POR COHORTE EMPRESA")
print("====================================================")
print(res_cohorte.sort_values("grupo"))

print("\n\n====================================================")
print("2. EMPRESAS 2026 VS RESTO")
print("====================================================")
print(res_2026.sort_values("grupo"))

print("\n\n====================================================")
print("3. INTERPRETACIÓN")
print("====================================================")

row_2026 = res_2026[res_2026["grupo"] == "EMPRESA_2026"]
if row_2026.empty:
    print("No hay empresas 2026 suficientes para analizar.")
else:
    r = row_2026.iloc[0]
    ratio = r["ratio_pred_real"]
    print("Empresas 2026:")
    print(f"- N: {r['n_con_score']:,.0f}")
    print(f"- Descuelgues reales: {r['descuelgues']:,.0f}")
    print(f"- Tasa real de descuelgue: {r['tasa_descuelgue_real']:.4f}")
    print(f"- Probabilidad media predicha: {r['prob_contacto_media']:.4f}")
    print(f"- Ratio pred/real: {ratio:.3f}")
    print(f"- AUC: {r['auc']:.3f}")

    if pd.isna(ratio):
        print("No se puede concluir calibración.")
    elif ratio < 0.8:
        print("Conclusión: el modelo de contacto INFRAESTIMA empresas 2026.")
    elif ratio > 1.2:
        print("Conclusión: el modelo de contacto SOBREESTIMA empresas 2026.")
    else:
        print("Conclusión: el modelo de contacto está razonablemente calibrado en empresas 2026.")

with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
    res_cohorte.to_excel(writer, sheet_name="contacto_por_cohorte", index=False)
    res_2026.to_excel(writer, sheet_name="empresa_2026_vs_resto", index=False)

print(f"\nExcel generado: {OUT_EXCEL}")
