import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

CSV_INPUT = "todo_con_resultados_prob_final_venta_precontacto_future.csv"

OUT_EXCEL = "eda_fechas_registro_eval_correcta.xlsx"

TARGET_CONTACTO = "target_descuelgue_calc"
TARGET_VENTA_FINAL = "ganada"

SCORE_CONTACTO = "prob_descuelgue_modelo"
SCORE_VENTA_POST_CONTACTO = "prob_venta_modelo"
SCORE_FINAL_PRECONTACTO = "prob_final_venta_precontacto"

COL_DESC = "camp_total_descuelgues"


# ============================================================
# HELPERS
# ============================================================

def clean_numeric(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NAN": np.nan,
            "NaN": np.nan
        }),
        errors="coerce"
    )


def parse_date(s):
    return pd.to_datetime(
        s.astype(str)
        .str.strip()
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NAN": np.nan,
            "NaN": np.nan
        }),
        errors="coerce"
    )


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


def resumen_grupo(df, group_col, target_col, score_col):
    rows = []

    for grupo, g in df.groupby(group_col, dropna=False):
        g_score = g[[target_col, score_col]].dropna().copy()

        if len(g_score) == 0:
            rows.append({
                "grupo": grupo,
                "n": len(g),
                "n_con_score": 0,
                "positivos": np.nan,
                "tasa_real": np.nan,
                "score_medio": np.nan,
                "error_pred_real": np.nan,
                "ratio_pred_real": np.nan,
                "auc": np.nan,
                "auprc": np.nan
            })
            continue

        tasa_real = g_score[target_col].mean()
        score_medio = g_score[score_col].mean()

        rows.append({
            "grupo": grupo,
            "n": len(g),
            "n_con_score": len(g_score),
            "positivos": g_score[target_col].sum(),
            "tasa_real": tasa_real,
            "score_medio": score_medio,
            "error_pred_real": score_medio - tasa_real,
            "ratio_pred_real": (
                score_medio / tasa_real
                if tasa_real > 0 else np.nan
            ),
            "auc": safe_auc(g_score[target_col], g_score[score_col]),
            "auprc": safe_auprc(g_score[target_col], g_score[score_col])
        })

    return pd.DataFrame(rows)


def print_title(title):
    print("\n\n" + "=" * 90)
    print(title)
    print("=" * 90)


def interpretar(tabla, grupo_value, nombre):
    row = tabla[tabla["grupo"].astype(str) == grupo_value]

    if row.empty:
        print(f"\n{nombre}: no hay grupo {grupo_value}")
        return

    r = row.iloc[0]

    print(f"\n{nombre} - {grupo_value}")
    print(f"N: {r['n_con_score']:,.0f}")
    print(f"Positivos: {r['positivos']:,.0f}")
    print(f"Tasa real: {r['tasa_real']:.6f}")
    print(f"Score medio: {r['score_medio']:.6f}")
    print(f"Ratio pred/real: {r['ratio_pred_real']:.3f}")
    print(f"AUC: {r['auc']:.3f}")
    print(f"AUPRC: {r['auprc']:.6f}")

    if pd.isna(r["ratio_pred_real"]):
        print("Conclusión: no concluyente.")
    elif r["ratio_pred_real"] < 0.8:
        print("Conclusión: INFRAESTIMA.")
    elif r["ratio_pred_real"] > 1.2:
        print("Conclusión: SOBREESTIMA.")
    else:
        print("Conclusión: calibración razonable.")

    if not pd.isna(r["auc"]) and r["auc"] < 0.58:
        print("Además: ranking débil/casi aleatorio.")


# ============================================================
# 1. CARGAR
# ============================================================

print("Leyendo datos...")

df = pd.read_csv(
    CSV_INPUT,
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Filas:", len(df))
print("Columnas:", len(df.columns))


# ============================================================
# 2. VALIDACIONES
# ============================================================

required_cols = [
    COL_DESC,
    "fe_crea_reg",
    "fe_carga_efectiva",
    TARGET_VENTA_FINAL,
    SCORE_CONTACTO,
    SCORE_VENTA_POST_CONTACTO,
    SCORE_FINAL_PRECONTACTO
]

faltan = [c for c in required_cols if c not in df.columns]

if faltan:
    raise ValueError(f"Faltan columnas necesarias: {faltan}")


# ============================================================
# 3. TARGETS Y SCORES
# ============================================================

df[COL_DESC] = clean_numeric(df[COL_DESC])

df[TARGET_CONTACTO] = np.where(
    df[COL_DESC].fillna(0) > 0,
    1,
    0
)

df[TARGET_VENTA_FINAL] = (
    df[TARGET_VENTA_FINAL]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        "TRUE": 1,
        "FALSE": 0,
        "1": 1,
        "0": 0
    })
)

df[TARGET_VENTA_FINAL] = df[TARGET_VENTA_FINAL].fillna(0).astype(int)

for c in [
    SCORE_CONTACTO,
    SCORE_VENTA_POST_CONTACTO,
    SCORE_FINAL_PRECONTACTO
]:
    df[c] = clean_numeric(df[c])


# ============================================================
# 4. FECHAS Y BUCKETS
# ============================================================

df["fe_crea_reg_dt"] = parse_date(df["fe_crea_reg"])
df["fe_carga_efectiva_dt"] = parse_date(df["fe_carga_efectiva"])

df["dias_desde_crea_reg"] = (
    df["fe_carga_efectiva_dt"] - df["fe_crea_reg_dt"]
).dt.days

df.loc[
    df["dias_desde_crea_reg"] < 0,
    "dias_desde_crea_reg"
] = np.nan

df["bucket_dias_crea_reg"] = pd.cut(
    df["dias_desde_crea_reg"],
    bins=[-1, 7, 30, 90, 365, 730, np.inf],
    labels=[
        "00_0_7d",
        "01_8_30d",
        "02_31_90d",
        "03_91_365d",
        "04_1_2_anios",
        "05_mas_2_anios"
    ]
)

df["registro_nuevo_30d"] = np.where(
    df["dias_desde_crea_reg"].le(30),
    "REG_NUEVO_30D",
    "RESTO"
)

df["registro_nuevo_90d"] = np.where(
    df["dias_desde_crea_reg"].le(90),
    "REG_NUEVO_90D",
    "RESTO"
)


# ============================================================
# 5. DATASETS CORRECTOS DE EVALUACIÓN
# ============================================================

# Modelo de contacto:
# Evalúa P(descuelgue) sobre toda la población.
df_eval_contacto = df.copy()

# Modelo de venta post-contacto:
# Evalúa P(venta | contacto) SOLO en registros contactados.
df_eval_venta_post = df[
    df[TARGET_CONTACTO] == 1
].copy()

# Modelo final pre-contacto:
# Evalúa P(venta final) sobre toda la población.
df_eval_final = df.copy()


print_title("COBERTURA Y UNIVERSOS")

print("Filas totales:", len(df))
print("Filas con contacto real:", len(df_eval_venta_post))
print("Tasa contacto global:", df[TARGET_CONTACTO].mean())
print("Tasa venta final global:", df[TARGET_VENTA_FINAL].mean())

print("\nCobertura fe_crea_reg:", df["fe_crea_reg_dt"].notna().mean())
print("Cobertura fe_carga_efectiva:", df["fe_carga_efectiva_dt"].notna().mean())

print("\nDistribución bucket registro:")
print(
    df["bucket_dias_crea_reg"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nDistribución registro nuevo 30d:")
print(df["registro_nuevo_30d"].value_counts(dropna=False))

print("\nDistribución registro nuevo 90d:")
print(df["registro_nuevo_90d"].value_counts(dropna=False))


# ============================================================
# 6. RESÚMENES CORRECTOS
# ============================================================

res_contacto_bucket = resumen_grupo(
    df_eval_contacto,
    "bucket_dias_crea_reg",
    TARGET_CONTACTO,
    SCORE_CONTACTO
)

res_contacto_30 = resumen_grupo(
    df_eval_contacto,
    "registro_nuevo_30d",
    TARGET_CONTACTO,
    SCORE_CONTACTO
)

res_contacto_90 = resumen_grupo(
    df_eval_contacto,
    "registro_nuevo_90d",
    TARGET_CONTACTO,
    SCORE_CONTACTO
)


res_venta_post_bucket = resumen_grupo(
    df_eval_venta_post,
    "bucket_dias_crea_reg",
    TARGET_VENTA_FINAL,
    SCORE_VENTA_POST_CONTACTO
)

res_venta_post_30 = resumen_grupo(
    df_eval_venta_post,
    "registro_nuevo_30d",
    TARGET_VENTA_FINAL,
    SCORE_VENTA_POST_CONTACTO
)

res_venta_post_90 = resumen_grupo(
    df_eval_venta_post,
    "registro_nuevo_90d",
    TARGET_VENTA_FINAL,
    SCORE_VENTA_POST_CONTACTO
)


res_final_bucket = resumen_grupo(
    df_eval_final,
    "bucket_dias_crea_reg",
    TARGET_VENTA_FINAL,
    SCORE_FINAL_PRECONTACTO
)

res_final_30 = resumen_grupo(
    df_eval_final,
    "registro_nuevo_30d",
    TARGET_VENTA_FINAL,
    SCORE_FINAL_PRECONTACTO
)

res_final_90 = resumen_grupo(
    df_eval_final,
    "registro_nuevo_90d",
    TARGET_VENTA_FINAL,
    SCORE_FINAL_PRECONTACTO
)


# ============================================================
# 7. OUTPUT CONSOLA
# ============================================================

print_title("1. CONTACTO - TODA LA POBLACION")
print(res_contacto_bucket)

print_title("2. CONTACTO - REGISTRO NUEVO 30D")
print(res_contacto_30)

print_title("3. CONTACTO - REGISTRO NUEVO 90D")
print(res_contacto_90)


print_title("4. VENTA POST-CONTACTO - SOLO CONTACTADOS")
print(res_venta_post_bucket)

print_title("5. VENTA POST-CONTACTO - REGISTRO NUEVO 30D SOLO CONTACTADOS")
print(res_venta_post_30)

print_title("6. VENTA POST-CONTACTO - REGISTRO NUEVO 90D SOLO CONTACTADOS")
print(res_venta_post_90)


print_title("7. VENTA FINAL PRE-CONTACTO - TODA LA POBLACION")
print(res_final_bucket)

print_title("8. VENTA FINAL PRE-CONTACTO - REGISTRO NUEVO 30D")
print(res_final_30)

print_title("9. VENTA FINAL PRE-CONTACTO - REGISTRO NUEVO 90D")
print(res_final_90)


# ============================================================
# 8. INTERPRETACIÓN AUTOMÁTICA
# ============================================================

print_title("INTERPRETACIÓN AUTOMÁTICA")

interpretar(
    res_contacto_30,
    "REG_NUEVO_30D",
    "CONTACTO sobre toda la población"
)

interpretar(
    res_venta_post_30,
    "REG_NUEVO_30D",
    "VENTA POST-CONTACTO solo contactados"
)

interpretar(
    res_final_30,
    "REG_NUEVO_30D",
    "VENTA FINAL PRE-CONTACTO toda población"
)

print("\nResumen conceptual:")
print("- prob_descuelgue_modelo debe evaluarse contra target_descuelgue_calc en toda la población.")
print("- prob_venta_modelo debe evaluarse contra ganada SOLO entre contactados.")
print("- prob_final_venta_precontacto debe evaluarse contra ganada en toda la población.")


# ============================================================
# 9. EXPORTAR
# ============================================================

with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:

    res_contacto_bucket.to_excel(
        writer,
        sheet_name="contacto_bucket",
        index=False
    )

    res_contacto_30.to_excel(
        writer,
        sheet_name="contacto_30d",
        index=False
    )

    res_contacto_90.to_excel(
        writer,
        sheet_name="contacto_90d",
        index=False
    )

    res_venta_post_bucket.to_excel(
        writer,
        sheet_name="venta_post_bucket",
        index=False
    )

    res_venta_post_30.to_excel(
        writer,
        sheet_name="venta_post_30d",
        index=False
    )

    res_venta_post_90.to_excel(
        writer,
        sheet_name="venta_post_90d",
        index=False
    )

    res_final_bucket.to_excel(
        writer,
        sheet_name="final_bucket",
        index=False
    )

    res_final_30.to_excel(
        writer,
        sheet_name="final_30d",
        index=False
    )

    res_final_90.to_excel(
        writer,
        sheet_name="final_90d",
        index=False
    )

print(f"\nExcel generado: {OUT_EXCEL}")