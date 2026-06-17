import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

CSV_INPUT = "todo_con_resultados_prob_final_venta_precontacto_future.csv"
OUT_EXCEL = "diagnostico_registros_nuevos_venta.xlsx"

TARGET_VENTA = "ganada"
TARGET_CONTACTO = "target_descuelgue_calc"

SCORE_CONTACTO = "prob_descuelgue_modelo"
SCORE_VENTA = "prob_venta_modelo"
SCORE_FINAL = "prob_final_venta_precontacto"


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

    if len(tmp) == 0:
        return np.nan

    if tmp["y"].nunique() < 2:
        return np.nan

    return roc_auc_score(tmp["y"], tmp["s"])


def safe_auprc(y, s):
    tmp = pd.DataFrame({"y": y, "s": s}).dropna()

    if len(tmp) == 0:
        return np.nan

    if tmp["y"].nunique() < 2:
        return np.nan

    return average_precision_score(tmp["y"], tmp["s"])


def resumen_grupo(df, group_col, target_col, score_col):
    rows = []

    for grupo, g in df.groupby(group_col, dropna=False):
        g_score = g[[target_col, score_col]].dropna().copy()

        if len(g_score) == 0:
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
            "ratio_pred_real": score_medio / tasa_real if tasa_real > 0 else np.nan,
            "auc": safe_auc(g_score[target_col], g_score[score_col]),
            "auprc": safe_auprc(g_score[target_col], g_score[score_col]),
        })

    return pd.DataFrame(rows)


def print_title(title):
    print("\n\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# 1. CARGA
# ============================================================

print("Leyendo datos...")

df = pd.read_csv(CSV_INPUT, low_memory=False)
df.columns = df.columns.str.strip()

print("Filas:", len(df))
print("Columnas:", len(df.columns))


# ============================================================
# 2. TARGETS / SCORES
# ============================================================

if TARGET_CONTACTO not in df.columns:
    if "camp_total_descuelgues" not in df.columns:
        raise ValueError("No existe target_descuelgue_calc ni camp_total_descuelgues.")
    df["camp_total_descuelgues"] = clean_numeric(df["camp_total_descuelgues"])
    df[TARGET_CONTACTO] = np.where(
        df["camp_total_descuelgues"].fillna(0) > 0,
        1,
        0
    )

required_cols = [
    TARGET_VENTA,
    TARGET_CONTACTO,
    SCORE_CONTACTO,
    SCORE_VENTA,
    SCORE_FINAL,
    "fe_crea_reg",
    "fe_carga_efectiva"
]

faltan = [c for c in required_cols if c not in df.columns]

if faltan:
    raise ValueError(f"Faltan columnas: {faltan}")

df[TARGET_VENTA] = clean_numeric(df[TARGET_VENTA]).fillna(0).astype(int)
df[TARGET_CONTACTO] = clean_numeric(df[TARGET_CONTACTO]).fillna(0).astype(int)

for c in [SCORE_CONTACTO, SCORE_VENTA, SCORE_FINAL]:
    df[c] = clean_numeric(df[c])


# ============================================================
# 3. VARIABLES FECHA REGISTRO
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
# 4. UNIVERSOS
# ============================================================

df_contactados = df[df[TARGET_CONTACTO] == 1].copy()
df_no_contactados = df[df[TARGET_CONTACTO] == 0].copy()

print_title("COBERTURA Y UNIVERSOS")

print("Filas totales:", len(df))
print("Filas contactadas:", len(df_contactados))
print("Filas no contactadas:", len(df_no_contactados))
print("Tasa contacto global:", df[TARGET_CONTACTO].mean())
print("Tasa venta final global:", df[TARGET_VENTA].mean())
print("Tasa venta post-contacto:", df_contactados[TARGET_VENTA].mean())

print("\nCobertura fe_crea_reg:", df["fe_crea_reg_dt"].notna().mean())
print("Cobertura fe_carga_efectiva:", df["fe_carga_efectiva_dt"].notna().mean())

print("\nDistribución bucket_dias_crea_reg:")
print(
    df["bucket_dias_crea_reg"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 5. TABLAS PRINCIPALES
# ============================================================

# Venta post-contacto: solo contactados
res_venta_post_bucket = resumen_grupo(
    df_contactados,
    "bucket_dias_crea_reg",
    TARGET_VENTA,
    SCORE_VENTA
)

res_venta_post_30 = resumen_grupo(
    df_contactados,
    "registro_nuevo_30d",
    TARGET_VENTA,
    SCORE_VENTA
)

res_venta_post_90 = resumen_grupo(
    df_contactados,
    "registro_nuevo_90d",
    TARGET_VENTA,
    SCORE_VENTA
)

# Venta final: toda población
res_final_bucket = resumen_grupo(
    df,
    "bucket_dias_crea_reg",
    TARGET_VENTA,
    SCORE_FINAL
)

res_final_30 = resumen_grupo(
    df,
    "registro_nuevo_30d",
    TARGET_VENTA,
    SCORE_FINAL
)

res_final_90 = resumen_grupo(
    df,
    "registro_nuevo_90d",
    TARGET_VENTA,
    SCORE_FINAL
)

# Contacto: toda población
res_contacto_bucket = resumen_grupo(
    df,
    "bucket_dias_crea_reg",
    TARGET_CONTACTO,
    SCORE_CONTACTO
)


# ============================================================
# 6. MATRIZ 2X2: REGISTRO NUEVO VS EMPRESA 2026
# ============================================================

if "fe_creacion_empresa" in df.columns:

    df["year_creacion_empresa"] = clean_numeric(df["fe_creacion_empresa"])

    df["empresa_2026"] = np.where(
        df["year_creacion_empresa"].eq(2026),
        "EMPRESA_2026",
        "RESTO_EMPRESAS"
    )

    df["combo_registro_empresa"] = (
        df["registro_nuevo_30d"].astype(str)
        + " | "
        + df["empresa_2026"].astype(str)
    )

    df_contactados["year_creacion_empresa"] = clean_numeric(
        df_contactados["fe_creacion_empresa"]
    )

    df_contactados["empresa_2026"] = np.where(
        df_contactados["year_creacion_empresa"].eq(2026),
        "EMPRESA_2026",
        "RESTO_EMPRESAS"
    )

    df_contactados["combo_registro_empresa"] = (
        df_contactados["registro_nuevo_30d"].astype(str)
        + " | "
        + df_contactados["empresa_2026"].astype(str)
    )

    res_combo_venta_post = resumen_grupo(
        df_contactados,
        "combo_registro_empresa",
        TARGET_VENTA,
        SCORE_VENTA
    )

    res_combo_final = resumen_grupo(
        df,
        "combo_registro_empresa",
        TARGET_VENTA,
        SCORE_FINAL
    )

else:
    res_combo_venta_post = pd.DataFrame()
    res_combo_final = pd.DataFrame()


# ============================================================
# 7. DISTRIBUCIÓN DE SCORE EN REGISTROS NUEVOS
# ============================================================

score_dist_rows = []

for grupo, g in df_contactados.groupby("registro_nuevo_30d"):
    s = g[SCORE_VENTA].dropna()

    score_dist_rows.append({
        "grupo": grupo,
        "n": len(g),
        "ventas": g[TARGET_VENTA].sum(),
        "tasa_real": g[TARGET_VENTA].mean(),
        "score_mean": s.mean(),
        "score_p01": s.quantile(0.01),
        "score_p05": s.quantile(0.05),
        "score_p10": s.quantile(0.10),
        "score_p25": s.quantile(0.25),
        "score_p50": s.quantile(0.50),
        "score_p75": s.quantile(0.75),
        "score_p90": s.quantile(0.90),
        "score_p95": s.quantile(0.95),
        "score_p99": s.quantile(0.99),
    })

score_dist = pd.DataFrame(score_dist_rows)


# ============================================================
# 8. PRINTS PARA PEGAR
# ============================================================

print_title("1. CONTACTO POR BUCKET DE ANTIGÜEDAD DEL REGISTRO")
print(res_contacto_bucket)

print_title("2. VENTA POST-CONTACTO POR BUCKET - SOLO CONTACTADOS")
print(res_venta_post_bucket)

print_title("3. VENTA POST-CONTACTO - REGISTRO NUEVO 30D")
print(res_venta_post_30)

print_title("4. VENTA POST-CONTACTO - REGISTRO NUEVO 90D")
print(res_venta_post_90)

print_title("5. VENTA FINAL PRECONTACTO POR BUCKET - TODA POBLACIÓN")
print(res_final_bucket)

print_title("6. VENTA FINAL PRECONTACTO - REGISTRO NUEVO 30D")
print(res_final_30)

print_title("7. VENTA FINAL PRECONTACTO - REGISTRO NUEVO 90D")
print(res_final_90)

if not res_combo_venta_post.empty:
    print_title("8. MATRIZ REGISTRO NUEVO 30D VS EMPRESA 2026 - VENTA POST-CONTACTO")
    print(res_combo_venta_post)

if not res_combo_final.empty:
    print_title("9. MATRIZ REGISTRO NUEVO 30D VS EMPRESA 2026 - VENTA FINAL")
    print(res_combo_final)

print_title("10. DISTRIBUCIÓN SCORE VENTA POST-CONTACTO")
print(score_dist)


# ============================================================
# 9. INTERPRETACIÓN AUTOMÁTICA
# ============================================================

print_title("11. INTERPRETACIÓN AUTOMÁTICA")

row = res_venta_post_30[
    res_venta_post_30["grupo"].astype(str).eq("REG_NUEVO_30D")
]

if not row.empty:
    r = row.iloc[0]

    print("\nVENTA POST-CONTACTO - REG_NUEVO_30D")
    print(f"N contactados: {r['n_con_score']:,.0f}")
    print(f"Ventas: {r['positivos']:,.0f}")
    print(f"Tasa real: {r['tasa_real']:.6f}")
    print(f"Score medio: {r['score_medio']:.6f}")
    print(f"Ratio pred/real: {r['ratio_pred_real']:.3f}")
    print(f"AUC: {r['auc']:.3f}")

    if r["ratio_pred_real"] < 0.8:
        print("Conclusión: el modelo de venta post-contacto sigue INFRAESTIMANDO registros nuevos.")
    elif r["ratio_pred_real"] > 1.2:
        print("Conclusión: el modelo de venta post-contacto SOBREESTIMA registros nuevos.")
    else:
        print("Conclusión: el modelo de venta post-contacto está razonablemente calibrado en registros nuevos.")

row = res_final_30[
    res_final_30["grupo"].astype(str).eq("REG_NUEVO_30D")
]

if not row.empty:
    r = row.iloc[0]

    print("\nVENTA FINAL PRECONTACTO - REG_NUEVO_30D")
    print(f"N total: {r['n_con_score']:,.0f}")
    print(f"Ventas: {r['positivos']:,.0f}")
    print(f"Tasa real: {r['tasa_real']:.6f}")
    print(f"Score medio: {r['score_medio']:.6f}")
    print(f"Ratio pred/real: {r['ratio_pred_real']:.3f}")
    print(f"AUC: {r['auc']:.3f}")

    if r["ratio_pred_real"] < 0.8:
        print("Conclusión: el score final sigue INFRAESTIMANDO registros nuevos.")
    elif r["ratio_pred_real"] > 1.2:
        print("Conclusión: el score final SOBREESTIMA registros nuevos.")
    else:
        print("Conclusión: el score final está razonablemente calibrado en registros nuevos.")


# ============================================================
# 10. EXPORTAR
# ============================================================

with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
    res_contacto_bucket.to_excel(writer, sheet_name="contacto_bucket", index=False)
    res_venta_post_bucket.to_excel(writer, sheet_name="venta_post_bucket", index=False)
    res_venta_post_30.to_excel(writer, sheet_name="venta_post_30d", index=False)
    res_venta_post_90.to_excel(writer, sheet_name="venta_post_90d", index=False)
    res_final_bucket.to_excel(writer, sheet_name="final_bucket", index=False)
    res_final_30.to_excel(writer, sheet_name="final_30d", index=False)
    res_final_90.to_excel(writer, sheet_name="final_90d", index=False)

    if not res_combo_venta_post.empty:
        res_combo_venta_post.to_excel(writer, sheet_name="combo_venta_post", index=False)

    if not res_combo_final.empty:
        res_combo_final.to_excel(writer, sheet_name="combo_final", index=False)

    score_dist.to_excel(writer, sheet_name="score_dist_venta_post", index=False)

print(f"\nExcel generado: {OUT_EXCEL}")