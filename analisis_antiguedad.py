import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# ============================================================
# CONFIG
# ============================================================

CSV_INPUT = "seguimiento_camps.csv"

OUT_EXCEL = "analisis_empresas_recientes_2026.xlsx"

TARGET_COL = "ganada"

SCORE_COLS = [
    "pr_venta_2026",
    "pr_contact",
    "pr_final_venta"
]

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
            "NAN": np.nan
        }),
        errors="coerce"
    )


def safe_auc(y, s):

    tmp = pd.DataFrame({
        "y": y,
        "s": s
    }).dropna()

    if len(tmp) == 0:
        return np.nan

    if tmp["y"].nunique() < 2:
        return np.nan

    return roc_auc_score(tmp["y"], tmp["s"])


def safe_auprc(y, s):

    tmp = pd.DataFrame({
        "y": y,
        "s": s
    }).dropna()

    if len(tmp) == 0:
        return np.nan

    if tmp["y"].nunique() < 2:
        return np.nan

    return average_precision_score(tmp["y"], tmp["s"])


def resumen_por_grupo(df, grupo_col, score_col):

    rows = []

    for grupo, g in df.groupby(grupo_col, dropna=False):

        g_score = g[[TARGET_COL, score_col]].dropna().copy()

        if len(g_score) == 0:
            continue

        tasa_real = g_score[TARGET_COL].mean()
        score_medio = g_score[score_col].mean()

        rows.append({
            "grupo": grupo,
            "score": score_col,
            "n": len(g),
            "ventas": g_score[TARGET_COL].sum(),
            "tasa_real": tasa_real,
            "score_medio": score_medio,
            "error_pred_real": score_medio - tasa_real,
            "ratio_pred_real": (
                score_medio / tasa_real
                if tasa_real > 0 else np.nan
            ),
            "auc": safe_auc(
                g_score[TARGET_COL],
                g_score[score_col]
            ),
            "auprc": safe_auprc(
                g_score[TARGET_COL],
                g_score[score_col]
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# 1. LEER CSV
# ============================================================

print("Leyendo datos...")

df = pd.read_csv(
    CSV_INPUT,
    sep=";",
    low_memory=False
)

df.columns = df.columns.str.strip()

print("\nColumnas:")
print(df.columns.tolist())


# ============================================================
# 2. FILTRO FECHAS CAMPAÑA
# ============================================================

df["fe_carga_dt"] = pd.to_datetime(
    df["fe_carga"],
    errors="coerce"
)

df = df[
    (df["fe_carga_dt"] >= pd.Timestamp("2026-04-14")) &
    (df["fe_carga_dt"] < pd.Timestamp("2026-05-01"))
].copy()

print("\nFilas tras filtro fechas:", len(df))


# ============================================================
# 3. LIMPIAR NUMÉRICAS
# ============================================================

for col in [TARGET_COL] + SCORE_COLS:

    if col in df.columns:

        df[col] = clean_numeric(df[col])

df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)


# ============================================================
# 4. ARREGLAR FECHA CREACIÓN EMPRESA
# ============================================================

print("\nEjemplos fe_creacion_empresa RAW:")
print(df["fe_creacion_empresa"].head(20))

# IMPORTANTE:
# La columna YA contiene el año:
# 2025.0
# 2026.0
# NO usar to_datetime()

df["year_creacion_empresa"] = pd.to_numeric(
    df["fe_creacion_empresa"],
    errors="coerce"
)

CURRENT_YEAR = pd.Timestamp.now().year

df.loc[
    (
        (df["year_creacion_empresa"] < 1900) |
        (df["year_creacion_empresa"] > CURRENT_YEAR)
    ),
    "year_creacion_empresa"
] = np.nan

print("\nDistribución años creación empresa:")
print(
    df["year_creacion_empresa"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 5. CREAR COHORTES
# ============================================================

df["grupo_creacion_empresa"] = np.select(
    [
        df["year_creacion_empresa"].eq(2026),
        df["year_creacion_empresa"].eq(2025),
        df["year_creacion_empresa"].between(2023, 2024),
        df["year_creacion_empresa"].between(2020, 2022),
        df["year_creacion_empresa"].lt(2020),
        df["year_creacion_empresa"].isna(),
    ],
    [
        "2026",
        "2025",
        "2023_2024",
        "2020_2022",
        "antes_2020",
        "sin_fecha"
    ],
    default="otros"
)

df["empresa_2026_vs_resto"] = np.where(
    df["year_creacion_empresa"].eq(2026),
    "EMPRESA_2026",
    "RESTO"
)

print("\nDistribución cohortes:")
print(
    df["grupo_creacion_empresa"]
    .value_counts(dropna=False)
)

print("\nVentas por cohorte:")
print(
    df.groupby("grupo_creacion_empresa")[TARGET_COL]
    .agg(["count", "sum", "mean"])
)


# ============================================================
# 6. VALIDAR SCORES
# ============================================================

score_cols_presentes = [
    c for c in SCORE_COLS
    if c in df.columns and df[c].notna().sum() > 0
]

print("\nScores encontrados:")
print(score_cols_presentes)

if not score_cols_presentes:
    raise ValueError("No hay scores válidos.")


# ============================================================
# 7. RESÚMENES
# ============================================================

resumenes = []
resumenes_2026 = []

for score_col in score_cols_presentes:

    print(f"\nAnalizando: {score_col}")

    res = resumen_por_grupo(
        df,
        "grupo_creacion_empresa",
        score_col
    )

    resumenes.append(res)

    res_2026 = resumen_por_grupo(
        df,
        "empresa_2026_vs_resto",
        score_col
    )

    resumenes_2026.append(res_2026)


resumen = pd.concat(
    resumenes,
    ignore_index=True
)

resumen_2026_vs_resto = pd.concat(
    resumenes_2026,
    ignore_index=True
)


# ============================================================
# 8. OUTPUTS PRINCIPALES
# ============================================================

print("\n\n====================================================")
print("1. RESUMEN POR COHORTE")
print("====================================================")

print(
    resumen[
        [
            "score",
            "grupo",
            "n",
            "ventas",
            "tasa_real",
            "score_medio",
            "error_pred_real",
            "ratio_pred_real",
            "auc",
            "auprc"
        ]
    ]
    .sort_values(["score", "grupo"])
)


print("\n\n====================================================")
print("2. EMPRESAS 2026 VS RESTO")
print("====================================================")

print(
    resumen_2026_vs_resto[
        [
            "score",
            "grupo",
            "n",
            "ventas",
            "tasa_real",
            "score_medio",
            "error_pred_real",
            "ratio_pred_real",
            "auc",
            "auprc"
        ]
    ]
    .sort_values(["score", "grupo"])
)


# ============================================================
# 9. INTERPRETACIÓN AUTOMÁTICA
# ============================================================

print("\n\n====================================================")
print("3. INTERPRETACIÓN AUTOMÁTICA")
print("====================================================")

tmp_2026 = resumen_2026_vs_resto[
    resumen_2026_vs_resto["grupo"] == "EMPRESA_2026"
]

if len(tmp_2026) == 0:

    print("\nNo hay empresas creadas en 2026.")

else:

    for _, row in tmp_2026.iterrows():

        score = row["score"]
        ratio = row["ratio_pred_real"]
        auc = row["auc"]

        print(f"\nScore: {score}")

        print(f"- Ratio pred/real: {ratio:.3f}")
        print(f"- AUC: {auc:.3f}")

        if pd.isna(ratio):

            print("No hay datos suficientes.")

        elif ratio < 0.8:

            print(
                "➡️ El modelo INFRAESTIMA empresas creadas en 2026."
            )

        elif ratio > 1.2:

            print(
                "➡️ El modelo SOBREESTIMA empresas creadas en 2026."
            )

        else:

            print(
                "➡️ El modelo está razonablemente calibrado."
            )


# ============================================================
# 10. EXPORTAR EXCEL
# ============================================================

with pd.ExcelWriter(
    OUT_EXCEL,
    engine="xlsxwriter"
) as writer:

    resumen.to_excel(
        writer,
        sheet_name="por_cohorte",
        index=False
    )

    resumen_2026_vs_resto.to_excel(
        writer,
        sheet_name="2026_vs_resto",
        index=False
    )

print(f"\nExcel generado: {OUT_EXCEL}")