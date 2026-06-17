import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

CSV_INPUT = "t_pr_venta_update_8jun26.csv"

OUT_EXCEL = "antiguedad_registro_vs_venta.xlsx"
OUT_DIR = "graficos_antiguedad_registro"

TARGET_VENTA = "ganada"
COL_CONTACTO = "camp_total_descuelgues"

os.makedirs(OUT_DIR, exist_ok=True)


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
            "NaN": np.nan,
            "<NA>": np.nan
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
            "NaN": np.nan,
            "<NA>": np.nan
        }),
        errors="coerce"
    )


def resumen_por_tramo(df, group_col, target_col):
    out = (
        df.groupby(group_col, dropna=False)
        .agg(
            n=("dias_desde_crea_reg", "size"),
            positivos=(target_col, "sum"),
            tasa=(target_col, "mean"),
            dias_min=("dias_desde_crea_reg", "min"),
            dias_max=("dias_desde_crea_reg", "max"),
            dias_mean=("dias_desde_crea_reg", "mean")
        )
        .reset_index()
    )

    out["pct_poblacion"] = out["n"] / len(df)

    return out


def add_interval_bucket(df, step, max_day=730):
    """
    Crea tramos de step días hasta max_day y un bucket >max_day.
    """
    bins = list(range(0, max_day + step, step))

    if bins[0] != 0:
        bins = [0] + bins

    bins = [-1] + bins[1:] + [np.inf]

    labels = []

    prev = 0
    for upper in bins[1:-1]:
        labels.append(f"{int(prev)}_{int(upper)}d")
        prev = upper + 1

    labels.append(f"mas_{max_day}d")

    col = f"bucket_{step}d"
    df[col] = pd.cut(
        df["dias_desde_crea_reg"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return col


def plot_curve(tabla, y_col, title, filename, min_n=100):
    t = tabla.copy()
    t = t[t["n"] >= min_n].copy()
    t = t.sort_values("dias_mean")

    plt.figure(figsize=(11, 5))

    plt.plot(
        t["dias_mean"],
        t[y_col],
        marker="o"
    )

    plt.title(title)
    plt.xlabel("Días desde creación del registro")
    plt.ylabel(y_col)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Gráfico guardado: {path}")


def print_title(title):
    print("\n\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# 1. CARGA
# ============================================================

print("Leyendo datos...")

df = pd.read_csv(CSV_INPUT, low_memory=False)
df.columns = df.columns.str.strip()

print("Filas:", len(df))
print("Columnas:", len(df.columns))

required = [
    "fe_crea_reg",
    "fe_carga_efectiva",
    TARGET_VENTA,
    COL_CONTACTO
]

faltan = [c for c in required if c not in df.columns]

if faltan:
    raise ValueError(f"Faltan columnas: {faltan}")


# ============================================================
# 2. FECHAS Y TARGETS
# ============================================================

df["fe_crea_reg_dt"] = parse_date(df["fe_crea_reg"])
df["fe_carga_efectiva_dt"] = parse_date(df["fe_carga_efectiva"])

df["dias_desde_crea_reg"] = (
    df["fe_carga_efectiva_dt"] - df["fe_crea_reg_dt"]
).dt.days

df.loc[df["dias_desde_crea_reg"] < 0, "dias_desde_crea_reg"] = np.nan

df[TARGET_VENTA] = clean_numeric(df[TARGET_VENTA])
df["target_venta"] = np.where(
    df[TARGET_VENTA].isin([0, 1]),
    df[TARGET_VENTA],
    np.nan
)

df[COL_CONTACTO] = clean_numeric(df[COL_CONTACTO])

df["target_contacto"] = np.where(
    df[COL_CONTACTO].fillna(0) > 0,
    1,
    0
)

df_valid = df[df["dias_desde_crea_reg"].notna()].copy()
df_contactados = df_valid[df_valid["target_contacto"].eq(1)].copy()


# ============================================================
# 3. COBERTURA
# ============================================================

print_title("1. COBERTURA")

print("Cobertura fe_crea_reg:", df["fe_crea_reg_dt"].notna().mean())
print("Cobertura fe_carga_efectiva:", df["fe_carga_efectiva_dt"].notna().mean())
print("N dias_desde_crea_reg NA:", df["dias_desde_crea_reg"].isna().sum())
print("% dias_desde_crea_reg NA:", df["dias_desde_crea_reg"].isna().mean())

print("\nResumen dias_desde_crea_reg:")
print(
    df["dias_desde_crea_reg"]
    .describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
)


# ============================================================
# 4. BUCKETS 5/10/30 DÍAS
# ============================================================

bucket_5 = add_interval_bucket(df_valid, step=5, max_day=365)
bucket_10 = add_interval_bucket(df_valid, step=10, max_day=730)
bucket_30 = add_interval_bucket(df_valid, step=30, max_day=730)

bucket_5_c = add_interval_bucket(df_contactados, step=5, max_day=365)
bucket_10_c = add_interval_bucket(df_contactados, step=10, max_day=730)
bucket_30_c = add_interval_bucket(df_contactados, step=30, max_day=730)


# ============================================================
# 5. RESÚMENES
# ============================================================

venta_final_5 = resumen_por_tramo(df_valid, bucket_5, "target_venta")
venta_final_10 = resumen_por_tramo(df_valid, bucket_10, "target_venta")
venta_final_30 = resumen_por_tramo(df_valid, bucket_30, "target_venta")

contacto_10 = resumen_por_tramo(df_valid, bucket_10, "target_contacto")

venta_post_5 = resumen_por_tramo(df_contactados, bucket_5_c, "target_venta")
venta_post_10 = resumen_por_tramo(df_contactados, bucket_10_c, "target_venta")
venta_post_30 = resumen_por_tramo(df_contactados, bucket_30_c, "target_venta")


# ============================================================
# 6. PRINTS CLAVE
# ============================================================

print_title("2. VENTA FINAL - TRAMOS DE 30 DÍAS")
print(venta_final_30.head(40))

print_title("3. VENTA POST-CONTACTO - TRAMOS DE 30 DÍAS")
print(venta_post_30.head(40))

print_title("4. CONTACTO - TRAMOS DE 30 DÍAS")
print(contacto_10.head(80))

print_title("5. VENTA FINAL - TRAMOS DE 10 DÍAS, PRIMEROS 180 DÍAS")
print(venta_final_10[venta_final_10["dias_max"] <= 180])

print_title("6. VENTA POST-CONTACTO - TRAMOS DE 10 DÍAS, PRIMEROS 180 DÍAS")
print(venta_post_10[venta_post_10["dias_max"] <= 180])


# ============================================================
# 7. GRÁFICOS
# ============================================================

plot_curve(
    venta_final_10,
    "tasa",
    "Tasa de venta final por antigüedad del registro - tramos 10 días",
    "venta_final_tramos_10d.png",
    min_n=100
)

plot_curve(
    venta_post_10,
    "tasa",
    "Tasa de venta post-contacto por antigüedad del registro - tramos 10 días",
    "venta_postcontacto_tramos_10d.png",
    min_n=50
)

plot_curve(
    contacto_10,
    "tasa",
    "Tasa de contacto por antigüedad del registro - tramos 10 días",
    "contacto_tramos_10d.png",
    min_n=100
)


# ============================================================
# 8. PROPUESTA AUTOMÁTICA DE CORTES
# ============================================================

print_title("7. PROPUESTA ORIENTATIVA DE TRAMOS")

manual_bins = [-1, 7, 30, 60, 90, 120, 180, 365, 730, np.inf]

manual_labels = [
    "00_0_7d",
    "01_8_30d",
    "02_31_60d",
    "03_61_90d",
    "04_91_120d",
    "05_121_180d",
    "06_181_365d",
    "07_366_730d",
    "08_mas_730d"
]

df_valid["bucket_propuesto"] = pd.cut(
    df_valid["dias_desde_crea_reg"],
    bins=manual_bins,
    labels=manual_labels
)

df_contactados["bucket_propuesto"] = pd.cut(
    df_contactados["dias_desde_crea_reg"],
    bins=manual_bins,
    labels=manual_labels
)

res_final_prop = resumen_por_tramo(
    df_valid,
    "bucket_propuesto",
    "target_venta"
)

res_post_prop = resumen_por_tramo(
    df_contactados,
    "bucket_propuesto",
    "target_venta"
)

res_contacto_prop = resumen_por_tramo(
    df_valid,
    "bucket_propuesto",
    "target_contacto"
)

print("\nVenta final - bucket propuesto:")
print(res_final_prop)

print("\nVenta post-contacto - bucket propuesto:")
print(res_post_prop)

print("\nContacto - bucket propuesto:")
print(res_contacto_prop)


# ============================================================
# 9. EXPORTAR
# ============================================================

with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
    venta_final_5.to_excel(writer, sheet_name="venta_final_5d", index=False)
    venta_final_10.to_excel(writer, sheet_name="venta_final_10d", index=False)
    venta_final_30.to_excel(writer, sheet_name="venta_final_30d", index=False)

    venta_post_5.to_excel(writer, sheet_name="venta_post_5d", index=False)
    venta_post_10.to_excel(writer, sheet_name="venta_post_10d", index=False)
    venta_post_30.to_excel(writer, sheet_name="venta_post_30d", index=False)

    contacto_10.to_excel(writer, sheet_name="contacto_10d", index=False)

    res_final_prop.to_excel(writer, sheet_name="bucket_prop_final", index=False)
    res_post_prop.to_excel(writer, sheet_name="bucket_prop_post", index=False)
    res_contacto_prop.to_excel(writer, sheet_name="bucket_prop_contacto", index=False)

print(f"\nExcel generado: {OUT_EXCEL}")