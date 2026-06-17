
import pandas as pd
import numpy as np

from common_pr_venta_8jun26 import add_common_features, clean_numeric

CSV_INPUT = "t_pr_venta_update_8jun26.csv"
OUT_EXCEL = "eda_dias_desde_crea_reg_fe_carga.xlsx"

TARGET_VENTA = "ganada"
TARGET_CONTACTO = "target_descuelgue_calc"


def resumen_grupo(df, group_col, target_col):
    rows = []
    for grupo, g in df.groupby(group_col, dropna=False):
        y = g[target_col].dropna()
        rows.append({
            "grupo": grupo,
            "n": len(g),
            "n_target_valido": len(y),
            "positivos": y.sum() if len(y) > 0 else np.nan,
            "tasa": y.mean() if len(y) > 0 else np.nan,
            "pct_poblacion": len(g) / len(df) if len(df) > 0 else np.nan
        })
    return pd.DataFrame(rows)


def print_title(title):
    print("\n\n" + "=" * 100)
    print(title)
    print("=" * 100)


print("Leyendo datos...")
df = pd.read_csv(CSV_INPUT, low_memory=False)
df.columns = df.columns.str.strip()
print("Filas:", len(df))
print("Columnas:", len(df.columns))

df = add_common_features(df)

if TARGET_VENTA not in df.columns:
    raise ValueError(f"No existe {TARGET_VENTA}")

df[TARGET_VENTA] = clean_numeric(df[TARGET_VENTA])
df[TARGET_CONTACTO] = clean_numeric(df[TARGET_CONTACTO]).fillna(0).astype(int)
df["ganada_valida"] = np.where(df[TARGET_VENTA].isin([0, 1]), df[TARGET_VENTA], np.nan)

df_contactados = df[df[TARGET_CONTACTO].eq(1)].copy()

print_title("1. COBERTURA FECHAS")
print("Fecha usada para dias_desde_crea_reg: fe_carga_efectiva")
print("\nCobertura fe_crea_reg:", df["fe_crea_reg_dt"].notna().mean())
print("Cobertura fe_carga_efectiva / fecha_ref:", df["fecha_ref_registro_dt"].notna().mean())
print("Cobertura fe_test:", df["fe_test_dt"].notna().mean())

print("\nRango fe_crea_reg:")
print(df["fe_crea_reg_dt"].min(), "->", df["fe_crea_reg_dt"].max())
print("\nRango fe_carga_efectiva / fecha_ref:")
print(df["fecha_ref_registro_dt"].min(), "->", df["fecha_ref_registro_dt"].max())
print("\nRango fe_test:")
print(df["fe_test_dt"].min(), "->", df["fe_test_dt"].max())

print("\nN dias_desde_crea_reg NA:", df["dias_desde_crea_reg"].isna().sum())
print("% dias_desde_crea_reg NA:", df["dias_desde_crea_reg"].isna().mean())

print("\nResumen numérico dias_desde_crea_reg:")
print(df["dias_desde_crea_reg"].describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]))

dist_dias = (
    df["dias_desde_crea_reg"]
    .value_counts(dropna=False)
    .rename_axis("dias_desde_crea_reg")
    .reset_index(name="n")
    .sort_values("dias_desde_crea_reg", na_position="first")
)
dist_dias["pct"] = dist_dias["n"] / len(df)

print_title("2. DISTRIBUCIÓN EXACTA DE dias_desde_crea_reg")
print("Primeros 100 valores:")
print(dist_dias.head(100))
print("\nÚltimos 50 valores:")
print(dist_dias.tail(50))
print("\nDías más frecuentes:")
print(dist_dias.sort_values("n", ascending=False).head(50))

print_title("3. DISTRIBUCIÓN BUCKETS PROVISIONALES")
print(df["bucket_dias_crea_reg"].value_counts(dropna=False).sort_index())

bucket_dist = df["bucket_dias_crea_reg"].value_counts(dropna=False).rename_axis("bucket").reset_index(name="n")
bucket_dist["pct"] = bucket_dist["n"] / len(df)

res_contacto_bucket = resumen_grupo(df, "bucket_dias_crea_reg", TARGET_CONTACTO)
res_venta_final_bucket = resumen_grupo(df, "bucket_dias_crea_reg", "ganada_valida")
res_venta_post_bucket = resumen_grupo(df_contactados, "bucket_dias_crea_reg", "ganada_valida")

print_title("4. CONTACTO POR BUCKET")
print(res_contacto_bucket)
print_title("5. VENTA FINAL POR BUCKET - TODA POBLACIÓN")
print(res_venta_final_bucket)
print_title("6. VENTA POST-CONTACTO POR BUCKET - SOLO CONTACTADOS")
print(res_venta_post_bucket)

labels_finos = [
    "00_mismo_dia", "01_1d", "02_2d", "03_3d", "04_4_7d",
    "05_8_14d", "06_15_30d", "07_31_60d", "08_61_90d",
    "09_91_120d", "10_121_180d", "11_181_365d",
    "12_366_730d", "13_mas_730d"
]
bins_finos = [-1, 0, 1, 2, 3, 7, 14, 30, 60, 90, 120, 180, 365, 730, np.inf]

df["bucket_fino_dias_crea_reg"] = pd.cut(df["dias_desde_crea_reg"], bins=bins_finos, labels=labels_finos)
df_contactados["bucket_fino_dias_crea_reg"] = pd.cut(df_contactados["dias_desde_crea_reg"], bins=bins_finos, labels=labels_finos)

res_contacto_fino = resumen_grupo(df, "bucket_fino_dias_crea_reg", TARGET_CONTACTO)
res_venta_final_fino = resumen_grupo(df, "bucket_fino_dias_crea_reg", "ganada_valida")
res_venta_post_fino = resumen_grupo(df_contactados, "bucket_fino_dias_crea_reg", "ganada_valida")

print_title("7. CONTACTO POR BUCKET FINO")
print(res_contacto_fino)
print_title("8. VENTA FINAL POR BUCKET FINO")
print(res_venta_final_fino)
print_title("9. VENTA POST-CONTACTO POR BUCKET FINO")
print(res_venta_post_fino)

df_valid_days = df[df["dias_desde_crea_reg"].notna()].copy()
df_valid_days["decil_dias_crea_reg"] = pd.qcut(df_valid_days["dias_desde_crea_reg"], 10, labels=False, duplicates="drop") + 1

df_contactados_valid_days = df_contactados[df_contactados["dias_desde_crea_reg"].notna()].copy()
df_contactados_valid_days["decil_dias_crea_reg"] = pd.qcut(df_contactados_valid_days["dias_desde_crea_reg"], 10, labels=False, duplicates="drop") + 1

res_venta_final_deciles_dias = resumen_grupo(df_valid_days, "decil_dias_crea_reg", "ganada_valida")
res_venta_post_deciles_dias = resumen_grupo(df_contactados_valid_days, "decil_dias_crea_reg", "ganada_valida")

rangos_deciles = (
    df_valid_days.groupby("decil_dias_crea_reg")["dias_desde_crea_reg"]
    .agg(dias_min="min", dias_max="max")
    .reset_index()
    .rename(columns={"decil_dias_crea_reg": "grupo"})
)
res_venta_final_deciles_dias = res_venta_final_deciles_dias.merge(rangos_deciles, on="grupo", how="left")

rangos_deciles_post = (
    df_contactados_valid_days.groupby("decil_dias_crea_reg")["dias_desde_crea_reg"]
    .agg(dias_min="min", dias_max="max")
    .reset_index()
    .rename(columns={"decil_dias_crea_reg": "grupo"})
)
res_venta_post_deciles_dias = res_venta_post_deciles_dias.merge(rangos_deciles_post, on="grupo", how="left")

print_title("10. VENTA FINAL POR DECILES DE DIAS")
print(res_venta_final_deciles_dias)
print_title("11. VENTA POST-CONTACTO POR DECILES DE DIAS")
print(res_venta_post_deciles_dias)

with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
    dist_dias.to_excel(writer, sheet_name="dist_exacta_dias", index=False)
    bucket_dist.to_excel(writer, sheet_name="dist_buckets", index=False)
    res_contacto_bucket.to_excel(writer, sheet_name="contacto_bucket", index=False)
    res_venta_final_bucket.to_excel(writer, sheet_name="venta_final_bucket", index=False)
    res_venta_post_bucket.to_excel(writer, sheet_name="venta_post_bucket", index=False)
    res_contacto_fino.to_excel(writer, sheet_name="contacto_bucket_fino", index=False)
    res_venta_final_fino.to_excel(writer, sheet_name="venta_final_bucket_fino", index=False)
    res_venta_post_fino.to_excel(writer, sheet_name="venta_post_bucket_fino", index=False)
    res_venta_final_deciles_dias.to_excel(writer, sheet_name="venta_final_deciles_dias", index=False)
    res_venta_post_deciles_dias.to_excel(writer, sheet_name="venta_post_deciles_dias", index=False)
    debug_cols = [
        "co_cliente", "id_opp", "fe_crea_reg", "fe_crea_reg_dt",
        "fe_carga_efectiva", "fecha_ref_registro_dt",
        "fe_test", "fe_test_dt", "dias_desde_crea_reg",
        "bucket_dias_crea_reg", "bucket_fino_dias_crea_reg",
        TARGET_CONTACTO, TARGET_VENTA
    ]
    debug_cols = [c for c in debug_cols if c in df.columns]
    df[debug_cols].head(20000).to_excel(writer, sheet_name="sample_debug", index=False)

print(f"\nExcel generado: {OUT_EXCEL}")
