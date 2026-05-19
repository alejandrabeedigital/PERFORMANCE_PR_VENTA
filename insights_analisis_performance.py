import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

path = "evaluacion_pr_venta_campanas.xlsx"

# =========================
# Cargar hojas
# =========================

xls = pd.ExcelFile(path)
print("HOJAS DEL EXCEL:")
print(xls.sheet_names)

global_perf = pd.read_excel(path, sheet_name="global")
deciles = pd.read_excel(path, sheet_name="deciles")
calibracion = pd.read_excel(path, sheet_name="calibracion")
segmentos = pd.read_excel(path, sheet_name="segmentos")
buckets = pd.read_excel(path, sheet_name="buckets_numericos")
temporal = pd.read_excel(path, sheet_name="temporal")

# =========================
# 1. Performance global
# =========================

print("\n\n==============================")
print("1. PERFORMANCE GLOBAL")
print("==============================")
print(global_perf)

# =========================
# 2. Deciles completos
# =========================

print("\n\n==============================")
print("2. DECILES COMPLETOS")
print("==============================")
cols_deciles = [
    "decil", "n", "ventas", "tasa_venta", "score_medio",
    "score_min", "score_max", "lift_vs_media",
    "ventas_acum", "pct_ventas_acum", "pct_poblacion_acum"
]
print(deciles[[c for c in cols_deciles if c in deciles.columns]])

# =========================
# 3. Calibración completa
# =========================

print("\n\n==============================")
print("3. CALIBRACION")
print("==============================")
print(calibracion)

# =========================
# 4. Temporal
# =========================

print("\n\n==============================")
print("4. PERFORMANCE TEMPORAL")
print("==============================")
print(temporal)

# =========================
# 5. Segmentos principales
# =========================

def print_segmento(var, min_n=500):
    tmp = segmentos[
        (segmentos["variable"] == var) &
        (segmentos["n"] >= min_n)
    ].copy()

    if tmp.empty:
        print(f"\nNo hay datos suficientes para {var} con min_n={min_n}")
        return

    print("\n\n==============================")
    print(f"SEGMENTO: {var} | MAYOR VOLUMEN")
    print("==============================")
    print(
        tmp.sort_values("n", ascending=False)
        .head(30)
    )

    print("\n\n==============================")
    print(f"SEGMENTO: {var} | MEJOR AUC")
    print("==============================")
    print(
        tmp.dropna(subset=["auc"])
        .sort_values("auc", ascending=False)
        .head(30)
    )

    print("\n\n==============================")
    print(f"SEGMENTO: {var} | PEOR AUC")
    print("==============================")
    print(
        tmp.dropna(subset=["auc"])
        .sort_values("auc", ascending=True)
        .head(30)
    )

    print("\n\n==============================")
    print(f"SEGMENTO: {var} | MAYOR SOBREESTIMACION")
    print("==============================")
    print(
        tmp.sort_values("error_pred_real", ascending=False)
        .head(30)
    )

    print("\n\n==============================")
    print(f"SEGMENTO: {var} | MAYOR INFRAESTIMACION")
    print("==============================")
    print(
        tmp.sort_values("error_pred_real", ascending=True)
        .head(30)
    )


for var in [
    "co_actvad",
    "cat_contact",
    "origen",
    "movil",
    "campaña",
    "plataforma_destino",
    "ct_sociedad",
    "co_prov",
    "segm_score_ranking"
]:
    print_segmento(var, min_n=500)

# =========================
# 6. Peores calibraciones globales
# =========================

print("\n\n==============================")
print("6. PEORES CALIBRACIONES CON VOLUMEN")
print("==============================")

seg = segmentos.copy()
seg = seg[seg["n"] >= 500].copy()
seg["abs_error_pred_real"] = seg["error_pred_real"].abs()

print(
    seg.sort_values("abs_error_pred_real", ascending=False)
    .head(50)
)

# =========================
# 7. Buckets numéricos
# =========================

print("\n\n==============================")
print("7. BUCKETS NUMERICOS")
print("==============================")
print(buckets)

# =========================
# 8. Resumen ultra útil por variable
# =========================

print("\n\n==============================")
print("8. RESUMEN POR VARIABLE")
print("==============================")

resumen_vars = (
    segmentos[segmentos["n"] >= 500]
    .groupby("variable")
    .agg(
        n_segmentos=("segmento", "nunique"),
        n_total=("n", "sum"),
        auc_medio=("auc", "mean"),
        auc_min=("auc", "min"),
        auc_max=("auc", "max"),
        error_abs_medio=("error_pred_real", lambda x: x.abs().mean()),
        tasa_venta_media=("tasa_venta", "mean"),
        score_medio=("score_medio", "mean")
    )
    .reset_index()
    .sort_values("error_abs_medio", ascending=False)
)

print(resumen_vars)

print("\n\nFIN DEL OUTPUT")