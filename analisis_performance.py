import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score


# =========================
# 1. Cargar datos
# =========================

path = "seguimiento_camps.csv"

df = pd.read_csv(
    path,
    sep=";",
    decimal=",",
    dtype=str
)

# =========================
# 2. Parseo de fechas
# =========================

date_cols = [
    "fe_carga",
    "fe_carga_efectiva",
    "fe_datos",
    "primer_descuelgue_registrado",
    "ult_descuelgue_registrado",
    "camp_primer_descuelgue_registrado",
    "camp_ult_descuelgue_registrado"
]

for c in date_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")


# =========================
# 3. Convertir numéricas
# =========================

num_cols = [
    "prob_venta",
    "pr_venta_2026",
    "pr_contact",
    "pr_final_venta",
    "q_rk_score",
    "segm_score_ranking",
    "estimated_low_profit",
    "estimated_high_profit",
    "total_descuelgues",
    "camp_total_descuelgues",
    "n_contactado_ult6m",
    "n_sin_respuesta_ult6m",
    "intentos_ult6m",
    "dias_tras_carga",
    "ganada"
]

for c in num_cols:
    if c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")


# =========================
# 4. Filtro temporal pedido
# =========================

ini = pd.Timestamp("2026-05-01")
fin = pd.Timestamp("2026-07-14")

df_eval = df[
    (df["fe_carga"] >= ini) &
    (df["fe_carga"] < fin)
].copy()


# =========================
# 5. Elegir score a evaluar
# =========================

# Prioridad:
# 1. pr_final_venta si existe y tiene datos
# 2. pr_venta_2026 si existe
# 3. prob_venta si es la probabilidad cargada

score_candidates = ["pr_final_venta", "pr_venta_2026", "prob_venta"]

for col in score_candidates:
    if col in df_eval.columns and df_eval[col].notna().sum() > 0:
        score_col = col
        break

print("Score usado:", score_col)

df_eval = df_eval[df_eval[score_col].notna()].copy()


# =========================
# 6. Target real
# =========================

# Asumimos ganada = 1 venta, ganada = 0 no venta
df_eval["target_venta"] = df_eval["ganada"].fillna(0).astype(int)

# Opcional: si quieres excluir oportunidades aún abiertas
# df_eval = df_eval[df_eval["no_estado_oportunidad"].isin(["Ganada", "Perdida"])].copy()


# =========================
# 7. Métricas globales
# =========================

def metricas_globales(data, score_col, target_col="target_venta"):
    y = data[target_col]
    s = data[score_col]

    out = {
        "n": len(data),
        "ventas": int(y.sum()),
        "tasa_venta": y.mean(),
        "score_medio": s.mean(),
        "score_min": s.min(),
        "score_max": s.max()
    }

    if y.nunique() == 2:
        out["auc"] = roc_auc_score(y, s)
        out["auprc"] = average_precision_score(y, s)
    else:
        out["auc"] = np.nan
        out["auprc"] = np.nan

    return pd.Series(out)


global_perf = metricas_globales(df_eval, score_col)
print(global_perf)

# =========================
# 8. Performance por deciles
# =========================

def tabla_deciles(data, score_col, target_col="target_venta", n_deciles=10):
    d = data[[score_col, target_col]].dropna().copy()

    # Decil 1 = score más alto
    d["decil"] = pd.qcut(
        d[score_col].rank(method="first", ascending=False),
        q=n_deciles,
        labels=False
    ) + 1

    base_rate = d[target_col].mean()

    res = (
        d.groupby("decil")
        .agg(
            n=(target_col, "size"),
            ventas=(target_col, "sum"),
            tasa_venta=(target_col, "mean"),
            score_medio=(score_col, "mean"),
            score_min=(score_col, "min"),
            score_max=(score_col, "max")
        )
        .reset_index()
    )

    res["lift_vs_media"] = res["tasa_venta"] / base_rate
    res["ventas_acum"] = res["ventas"].cumsum()
    res["pct_ventas_acum"] = res["ventas_acum"] / res["ventas"].sum()
    res["pct_poblacion_acum"] = res["n"].cumsum() / res["n"].sum()

    return res

deciles = tabla_deciles(df_eval, score_col)
print(deciles)

# =========================
# 9. Calibración
# =========================

def tabla_calibracion(data, score_col, target_col="target_venta", n_bins=10):
    d = data[[score_col, target_col]].dropna().copy()

    d["bucket_score"] = pd.qcut(
        d[score_col].rank(method="first"),
        q=n_bins,
        labels=False
    ) + 1

    res = (
        d.groupby("bucket_score")
        .agg(
            n=(target_col, "size"),
            ventas=(target_col, "sum"),
            tasa_real=(target_col, "mean"),
            prob_predicha=(score_col, "mean")
        )
        .reset_index()
    )

    res["error_calibracion"] = res["prob_predicha"] - res["tasa_real"]
    res["ratio_pred_real"] = res["prob_predicha"] / res["tasa_real"].replace(0, np.nan)

    return res

calibracion = tabla_calibracion(df_eval, score_col)
print(calibracion)

# =========================
# 10. Performance por segmentos
# =========================

def performance_por_segmento(
    data,
    segment_col,
    score_col,
    target_col="target_venta",
    min_n=100
):
    rows = []

    for value, g in data.groupby(segment_col, dropna=False):
        if len(g) < min_n:
            continue

        y = g[target_col]
        s = g[score_col]

        row = {
            "variable": segment_col,
            "segmento": value,
            "n": len(g),
            "ventas": int(y.sum()),
            "tasa_venta": y.mean(),
            "score_medio": s.mean(),
            "error_pred_real": s.mean() - y.mean()
        }

        if y.nunique() == 2:
            row["auc"] = roc_auc_score(y, s)
            row["auprc"] = average_precision_score(y, s)
        else:
            row["auc"] = np.nan
            row["auprc"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["variable", "n"],
        ascending=[True, False]
    )


segment_vars = [
    "campaña",
    "plataforma_destino",
    "movil",
    "co_actvad",
    "cat_contact",
    "origen",
    "ct_sociedad",
    "co_prov",
    "segm_score_ranking"
]

segment_results = []

for var in segment_vars:
    if var in df_eval.columns:
        segment_results.append(
            performance_por_segmento(df_eval, var, score_col, min_n=100)
        )

segment_results = pd.concat(segment_results, ignore_index=True)

print(segment_results.head(50))

# =========================
# 11. Buckets de variables numéricas
# =========================

def performance_por_buckets_numericos(
    data,
    var,
    score_col,
    target_col="target_venta",
    n_bins=5,
    min_n=100
):
    d = data[[var, score_col, target_col]].dropna().copy()

    if d[var].nunique() < 2:
        return pd.DataFrame()

    d[f"{var}_bucket"] = pd.qcut(
        d[var].rank(method="first"),
        q=n_bins,
        duplicates="drop"
    )

    return performance_por_segmento(
        d,
        f"{var}_bucket",
        score_col,
        target_col,
        min_n=min_n
    )


numeric_segment_vars = [
    "q_rk_score",
    "segm_score_ranking",
    "estimated_low_profit",
    "estimated_high_profit",
    "n_contactado_ult6m",
    "n_sin_respuesta_ult6m",
    "intentos_ult6m",
    "total_descuelgues",
    "camp_total_descuelgues",
    "dias_tras_carga"
]

bucket_results = []

for var in numeric_segment_vars:
    if var in df_eval.columns:
        tmp = performance_por_buckets_numericos(df_eval, var, score_col)
        if len(tmp) > 0:
            bucket_results.append(tmp)

bucket_results = pd.concat(bucket_results, ignore_index=True)

print(bucket_results.head(50))

# =========================
# 12. Dónde sobreestima / infraestima
# =========================

segment_results["abs_error_pred_real"] = segment_results["error_pred_real"].abs()

peores_calibracion = (
    segment_results
    .sort_values("abs_error_pred_real", ascending=False)
    .head(30)
)

sobreestima = (
    segment_results
    .sort_values("error_pred_real", ascending=False)
    .head(30)
)

infraestima = (
    segment_results
    .sort_values("error_pred_real", ascending=True)
    .head(30)
)

print("Peor calibración")
print(peores_calibracion)

print("Sobreestima")
print(sobreestima)

print("Infraestima")
print(infraestima)

# =========================
# 13. Performance temporal
# =========================

# Si fe_test existe, usar fe_test. Si no, usar fe_datos.
fecha_modelo_col = "fe_test" if "fe_test" in df_eval.columns else "fe_datos"

temporal = (
    df_eval
    .groupby(fecha_modelo_col)
    .apply(lambda x: metricas_globales(x, score_col))
    .reset_index()
)

print(temporal)

# =========================
# 14. Exportar resultados
# =========================

output_path = "evaluacion_pr_venta_campanas.xlsx"

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    global_perf.to_frame("valor").to_excel(writer, sheet_name="global")
    deciles.to_excel(writer, sheet_name="deciles", index=False)
    calibracion.to_excel(writer, sheet_name="calibracion", index=False)
    segment_results.to_excel(writer, sheet_name="segmentos", index=False)
    bucket_results.to_excel(writer, sheet_name="buckets_numericos", index=False)
    peores_calibracion.to_excel(writer, sheet_name="peor_calibracion", index=False)
    sobreestima.to_excel(writer, sheet_name="sobreestima", index=False)
    infraestima.to_excel(writer, sheet_name="infraestima", index=False)
    temporal.to_excel(writer, sheet_name="temporal", index=False)

print(f"Archivo generado: {output_path}")