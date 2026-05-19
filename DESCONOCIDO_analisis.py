import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# ANALISIS ESPECIFICO: outcome_sin_con_pred = DESCONOCIDO
# ============================================================

var = "outcome_sin_con_pred"

if var not in df_eval.columns:
    raise ValueError(f"No existe la columna {var} en df_eval")

df_out = df_eval.copy()

df_out[var] = (
    df_out[var]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({
        "NAN": np.nan,
        "NONE": np.nan,
        "": np.nan
    })
)

df_out["grupo_outcome_sin_con"] = np.where(
    df_out[var].eq("DESCONOCIDO"),
    "DESCONOCIDO",
    "RESTO"
)

fecha_col = "fe_test" if "fe_test" in df_out.columns else "fe_datos"


def safe_auc(y, s):
    if y.nunique() < 2:
        return np.nan
    return roc_auc_score(y, s)


def safe_auprc(y, s):
    if y.nunique() < 2:
        return np.nan
    return average_precision_score(y, s)


def resumen_grupo(data, group_cols):
    res = (
        data
        .groupby(group_cols, dropna=False)
        .apply(lambda g: pd.Series({
            "n": len(g),
            "ventas": g["target_venta"].sum(),
            "tasa_venta_real": g["target_venta"].mean(),
            "prob_media_modelo": g[score_col].mean(),
            "prob_mediana_modelo": g[score_col].median(),
            "score_p10": g[score_col].quantile(0.10),
            "score_p50": g[score_col].quantile(0.50),
            "score_p90": g[score_col].quantile(0.90),
            "error_pred_real": g[score_col].mean() - g["target_venta"].mean(),
            "ratio_pred_real": (
                g[score_col].mean() / g["target_venta"].mean()
                if g["target_venta"].mean() > 0
                else np.nan
            ),
            "auc": safe_auc(g["target_venta"], g[score_col]),
            "auprc": safe_auprc(g["target_venta"], g[score_col])
        }))
        .reset_index()
    )

    return res


# ============================================================
# 1. Resumen general DESCONOCIDO vs RESTO
# ============================================================

print("\n\n====================================================")
print("1. DESCONOCIDO VS RESTO")
print("====================================================")

res_outcome = resumen_grupo(df_out, ["grupo_outcome_sin_con"])
print(res_outcome)


# ============================================================
# 2. Distribución de categorías originales
# ============================================================

print("\n\n====================================================")
print("2. DISTRIBUCION COMPLETA DE outcome_sin_con_pred")
print("====================================================")

dist_outcome = resumen_grupo(df_out, [var]).sort_values("n", ascending=False)
print(dist_outcome)


# ============================================================
# 3. Evolución temporal DESCONOCIDO vs RESTO
# ============================================================

print("\n\n====================================================")
print("3. EVOLUCION TEMPORAL DESCONOCIDO VS RESTO")
print("====================================================")

temporal_outcome = resumen_grupo(
    df_out,
    [fecha_col, "grupo_outcome_sin_con"]
).sort_values([fecha_col, "grupo_outcome_sin_con"])

print(temporal_outcome)


# ============================================================
# 4. Peso de DESCONOCIDO por fecha
# ============================================================

print("\n\n====================================================")
print("4. PESO DE DESCONOCIDO POR FECHA")
print("====================================================")

peso_desconocido = (
    df_out
    .groupby(fecha_col)
    .agg(
        n_total=("target_venta", "size"),
        n_desconocido=("grupo_outcome_sin_con", lambda x: (x == "DESCONOCIDO").sum()),
        ventas_total=("target_venta", "sum"),
        ventas_desconocido=("target_venta", lambda x: df_out.loc[x.index, "target_venta"][df_out.loc[x.index, "grupo_outcome_sin_con"] == "DESCONOCIDO"].sum())
    )
    .reset_index()
)

peso_desconocido["pct_desconocido"] = (
    peso_desconocido["n_desconocido"] / peso_desconocido["n_total"]
)

peso_desconocido["pct_ventas_desconocido"] = (
    peso_desconocido["ventas_desconocido"] / peso_desconocido["ventas_total"].replace(0, np.nan)
)

print(peso_desconocido)


# ============================================================
# 5. Deciles dentro de DESCONOCIDO y RESTO
# ============================================================

print("\n\n====================================================")
print("5. DECILES DENTRO DE DESCONOCIDO Y RESTO")
print("====================================================")


def deciles_por_grupo(data, grupo_col, score_col, target_col="target_venta"):
    rows = []

    for grupo, g in data.groupby(grupo_col):
        g = g[[score_col, target_col]].dropna().copy()

        if len(g) < 10:
            continue

        g["decil"] = pd.qcut(
            g[score_col].rank(method="first", ascending=False),
            q=10,
            labels=False
        ) + 1

        base_rate = g[target_col].mean()

        tmp = (
            g.groupby("decil")
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

        tmp["grupo"] = grupo
        tmp["lift_vs_media_grupo"] = tmp["tasa_venta"] / base_rate if base_rate > 0 else np.nan
        tmp["ventas_acum"] = tmp["ventas"].cumsum()
        tmp["pct_ventas_acum"] = tmp["ventas_acum"] / tmp["ventas"].sum() if tmp["ventas"].sum() > 0 else np.nan
        tmp["pct_poblacion_acum"] = tmp["n"].cumsum() / tmp["n"].sum()

        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


deciles_outcome = deciles_por_grupo(
    df_out,
    "grupo_outcome_sin_con",
    score_col
)

print(deciles_outcome)


# ============================================================
# 6. Ranking global: dónde caen los DESCONOCIDO
# ============================================================

print("\n\n====================================================")
print("6. DISTRIBUCION DE DESCONOCIDO EN DECILES GLOBALES")
print("====================================================")

df_rank = df_out[[score_col, "target_venta", "grupo_outcome_sin_con"]].dropna().copy()

df_rank["decil_global"] = pd.qcut(
    df_rank[score_col].rank(method="first", ascending=False),
    q=10,
    labels=False
) + 1

dist_deciles_global = (
    df_rank
    .groupby(["decil_global", "grupo_outcome_sin_con"])
    .agg(
        n=("target_venta", "size"),
        ventas=("target_venta", "sum"),
        tasa_venta=("target_venta", "mean"),
        score_medio=(score_col, "mean")
    )
    .reset_index()
)

dist_deciles_global["pct_dentro_decil"] = (
    dist_deciles_global["n"] /
    dist_deciles_global.groupby("decil_global")["n"].transform("sum")
)

print(dist_deciles_global)


# ============================================================
# 7. Cruce con origen
# ============================================================

if "origen" in df_out.columns:
    print("\n\n====================================================")
    print("7. DESCONOCIDO VS RESTO POR ORIGEN")
    print("====================================================")

    origen_outcome = resumen_grupo(
        df_out,
        ["origen", "grupo_outcome_sin_con"]
    ).sort_values(["origen", "grupo_outcome_sin_con"])

    print(origen_outcome)


# ============================================================
# 8. Cruce con campaña
# ============================================================

if "campaña" in df_out.columns:
    print("\n\n====================================================")
    print("8. DESCONOCIDO VS RESTO POR CAMPAÑA")
    print("====================================================")

    camp_outcome = resumen_grupo(
        df_out,
        ["campaña", "grupo_outcome_sin_con"]
    ).sort_values(["campaña", "grupo_outcome_sin_con"])

    print(camp_outcome)


# ============================================================
# 9. Cruce con ct_sociedad
# ============================================================

if "ct_sociedad" in df_out.columns:
    print("\n\n====================================================")
    print("9. DESCONOCIDO VS RESTO POR CT_SOCIEDAD")
    print("====================================================")

    ct_outcome = resumen_grupo(
        df_out,
        ["ct_sociedad", "grupo_outcome_sin_con"]
    ).sort_values(["ct_sociedad", "grupo_outcome_sin_con"])

    print(ct_outcome)


# ============================================================
# 10. Export opcional
# ============================================================

output_outcome = "analisis_outcome_sin_con_pred.xlsx"

with pd.ExcelWriter(output_outcome, engine="xlsxwriter") as writer:
    res_outcome.to_excel(writer, sheet_name="desconocido_vs_resto", index=False)
    dist_outcome.to_excel(writer, sheet_name="categorias", index=False)
    temporal_outcome.to_excel(writer, sheet_name="temporal", index=False)
    peso_desconocido.to_excel(writer, sheet_name="peso_temporal", index=False)
    deciles_outcome.to_excel(writer, sheet_name="deciles_por_grupo", index=False)
    dist_deciles_global.to_excel(writer, sheet_name="decil_global", index=False)

    if "origen" in df_out.columns:
        origen_outcome.to_excel(writer, sheet_name="origen", index=False)

    if "campaña" in df_out.columns:
        camp_outcome.to_excel(writer, sheet_name="campaña", index=False)

    if "ct_sociedad" in df_out.columns:
        ct_outcome.to_excel(writer, sheet_name="ct_sociedad", index=False)

print(f"\nArchivo generado: {output_outcome}")