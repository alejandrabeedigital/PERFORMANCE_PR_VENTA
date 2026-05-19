import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

path_camps = "seguimiento_camps.csv"

# Fichero local en la carpeta del proyecto
path_preds = "datos_test_prob_final_venta_precontacto_dailypred.csv"

# ESTE fichero corresponde al output del 29/04/2026
fecha_preds = pd.Timestamp("2026-04-29")

fecha_ini = pd.Timestamp("2026-04-14")
fecha_fin = pd.Timestamp("2026-05-01")

output_path = "analisis_outcome_sin_con_pred.xlsx"


# ============================================================
# HELPERS
# ============================================================

def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
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


def print_title(title):
    print("\n\n" + "=" * 90)
    print(title)
    print("=" * 90)


def resumen_grupo(data, group_cols, score_col):

    def calc(g):

        g_score = g[g[score_col].notna()].copy()

        if len(g_score) == 0:
            return pd.Series({
                "n": len(g),
                "n_con_score": 0,
                "ventas": g["target_venta"].sum(),
                "ventas_con_score": np.nan,
                "tasa_venta_real": g["target_venta"].mean(),
                "tasa_venta_real_con_score": np.nan,
                "prob_media_modelo": np.nan,
                "prob_mediana_modelo": np.nan,
                "score_p10": np.nan,
                "score_p50": np.nan,
                "score_p90": np.nan,
                "error_pred_real": np.nan,
                "ratio_pred_real": np.nan,
                "auc": np.nan,
                "auprc": np.nan,
            })

        tasa_real_score = g_score["target_venta"].mean()
        prob_media = g_score[score_col].mean()

        return pd.Series({
            "n": len(g),
            "n_con_score": len(g_score),
            "ventas": g["target_venta"].sum(),
            "ventas_con_score": g_score["target_venta"].sum(),
            "tasa_venta_real": g["target_venta"].mean(),
            "tasa_venta_real_con_score": tasa_real_score,
            "prob_media_modelo": prob_media,
            "prob_mediana_modelo": g_score[score_col].median(),
            "score_p10": g_score[score_col].quantile(0.10),
            "score_p50": g_score[score_col].quantile(0.50),
            "score_p90": g_score[score_col].quantile(0.90),
            "error_pred_real": prob_media - tasa_real_score,
            "ratio_pred_real": (
                prob_media / tasa_real_score
                if tasa_real_score > 0
                else np.nan
            ),
            "auc": safe_auc(
                g_score["target_venta"],
                g_score[score_col]
            ),
            "auprc": safe_auprc(
                g_score["target_venta"],
                g_score[score_col]
            ),
        })

    return (
        data
        .groupby(group_cols, dropna=False)
        .apply(calc)
        .reset_index()
    )


def deciles_por_grupo(data, grupo_col, score_col):

    rows = []

    for grupo, g in data.groupby(grupo_col, dropna=False):

        g = g[[score_col, "target_venta"]].dropna(subset=[score_col]).copy()

        if len(g) < 10:
            continue

        g["decil"] = pd.qcut(
            g[score_col].rank(method="first", ascending=False),
            q=10,
            labels=False
        ) + 1

        base_rate = g["target_venta"].mean()

        tmp = (
            g.groupby("decil")
            .agg(
                n=("target_venta", "size"),
                ventas=("target_venta", "sum"),
                tasa_venta=("target_venta", "mean"),
                score_medio=(score_col, "mean"),
                score_min=(score_col, "min"),
                score_max=(score_col, "max"),
            )
            .reset_index()
        )

        tmp["grupo"] = grupo

        tmp["lift_vs_media_grupo"] = (
            tmp["tasa_venta"] / base_rate
            if base_rate > 0
            else np.nan
        )

        tmp["ventas_acum"] = tmp["ventas"].cumsum()

        tmp["pct_ventas_acum"] = (
            tmp["ventas_acum"] / tmp["ventas"].sum()
            if tmp["ventas"].sum() > 0
            else np.nan
        )

        tmp["pct_poblacion_acum"] = (
            tmp["n"].cumsum() / tmp["n"].sum()
        )

        rows.append(tmp)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ============================================================
# 1. CARGAR CAMPAÑAS
# ============================================================

print_title("1. CARGANDO SEGUIMIENTO CAMPAÑAS")

camps = pd.read_csv(
    path_camps,
    sep=";",
    decimal=",",
    dtype=str
)

for col in [
    "fe_carga",
    "fe_carga_efectiva",
    "fe_datos"
]:
    if col in camps.columns:
        camps[col] = pd.to_datetime(camps[col], errors="coerce")

for col in [
    "ganada",
    "prob_venta",
    "pr_venta_2026",
    "pr_contact",
    "pr_final_venta"
]:
    if col in camps.columns:
        camps[col] = clean_numeric(camps[col])

camps["co_cliente"] = camps["co_cliente"].astype(str).str.strip()

camps_eval = camps[
    (camps["fe_carga"] >= fecha_ini) &
    (camps["fe_carga"] < fecha_fin)
].copy()

camps_eval["target_venta"] = (
    camps_eval["ganada"]
    .fillna(0)
    .astype(int)
)

print("Filas camps_eval:", len(camps_eval))

print("\nDistribución fe_datos:")
print(
    camps_eval["fe_datos"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 2. CARGAR PREDICCIONES
# ============================================================

print_title("2. CARGANDO PREDICCIONES")

preds = pd.read_csv(
    path_preds,
    sep=",",
    decimal=".",
    dtype=str
)

preds["fe_datos"] = fecha_preds

preds["co_cliente"] = (
    preds["co_cliente"]
    .astype(str)
    .str.strip()
)

pred_num_cols = [
    "prob_final_venta_precontacto",
    "prob_final_venta_precontacto_raw",
    "prob_venta_modelo",
    "prob_venta_modelo_raw",
    "prob_descuelgue_modelo",
    "prob_descuelgue_modelo_raw",
]

for col in pred_num_cols:
    if col in preds.columns:
        preds[col] = clean_numeric(preds[col])

preds["outcome_sin_con_pred"] = (
    preds["outcome_sin_con_pred"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({
        "NAN": np.nan,
        "NONE": np.nan,
        "": np.nan
    })
)

print("Filas preds:", len(preds))


# ============================================================
# 3. DEDUP
# ============================================================

print_title("3. DEDUP")

dup = preds.duplicated(["co_cliente", "fe_datos"]).sum()

print("Duplicados:", dup)

if dup > 0:
    preds = preds.drop_duplicates(
        ["co_cliente", "fe_datos"],
        keep="first"
    )


# ============================================================
# 4. MERGE
# ============================================================

print_title("4. MERGE")

cols_pred_keep = [
    "co_cliente",
    "fe_datos",
    "outcome_sin_con_pred",
    "aut_o_no",
    "prob_final_venta_precontacto",
    "prob_venta_modelo",
    "prob_descuelgue_modelo",
    "decil_prob_final",
    "decil_modelo",
    "sin_gmb",
    "con_web",
    "con_local",
    "ct_merclie",
    "ant_empresa",
    "algun_contacto",
    "origen_sc_o_no",
]

cols_pred_keep = [
    c for c in cols_pred_keep
    if c in preds.columns
]

preds_keep = preds[cols_pred_keep].copy()

df_eval = camps_eval.merge(
    preds_keep,
    on=["co_cliente", "fe_datos"],
    how="left"
)

print("Filas tras merge:", len(df_eval))

df_eval["tiene_outcome"] = (
    df_eval["outcome_sin_con_pred"]
    .notna()
)

df_eval["tiene_score_pred"] = (
    df_eval["prob_final_venta_precontacto"]
    .notna()
)

print(
    "Match outcome:",
    round(df_eval["tiene_outcome"].mean(), 4)
)

print(
    "Match score:",
    round(df_eval["tiene_score_pred"].mean(), 4)
)

print("\nMatch por fecha:")

match_fecha = (
    df_eval
    .groupby("fe_datos")
    .agg(
        n=("co_cliente", "size"),
        n_match=("tiene_outcome", "sum"),
        pct_match=("tiene_outcome", "mean"),
        ventas=("target_venta", "sum")
    )
    .reset_index()
)

print(match_fecha)


# ============================================================
# 5. SCORE
# ============================================================

print_title("5. SCORE")

score_candidates = [
    "prob_final_venta_precontacto",
    "pr_final_venta",
    "prob_venta",
    "pr_venta_2026",
]

score_col = None

for col in score_candidates:
    if col in df_eval.columns:
        if df_eval[col].notna().sum() > 0:
            score_col = col
            break

print("Score usado:", score_col)

print(
    "N con score:",
    df_eval[score_col].notna().sum()
)

print(
    "Pct con score:",
    round(df_eval[score_col].notna().mean(), 4)
)


# ============================================================
# 6. GRUPOS
# ============================================================

df_eval["grupo_outcome_sin_con"] = np.select(
    [
        df_eval["outcome_sin_con_pred"].eq("DESCONOCIDO"),
        df_eval["outcome_sin_con_pred"].notna(),
    ],
    [
        "DESCONOCIDO",
        "RESTO_CON_OUTCOME",
    ],
    default="SIN_MATCH_OUTCOME"
)


# ============================================================
# 7. ANALISIS PRINCIPAL
# ============================================================

print_title("6. RESUMEN PRINCIPAL")

res_outcome = resumen_grupo(
    df_eval,
    ["grupo_outcome_sin_con"],
    score_col
)

print(res_outcome)


print_title("7. CATEGORIAS COMPLETAS")

categorias = resumen_grupo(
    df_eval,
    ["outcome_sin_con_pred"],
    score_col
)

categorias = categorias.sort_values(
    "n",
    ascending=False
)

print(categorias)


print_title("8. TEMPORAL")

temporal = resumen_grupo(
    df_eval,
    ["fe_datos", "grupo_outcome_sin_con"],
    score_col
)

print(temporal)


print_title("9. PESO DESCONOCIDO")

peso = (
    df_eval
    .groupby("fe_datos")
    .agg(
        n_total=("target_venta", "size"),
        n_desconocido=(
            "grupo_outcome_sin_con",
            lambda x: (x == "DESCONOCIDO").sum()
        ),
        n_match=(
            "outcome_sin_con_pred",
            lambda x: x.notna().sum()
        ),
        ventas_total=("target_venta", "sum"),
    )
    .reset_index()
)

peso["pct_desconocido_total"] = (
    peso["n_desconocido"] /
    peso["n_total"]
)

peso["pct_desconocido_match"] = (
    peso["n_desconocido"] /
    peso["n_match"]
)

print(peso)


# ============================================================
# 8. DECILES
# ============================================================

print_title("10. DECILES POR GRUPO")

deciles = deciles_por_grupo(
    df_eval,
    "grupo_outcome_sin_con",
    score_col
)

print(deciles)


print_title("11. DISTRIBUCION EN DECILES GLOBALES")

df_rank = df_eval[
    [
        score_col,
        "target_venta",
        "grupo_outcome_sin_con"
    ]
].dropna(subset=[score_col]).copy()

df_rank["decil_global"] = pd.qcut(
    df_rank[score_col]
    .rank(method="first", ascending=False),
    q=10,
    labels=False
) + 1

dist_deciles = (
    df_rank
    .groupby(
        ["decil_global", "grupo_outcome_sin_con"]
    )
    .agg(
        n=("target_venta", "size"),
        ventas=("target_venta", "sum"),
        tasa_venta=("target_venta", "mean"),
        score_medio=(score_col, "mean"),
    )
    .reset_index()
)

dist_deciles["pct_dentro_decil"] = (
    dist_deciles["n"] /
    dist_deciles.groupby("decil_global")["n"]
    .transform("sum")
)

print(dist_deciles)


# ============================================================
# 9. CRUCES
# ============================================================

cruces = [
    "origen",
    "campaña",
    "ct_sociedad",
    "co_actvad",
    "cat_contact",
    "movil",
]

cruce_outputs = {}

for col in cruces:

    if col in df_eval.columns:

        print_title(f"12. CRUCE CON {col}")

        tmp = resumen_grupo(
            df_eval,
            [col, "grupo_outcome_sin_con"],
            score_col
        )

        tmp = (
            tmp[tmp["n"] >= 500]
            .sort_values([col, "grupo_outcome_sin_con"])
        )

        cruce_outputs[col] = tmp

        print(tmp)


# ============================================================
# 10. EXPORT
# ============================================================

print_title("13. EXPORT")

with pd.ExcelWriter(
    output_path,
    engine="xlsxwriter"
) as writer:

    match_fecha.to_excel(
        writer,
        sheet_name="match_fecha",
        index=False
    )

    res_outcome.to_excel(
        writer,
        sheet_name="resumen_principal",
        index=False
    )

    categorias.to_excel(
        writer,
        sheet_name="categorias",
        index=False
    )

    temporal.to_excel(
        writer,
        sheet_name="temporal",
        index=False
    )

    peso.to_excel(
        writer,
        sheet_name="peso",
        index=False
    )

    deciles.to_excel(
        writer,
        sheet_name="deciles",
        index=False
    )

    dist_deciles.to_excel(
        writer,
        sheet_name="deciles_globales",
        index=False
    )

    for col, tmp in cruce_outputs.items():

        sheet = f"cruce_{col}"[:31]

        tmp.to_excel(
            writer,
            sheet_name=sheet,
            index=False
        )

print(f"\nArchivo generado: {output_path}")