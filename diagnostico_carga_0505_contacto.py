import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score

from common_pr_venta_8jun26 import add_common_features, clean_numeric, parse_date


# ============================================================
# CONFIG
# ============================================================

CSV_INPUT = "todo_con_prob_descuelgue.csv"
OUT_EXCEL = "diagnostico_carga_0505_contacto.xlsx"
OUT_DIR = "graficos_diagnostico_carga_0505"

TARGET = "target_descuelgue_calc"
SCORE = "prob_descuelgue_modelo"

COL_FECHA = "fe_carga_efectiva"

FECHA_TEST = "2026-05-05"

VARIABLES_ANALISIS = [
    "cat_contact",
    "algun_contacto",
    "sin_gmb",
    "movil",
    "bucket_dias_crea_reg",
    "registro_nuevo_30d",
    "registro_nuevo_90d",
    "origen",
    "ct_merclie",
    "con_web",
    "con_local",
    "ant_empresa",
    "tiene_reactivacion",
]

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

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


def print_title(title):
    print("\n\n" + "=" * 100)
    print(title)
    print("=" * 100)


def resumen_fecha(df):
    out = (
        df.groupby("fecha_carga_dt")
        .agg(
            n=(TARGET, "size"),
            positivos=(TARGET, "sum"),
            tasa_real=(TARGET, "mean"),
            score_medio=(SCORE, "mean"),
        )
        .reset_index()
        .sort_values("fecha_carga_dt")
    )

    out["error_pred_real"] = out["score_medio"] - out["tasa_real"]
    out["ratio_pred_real"] = np.where(
        out["tasa_real"] > 0,
        out["score_medio"] / out["tasa_real"],
        np.nan
    )

    aucs = []
    auprcs = []

    for fecha, g in df.groupby("fecha_carga_dt"):
        aucs.append({
            "fecha_carga_dt": fecha,
            "auc": safe_auc(g[TARGET], g[SCORE]),
            "auprc": safe_auprc(g[TARGET], g[SCORE]),
        })

    metricas = pd.DataFrame(aucs)
    out = out.merge(metricas, on="fecha_carga_dt", how="left")

    return out


def distribucion_variable(df, var):
    tmp = df.copy()
    tmp[var] = tmp[var].astype(object).where(tmp[var].notna(), "NULO")

    out = (
        tmp.groupby(["grupo_carga", var])
        .size()
        .reset_index(name="n")
    )

    total = (
        out.groupby("grupo_carga")["n"]
        .sum()
        .reset_index(name="total_grupo")
    )

    out = out.merge(total, on="grupo_carga", how="left")
    out["pct"] = out["n"] / out["total_grupo"]

    piv = out.pivot_table(
        index=var,
        columns="grupo_carga",
        values="pct",
        fill_value=0
    ).reset_index()

    for col in ["TEST_0505", "RESTO"]:
        if col not in piv.columns:
            piv[col] = 0

    piv["diff_pct_test_menos_resto"] = piv["TEST_0505"] - piv["RESTO"]
    piv["abs_diff_pct"] = piv["diff_pct_test_menos_resto"].abs()
    piv["variable"] = var

    return piv.sort_values("abs_diff_pct", ascending=False)


def resumen_modelo_por_grupo(df, group_col):
    rows = []

    for grupo, g in df.groupby(group_col, dropna=False):
        rows.append({
            "grupo": grupo,
            "n": len(g),
            "positivos": g[TARGET].sum(),
            "tasa_real": g[TARGET].mean(),
            "score_medio": g[SCORE].mean(),
            "error_pred_real": g[SCORE].mean() - g[TARGET].mean(),
            "ratio_pred_real": (
                g[SCORE].mean() / g[TARGET].mean()
                if g[TARGET].mean() > 0 else np.nan
            ),
            "auc": safe_auc(g[TARGET], g[SCORE]),
            "auprc": safe_auprc(g[TARGET], g[SCORE]),
        })

    return pd.DataFrame(rows)


# ============================================================
# 1. CARGA
# ============================================================

print("Leyendo datos...")

df = pd.read_csv(CSV_INPUT, low_memory=False)
df.columns = df.columns.str.strip()

print("Filas:", len(df))
print("Columnas:", len(df.columns))

# Recalcula common por si faltan variables derivadas
df = add_common_features(df)

required = [COL_FECHA, TARGET, SCORE]

faltan = [c for c in required if c not in df.columns]
if faltan:
    raise ValueError(f"Faltan columnas: {faltan}")

df[COL_FECHA] = parse_date(df[COL_FECHA])
df["fecha_carga_dt"] = df[COL_FECHA].dt.normalize()

df[TARGET] = clean_numeric(df[TARGET]).fillna(0).astype(int)
df[SCORE] = clean_numeric(df[SCORE])

fecha_test = pd.to_datetime(FECHA_TEST)

df["grupo_carga"] = np.where(
    df["fecha_carga_dt"].eq(fecha_test),
    "TEST_0505",
    "RESTO"
)

df_test = df[df["grupo_carga"].eq("TEST_0505")].copy()
df_resto = df[df["grupo_carga"].eq("RESTO")].copy()

if len(df_test) == 0:
    raise ValueError(f"No hay filas para fecha test {FECHA_TEST}")


# ============================================================
# 2. RESUMEN POR FECHA
# ============================================================

res_fecha = resumen_fecha(df)

print_title("1. RESUMEN POR FE_CARGA_EFECTIVA")

print(res_fecha.tail(25))


# ============================================================
# 3. COMPARATIVA TEST VS RESTO
# ============================================================

res_grupo = resumen_modelo_por_grupo(df, "grupo_carga")

print_title("2. TEST 05/05 VS RESTO")

print(res_grupo)


# ============================================================
# 4. DISTRIBUCIONES DE VARIABLES
# ============================================================

dist_tables = {}
top_diffs = []

print_title("3. VARIABLES CON MAYOR CAMBIO DE COMPOSICIÓN")

for var in VARIABLES_ANALISIS:
    if var not in df.columns:
        print(f"Variable no encontrada, se omite: {var}")
        continue

    dist = distribucion_variable(df, var)
    dist_tables[var] = dist

    top = dist.head(10).copy()
    top_diffs.append(top)

    print(f"\n--- {var} ---")
    print(top[[var, "TEST_0505", "RESTO", "diff_pct_test_menos_resto"]])

top_diffs_df = pd.concat(top_diffs, ignore_index=True) if top_diffs else pd.DataFrame()


# ============================================================
# 5. PERFORMANCE POR VARIABLE DENTRO DEL TEST
# ============================================================

perf_tables = {}

print_title("4. PERFORMANCE DENTRO DEL TEST 05/05 POR VARIABLES")

for var in VARIABLES_ANALISIS:
    if var not in df_test.columns:
        continue

    perf = resumen_modelo_por_grupo(df_test, var)
    perf = perf.sort_values("n", ascending=False)
    perf_tables[var] = perf

    print(f"\n--- {var} ---")
    print(perf.head(20))


# ============================================================
# 6. GRÁFICOS
# ============================================================

# Gráfico 1: tasa real vs score medio por fecha
plt.figure(figsize=(12, 5))

plt.plot(
    res_fecha["fecha_carga_dt"],
    res_fecha["tasa_real"],
    marker="o",
    label="Tasa real contacto"
)

plt.plot(
    res_fecha["fecha_carga_dt"],
    res_fecha["score_medio"],
    marker="o",
    label="Score medio contacto"
)

plt.axvline(
    fecha_test,
    linestyle="--",
    label="Test 05/05"
)

plt.title("Contacto real vs score medio por fecha de carga")
plt.xlabel("Fecha carga efectiva")
plt.ylabel("Tasa / score")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

path = os.path.join(OUT_DIR, "contacto_real_vs_score_por_fecha.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.show()

# Gráfico 2: ratio pred/real por fecha
plt.figure(figsize=(12, 5))

plt.plot(
    res_fecha["fecha_carga_dt"],
    res_fecha["ratio_pred_real"],
    marker="o"
)

plt.axhline(1, linestyle="--")
plt.axvline(fecha_test, linestyle="--")

plt.title("Ratio score medio / tasa real por fecha de carga")
plt.xlabel("Fecha carga efectiva")
plt.ylabel("Ratio predicho / real")
plt.xticks(rotation=45)
plt.tight_layout()

path = os.path.join(OUT_DIR, "ratio_pred_real_contacto_por_fecha.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.show()

# Gráfico 3: mayores cambios de composición
if not top_diffs_df.empty:
    plot_df = top_diffs_df.copy()
    plot_df["item"] = plot_df["variable"] + " = " + plot_df.iloc[:, 0].astype(str)

    plot_df = (
        plot_df
        .sort_values("abs_diff_pct", ascending=False)
        .head(20)
        .sort_values("diff_pct_test_menos_resto")
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        plot_df["item"],
        plot_df["diff_pct_test_menos_resto"]
    )

    plt.axvline(0, linestyle="--")
    plt.title("Mayores cambios de composición: TEST 05/05 vs RESTO")
    plt.xlabel("Diferencia de porcentaje en test vs resto")
    plt.ylabel("Variable = categoría")
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "mayores_cambios_composicion_0505.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# 7. EXPORTAR
# ============================================================

with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
    res_fecha.to_excel(writer, sheet_name="resumen_por_fecha", index=False)
    res_grupo.to_excel(writer, sheet_name="test_vs_resto", index=False)

    if not top_diffs_df.empty:
        top_diffs_df.to_excel(writer, sheet_name="top_cambios_composicion", index=False)

    for var, dist in dist_tables.items():
        sheet = f"dist_{var}"[:31]
        dist.to_excel(writer, sheet_name=sheet, index=False)

    for var, perf in perf_tables.items():
        sheet = f"perf_test_{var}"[:31]
        perf.to_excel(writer, sheet_name=sheet, index=False)

print(f"\nExcel generado: {OUT_EXCEL}")
print(f"Gráficos guardados en: {OUT_DIR}")