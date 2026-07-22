import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common_pr_venta_8jun26 import clean_numeric, parse_date


# ============================================================
# CONFIG
# ============================================================

CSV_INPUT = "todo_con_prob_descuelgue.csv"

OUT_EXCEL = "diagnostico_maduracion_descuelgue.xlsx"
OUT_DIR = "graficos_diagnostico_maduracion"

COL_FECHA_CARGA = "fe_carga_efectiva"
COL_DESCUELGUES = "camp_total_descuelgues"
COL_SCORE = "prob_descuelgue_modelo"

FECHA_FOCO = "2026-05-05"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

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

required = [
    COL_FECHA_CARGA,
    COL_DESCUELGUES,
    COL_SCORE
]

faltan = [c for c in required if c not in df.columns]
if faltan:
    raise ValueError(f"Faltan columnas: {faltan}")

df["fecha_carga_dt"] = parse_date(df[COL_FECHA_CARGA]).dt.normalize()
df[COL_DESCUELGUES] = clean_numeric(df[COL_DESCUELGUES]).fillna(0)
df[COL_SCORE] = clean_numeric(df[COL_SCORE])

df["target_descuelgue_calc"] = np.where(
    df[COL_DESCUELGUES] > 0,
    1,
    0
)

df["fecha_foco"] = np.where(
    df["fecha_carga_dt"].eq(pd.to_datetime(FECHA_FOCO)),
    "FOCO_0505",
    "RESTO"
)

print("Filas:", len(df))
print("Fecha mínima carga:", df["fecha_carga_dt"].min())
print("Fecha máxima carga:", df["fecha_carga_dt"].max())


# ============================================================
# 2. RESUMEN POR FECHA DE CARGA
# ============================================================

res_fecha = (
    df.groupby("fecha_carga_dt")
    .agg(
        n=("target_descuelgue_calc", "size"),
        registros_con_descuelgue=("target_descuelgue_calc", "sum"),
        total_descuelgues=(COL_DESCUELGUES, "sum"),
        media_descuelgues=(COL_DESCUELGUES, "mean"),
        mediana_descuelgues=(COL_DESCUELGUES, "median"),
        score_medio=(COL_SCORE, "mean")
    )
    .reset_index()
    .sort_values("fecha_carga_dt")
)

res_fecha["tasa_descuelgue"] = (
    res_fecha["registros_con_descuelgue"] / res_fecha["n"]
)

res_fecha["descuelgues_por_registro"] = (
    res_fecha["total_descuelgues"] / res_fecha["n"]
)

res_fecha["ratio_score_real"] = np.where(
    res_fecha["tasa_descuelgue"] > 0,
    res_fecha["score_medio"] / res_fecha["tasa_descuelgue"],
    np.nan
)

print_title("1. RESUMEN POR FECHA DE CARGA")
print(res_fecha.tail(30))


# ============================================================
# 3. FOCO 05/05 VS RESTO
# ============================================================

res_foco = (
    df.groupby("fecha_foco")
    .agg(
        n=("target_descuelgue_calc", "size"),
        registros_con_descuelgue=("target_descuelgue_calc", "sum"),
        total_descuelgues=(COL_DESCUELGUES, "sum"),
        media_descuelgues=(COL_DESCUELGUES, "mean"),
        mediana_descuelgues=(COL_DESCUELGUES, "median"),
        score_medio=(COL_SCORE, "mean")
    )
    .reset_index()
)

res_foco["tasa_descuelgue"] = (
    res_foco["registros_con_descuelgue"] / res_foco["n"]
)

res_foco["descuelgues_por_registro"] = (
    res_foco["total_descuelgues"] / res_foco["n"]
)

res_foco["ratio_score_real"] = np.where(
    res_foco["tasa_descuelgue"] > 0,
    res_foco["score_medio"] / res_foco["tasa_descuelgue"],
    np.nan
)

print_title("2. 05/05 VS RESTO")
print(res_foco)


# ============================================================
# 4. ÚLTIMAS CARGAS
# ============================================================

ultimas_fechas = (
    res_fecha["fecha_carga_dt"]
    .dropna()
    .sort_values()
    .tail(12)
    .tolist()
)

df_ultimas = df[df["fecha_carga_dt"].isin(ultimas_fechas)].copy()

res_ultimas = (
    df_ultimas.groupby("fecha_carga_dt")
    .agg(
        n=("target_descuelgue_calc", "size"),
        registros_con_descuelgue=("target_descuelgue_calc", "sum"),
        total_descuelgues=(COL_DESCUELGUES, "sum"),
        score_medio=(COL_SCORE, "mean")
    )
    .reset_index()
    .sort_values("fecha_carga_dt")
)

res_ultimas["tasa_descuelgue"] = (
    res_ultimas["registros_con_descuelgue"] / res_ultimas["n"]
)

res_ultimas["descuelgues_por_registro"] = (
    res_ultimas["total_descuelgues"] / res_ultimas["n"]
)

res_ultimas["ratio_score_real"] = np.where(
    res_ultimas["tasa_descuelgue"] > 0,
    res_ultimas["score_medio"] / res_ultimas["tasa_descuelgue"],
    np.nan
)

print_title("3. ÚLTIMAS 12 CARGAS")
print(res_ultimas)


# ============================================================
# 5. DISTRIBUCIÓN DE Nº DE DESCUELGUES POR REGISTRO
# ============================================================

df["bucket_num_descuelgues"] = pd.cut(
    df[COL_DESCUELGUES],
    bins=[-1, 0, 1, 2, 3, 5, 10, np.inf],
    labels=[
        "0",
        "1",
        "2",
        "3",
        "4_5",
        "6_10",
        "mas_10"
    ]
).astype(object)

dist_desc = (
    df.groupby(["fecha_foco", "bucket_num_descuelgues"])
    .size()
    .reset_index(name="n")
)

dist_desc["pct"] = (
    dist_desc["n"]
    / dist_desc.groupby("fecha_foco")["n"].transform("sum")
)

print_title("4. DISTRIBUCIÓN Nº DESCUELGUES POR REGISTRO")
print(dist_desc)


# ============================================================
# 6. GRÁFICOS
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    res_fecha["fecha_carga_dt"],
    res_fecha["tasa_descuelgue"],
    marker="o",
    label="Tasa real descuelgue"
)

plt.plot(
    res_fecha["fecha_carga_dt"],
    res_fecha["score_medio"],
    marker="o",
    label="Score medio contacto"
)

plt.axvline(
    pd.to_datetime(FECHA_FOCO),
    linestyle="--",
    label="05/05"
)

plt.title("Tasa real de descuelgue vs score medio por fecha de carga")
plt.xlabel("Fecha carga efectiva")
plt.ylabel("Tasa / score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "tasa_real_vs_score_por_fecha.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(12, 5))

plt.bar(
    res_fecha["fecha_carga_dt"].astype(str),
    res_fecha["descuelgues_por_registro"]
)

plt.axvline(
    x=str(pd.to_datetime(FECHA_FOCO).date()),
    linestyle="--"
)

plt.title("Descuelgues por registro por fecha de carga")
plt.xlabel("Fecha carga efectiva")
plt.ylabel("Descuelgues por registro")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "descuelgues_por_registro_por_fecha.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(12, 5))

plt.bar(
    res_fecha["fecha_carga_dt"].astype(str),
    res_fecha["total_descuelgues"]
)

plt.axvline(
    x=str(pd.to_datetime(FECHA_FOCO).date()),
    linestyle="--"
)

plt.title("Total de descuelgues por fecha de carga")
plt.xlabel("Fecha carga efectiva")
plt.ylabel("Total descuelgues")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "total_descuelgues_por_fecha.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. EXPORTAR
# ============================================================

with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
    res_fecha.to_excel(writer, sheet_name="resumen_por_fecha", index=False)
    res_foco.to_excel(writer, sheet_name="0505_vs_resto", index=False)
    res_ultimas.to_excel(writer, sheet_name="ultimas_12_cargas", index=False)
    dist_desc.to_excel(writer, sheet_name="dist_num_descuelgues", index=False)

print(f"\nExcel generado: {OUT_EXCEL}")
print(f"Gráficos guardados en: {OUT_DIR}")


# ============================================================
# 8. INTERPRETACIÓN RÁPIDA
# ============================================================

row_foco = res_foco[res_foco["fecha_foco"].eq("FOCO_0505")]

if not row_foco.empty:
    r = row_foco.iloc[0]

    print_title("5. LECTURA RÁPIDA 05/05")

    print(f"N 05/05: {r['n']:,.0f}")
    print(f"Registros con descuelgue: {r['registros_con_descuelgue']:,.0f}")
    print(f"Tasa descuelgue 05/05: {r['tasa_descuelgue']:.6f}")
    print(f"Score medio 05/05: {r['score_medio']:.6f}")
    print(f"Ratio score/real 05/05: {r['ratio_score_real']:.3f}")
    print(f"Total descuelgues 05/05: {r['total_descuelgues']:,.0f}")
    print(f"Descuelgues por registro 05/05: {r['descuelgues_por_registro']:.6f}")

    resto = res_foco[res_foco["fecha_foco"].eq("RESTO")]

    if not resto.empty:
        rr = resto.iloc[0]

        print("\nComparativa RESTO:")
        print(f"Tasa descuelgue resto: {rr['tasa_descuelgue']:.6f}")
        print(f"Score medio resto: {rr['score_medio']:.6f}")
        print(f"Descuelgues por registro resto: {rr['descuelgues_por_registro']:.6f}")

        if r["tasa_descuelgue"] < rr["tasa_descuelgue"] * 0.7:
            print("\nConclusión probable: la carga 05/05 tiene una tasa de descuelgue muy inferior al histórico.")
            print("Esto puede indicar menor maduración del target, cambio operativo o composición distinta de la carga.")

        if r["descuelgues_por_registro"] < rr["descuelgues_por_registro"] * 0.7:
            print("Además, el volumen de descuelgues por registro también es bajo frente al resto.")
            print("Esto refuerza la hipótesis de target todavía incompleto o carga menos trabajada.")