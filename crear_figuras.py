import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

EVAL_EXCEL = "evaluacion_pr_venta_campanas.xlsx"
ANTIG_EXCEL = "analisis_empresas_recientes_2026.xlsx"

OUT_DIR = "figuras_notion_pr_venta"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True


# ============================================================
# HELPERS
# ============================================================

def find_file(preferred_name):
    if os.path.exists(preferred_name):
        return preferred_name

    candidates = glob.glob(preferred_name.replace(".xlsx", "*.xlsx"))

    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No encuentro {preferred_name}")


def pct(x):
    return x * 100


# ============================================================
# 1. CARGAR EXCEL PRINCIPAL
# ============================================================

eval_path = find_file(EVAL_EXCEL)

print(f"Leyendo evaluación: {eval_path}")

global_perf = pd.read_excel(eval_path, sheet_name="global")
deciles = pd.read_excel(eval_path, sheet_name="deciles")
calibracion = pd.read_excel(eval_path, sheet_name="calibracion")

try:
    segmentos = pd.read_excel(eval_path, sheet_name="segmentos")
except Exception:
    segmentos = None

try:
    temporal = pd.read_excel(eval_path, sheet_name="temporal")
except Exception:
    temporal = None


# ============================================================
# 2. TABLA KPI RESUMEN
# ============================================================

global_dict = dict(zip(global_perf.iloc[:, 0], global_perf.iloc[:, 1]))

n = int(global_dict.get("n", np.nan))
ventas = int(global_dict.get("ventas", np.nan))
tasa_venta = global_dict.get("tasa_venta", np.nan)
auc = global_dict.get("auc", np.nan)
auprc = global_dict.get("auprc", np.nan)

top10 = deciles.loc[deciles["decil"] == 1, "pct_ventas_acum"].iloc[0]
top20 = deciles.loc[deciles["decil"] == 2, "pct_ventas_acum"].iloc[0]
top50 = deciles.loc[deciles["decil"] == 5, "pct_ventas_acum"].iloc[0]
lift_top = deciles.loc[deciles["decil"] == 1, "lift_vs_media"].iloc[0]

kpis = pd.DataFrame({
    "Métrica": [
        "Registros analizados",
        "Ventas observadas",
        "Tasa media de venta",
        "AUC global",
        "AUPRC",
        "Lift top 10%",
        "Ventas captadas top 10%",
        "Ventas captadas top 20%",
        "Ventas captadas top 50%",
    ],
    "Valor": [
        f"{n:,.0f}",
        f"{ventas:,.0f}",
        f"{pct(tasa_venta):.3f}%",
        f"{auc:.3f}",
        f"{auprc:.4f}",
        f"{lift_top:.2f}x",
        f"{pct(top10):.1f}%",
        f"{pct(top20):.1f}%",
        f"{pct(top50):.1f}%",
    ]
})

kpis.to_csv(
    os.path.join(OUT_DIR, "tabla_kpis_resumen.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nTABLA KPI")
print(kpis)


# ============================================================
# FIGURA 1. LIFT POR DECIL
# ============================================================

fig, ax = plt.subplots()

ax.bar(
    deciles["decil"].astype(str),
    deciles["lift_vs_media"]
)

ax.axhline(1, linestyle="--")
ax.set_title("Lift por decil de probabilidad de venta")
ax.set_xlabel("Decil de score (1 = mayor probabilidad)")
ax.set_ylabel("Lift vs media")

for i, row in deciles.iterrows():
    ax.text(
        str(int(row["decil"])),
        row["lift_vs_media"],
        f'{row["lift_vs_media"]:.1f}x',
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "01_lift_por_deciles.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FIGURA 2. CURVA ACUMULADA DE CAPTACIÓN DE VENTAS
# ============================================================

fig, ax = plt.subplots()

ax.plot(
    pct(deciles["pct_poblacion_acum"]),
    pct(deciles["pct_ventas_acum"]),
    marker="o",
    label="Modelo"
)

ax.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Aleatorio"
)

ax.set_title("Ventas captadas acumuladas según priorización del modelo")
ax.set_xlabel("% población llamada")
ax.set_ylabel("% ventas captadas")
ax.legend()

for x, y in zip(
    pct(deciles["pct_poblacion_acum"]),
    pct(deciles["pct_ventas_acum"])
):
    if x in [10, 20, 50]:
        ax.text(x, y, f"{y:.0f}%", ha="center", va="bottom")

plt.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "02_curva_acumulada_ventas.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FIGURA 3. CALIBRACIÓN: PROBABILIDAD PREDICHA VS REAL
# ============================================================

fig, ax = plt.subplots()

x = calibracion["bucket_score"]

ax.plot(
    x,
    pct(calibracion["prob_predicha"]),
    marker="o",
    label="Probabilidad predicha"
)

ax.plot(
    x,
    pct(calibracion["tasa_real"]),
    marker="o",
    label="Conversión real"
)

ax.set_title("Calibración del modelo por tramos de score")
ax.set_xlabel("Bucket de score")
ax.set_ylabel("% venta")
ax.legend()

plt.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "03_calibracion_predicho_vs_real.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 3. TABLA SEGMENTOS PROBLEMÁTICOS
# ============================================================

if segmentos is not None and "error_pred_real" in segmentos.columns:
    seg = segmentos.copy()
    seg = seg[seg["n"] >= 500].copy()
    seg["abs_error_pred_real"] = seg["error_pred_real"].abs()

    cols = [
        "variable",
        "segmento",
        "n",
        "ventas",
        "tasa_venta",
        "score_medio",
        "error_pred_real",
        "auc"
    ]

    cols = [c for c in cols if c in seg.columns]

    peores_segmentos = (
        seg.sort_values("abs_error_pred_real", ascending=False)
        .head(20)[cols]
    )

    peores_segmentos.to_csv(
        os.path.join(OUT_DIR, "tabla_segmentos_problematicos.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSEGMENTOS PROBLEMÁTICOS")
    print(peores_segmentos)


# ============================================================
# FIGURA 4. EMPRESAS RECIENTES / COHORTES
# ============================================================

try:
    antig_path = find_file(ANTIG_EXCEL)

    print(f"\nLeyendo antigüedad: {antig_path}")

    cohortes = pd.read_excel(
        antig_path,
        sheet_name="por_cohorte"
    )

    cohortes_final = cohortes[
        cohortes["score"] == "pr_final_venta"
    ].copy()

    orden = [
        "antes_2020",
        "2020_2022",
        "2023_2024",
        "2025",
        "2026"
    ]

    cohortes_final["grupo"] = pd.Categorical(
        cohortes_final["grupo"],
        categories=orden,
        ordered=True
    )

    cohortes_final = (
        cohortes_final
        .dropna(subset=["grupo"])
        .sort_values("grupo")
    )

    cohortes_final.to_csv(
        os.path.join(OUT_DIR, "tabla_cohortes_empresa.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots()

    ax.bar(
        cohortes_final["grupo"].astype(str),
        pct(cohortes_final["tasa_real"])
    )

    ax.set_title("Conversión real por año/cohorte de creación de empresa")
    ax.set_xlabel("Cohorte de creación empresa")
    ax.set_ylabel("% conversión real")

    for i, row in cohortes_final.iterrows():
        ax.text(
            str(row["grupo"]),
            pct(row["tasa_real"]),
            f'{pct(row["tasa_real"]):.2f}%',
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "04_conversion_por_cohorte_empresa.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

except Exception as e:
    print("\nNo se ha generado gráfico de cohortes.")
    print("Motivo:", e)


# ============================================================
# 4. RESUMEN MARKDOWN PARA NOTION
# ============================================================

markdown = f"""
# Bloque visual recomendado para Notion

## KPIs principales

| Métrica | Valor |
|---|---|
| Registros analizados | {n:,.0f} |
| Ventas observadas | {ventas:,.0f} |
| Tasa media de venta | {pct(tasa_venta):.3f}% |
| AUC global | {auc:.3f} |
| Lift top 10% | {lift_top:.2f}x |
| Ventas captadas top 10% | {pct(top10):.1f}% |
| Ventas captadas top 20% | {pct(top20):.1f}% |
| Ventas captadas top 50% | {pct(top50):.1f}% |

## Figuras generadas

1. `01_lift_por_deciles.png`
2. `02_curva_acumulada_ventas.png`
3. `03_calibracion_predicho_vs_real.png`
4. `04_conversion_por_cohorte_empresa.png`

## Tablas generadas

- `tabla_kpis_resumen.csv`
- `tabla_segmentos_problematicos.csv`
- `tabla_cohortes_empresa.csv`
"""

with open(
    os.path.join(OUT_DIR, "bloque_notion.md"),
    "w",
    encoding="utf-8"
) as f:
    f.write(markdown)

print("\nArchivos generados en:", OUT_DIR)
print(markdown)