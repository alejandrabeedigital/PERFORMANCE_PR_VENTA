# analizar_fe_crea_reg.py
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_IN = "t_pr_venta_update_17jul26.csv"

# Nombre real de la variable en el fichero
VARIABLE_FECHA = "fe_crea_reg"

# Fecha respecto a la que se calcula la antigüedad del registro
FECHA_REFERENCIA = "fe_carga_efectiva"

TARGET = "ganada"
COLUMNA_CONTACTO = "camp_total_descuelgues"

OUTPUT_EXCEL = "analisis_fe_crea_reg.xlsx"
CARPETA_GRAFICOS = Path("graficos_fe_crea_reg")


# ============================================================
# LECTURA DEL CSV
# ============================================================

def detectar_formato_csv(path: Path) -> tuple[str, str]:
    """
    Detecta de forma sencilla si el archivo utiliza:

    - separador ; y decimal ,
    - separador , y decimal .
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]

    primera_linea = None

    for encoding in encodings:
        try:
            with path.open(
                "r",
                encoding=encoding,
                errors="strict",
            ) as archivo:
                primera_linea = archivo.readline()
            break
        except UnicodeDecodeError:
            continue

    if primera_linea is None:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as archivo:
            primera_linea = archivo.readline()

    n_punto_coma = primera_linea.count(";")
    n_comas = primera_linea.count(",")

    if n_punto_coma > n_comas:
        return ";", ","

    return ",", "."


def leer_csv(path: Path) -> pd.DataFrame:
    """
    Lee el CSV probando diferentes codificaciones.
    """
    sep, decimal = detectar_formato_csv(path)

    print(f"Separador detectado: {repr(sep)}")
    print(f"Decimal detectado: {repr(decimal)}")

    errores = []

    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                decimal=decimal,
                encoding=encoding,
                low_memory=False,
            )

            print(f"Codificación utilizada: {encoding}")
            return df

        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errores.append(f"{encoding}: {exc}")

    raise ValueError(
        "No se pudo leer correctamente el CSV.\n"
        + "\n".join(errores)
    )


# ============================================================
# LIMPIEZA
# ============================================================

def limpiar_numero(serie: pd.Series) -> pd.Series:
    """
    Convierte una columna a numérico, admitiendo coma decimal.
    """
    texto = (
        serie.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "NAN": pd.NA,
                "None": pd.NA,
                "NONE": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )

    return pd.to_numeric(texto, errors="coerce")


def limpiar_target_binario(serie: pd.Series) -> pd.Series:
    """
    Convierte diferentes representaciones de un target binario a 0/1.
    """
    if pd.api.types.is_bool_dtype(serie):
        return serie.astype("Int64")

    texto = serie.astype("string").str.strip().str.upper()

    salida = texto.map(
        {
            "TRUE": 1,
            "FALSE": 0,
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0,
            "SI": 1,
            "SÍ": 1,
            "YES": 1,
            "NO": 0,
            "GANADA": 1,
            "NO GANADA": 0,
            "NO_GANADA": 0,
        }
    )

    return salida.astype("Int64")


def convertir_fecha(serie: pd.Series) -> pd.Series:
    """
    Convierte fechas intentando soportar distintos formatos.

    Primero prueba con dayfirst=False y después recupera los valores
    no convertidos usando dayfirst=True.
    """
    texto = serie.astype("string").str.strip()

    texto = texto.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "NAN": pd.NA,
            "None": pd.NA,
            "NONE": pd.NA,
            "<NA>": pd.NA,
        }
    )

    try:
        fecha = pd.to_datetime(
            texto,
            errors="coerce",
            dayfirst=False,
            format="mixed",
        )
    except (TypeError, ValueError):
        fecha = pd.to_datetime(
            texto,
            errors="coerce",
            dayfirst=False,
        )

    pendientes = fecha.isna() & texto.notna()

    if pendientes.any():
        try:
            fecha_alternativa = pd.to_datetime(
                texto.loc[pendientes],
                errors="coerce",
                dayfirst=True,
                format="mixed",
            )
        except (TypeError, ValueError):
            fecha_alternativa = pd.to_datetime(
                texto.loc[pendientes],
                errors="coerce",
                dayfirst=True,
            )

        fecha.loc[pendientes] = fecha_alternativa

    return fecha


def safe_float(valor):
    if pd.isna(valor):
        return np.nan

    return float(valor)


# ============================================================
# BUCKETS
# ============================================================

def crear_bucket_antiguedad(dias: pd.Series) -> pd.Series:
    """
    Agrupa la antigüedad del registro en intervalos interpretables.
    """
    bins = [
        -np.inf,
        -1,
        0,
        7,
        30,
        90,
        180,
        365,
        730,
        1825,
        np.inf,
    ]

    labels = [
        "00_fecha_creacion_posterior_carga",
        "01_mismo_dia",
        "02_1_7_dias",
        "03_8_30_dias",
        "04_31_90_dias",
        "05_91_180_dias",
        "06_181_365_dias",
        "07_1_2_anios",
        "08_2_5_anios",
        "09_mas_5_anios",
    ]

    resultado = pd.cut(
        dias,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    # Se convierte a string para poder asignar SIN_FECHA sin problemas
    resultado = resultado.astype("string")
    resultado = resultado.fillna("SIN_FECHA")

    return resultado


# ============================================================
# RESÚMENES DE CALIDAD
# ============================================================

def resumen_calidad(
    df: pd.DataFrame,
    fecha_original: pd.Series,
) -> pd.DataFrame:
    fecha_creacion = df[VARIABLE_FECHA]
    fecha_carga = df[FECHA_REFERENCIA]

    ambas_fechas_validas = (
        fecha_creacion.notna()
        & fecha_carga.notna()
    )

    fechas_posteriores = (
        ambas_fechas_validas
        & fecha_creacion.gt(fecha_carga)
    )

    fechas_anteriores_1990 = (
        fecha_creacion.notna()
        & fecha_creacion.lt(pd.Timestamp("1990-01-01"))
    )

    hoy = pd.Timestamp.today().normalize()

    fechas_futuras_hoy = (
        fecha_creacion.notna()
        & fecha_creacion.gt(hoy)
    )

    valores_originales_informados = (
        fecha_original.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "NAN": pd.NA,
                "None": pd.NA,
                "NONE": pd.NA,
                "<NA>": pd.NA,
            }
        )
        .notna()
    )

    no_parseables = (
        valores_originales_informados
        & fecha_creacion.isna()
    )

    filas = [
        {
            "metrica": "n_filas",
            "valor": len(df),
        },
        {
            "metrica": "n_fecha_creacion_valida",
            "valor": int(fecha_creacion.notna().sum()),
        },
        {
            "metrica": "n_fecha_creacion_nula_o_no_parseable",
            "valor": int(fecha_creacion.isna().sum()),
        },
        {
            "metrica": "pct_fecha_creacion_nula_o_no_parseable",
            "valor": safe_float(fecha_creacion.isna().mean()),
        },
        {
            "metrica": "n_valores_informados_no_parseables",
            "valor": int(no_parseables.sum()),
        },
        {
            "metrica": "fecha_creacion_min",
            "valor": fecha_creacion.min(),
        },
        {
            "metrica": "fecha_creacion_max",
            "valor": fecha_creacion.max(),
        },
        {
            "metrica": "fecha_carga_min",
            "valor": fecha_carga.min(),
        },
        {
            "metrica": "fecha_carga_max",
            "valor": fecha_carga.max(),
        },
        {
            "metrica": "n_fecha_creacion_posterior_a_carga",
            "valor": int(fechas_posteriores.sum()),
        },
        {
            "metrica": "pct_fecha_creacion_posterior_a_carga",
            "valor": safe_float(
                fechas_posteriores.sum()
                / max(ambas_fechas_validas.sum(), 1)
            ),
        },
        {
            "metrica": "n_fecha_creacion_antes_1990",
            "valor": int(fechas_anteriores_1990.sum()),
        },
        {
            "metrica": "n_fecha_creacion_futura_respecto_hoy",
            "valor": int(fechas_futuras_hoy.sum()),
        },
    ]

    return pd.DataFrame(filas)


def resumen_antiguedad(df: pd.DataFrame) -> pd.DataFrame:
    serie = df["dias_desde_creacion_registro"].dropna()

    if serie.empty:
        return pd.DataFrame(
            columns=[
                "percentil",
                "dias_desde_creacion_registro",
            ]
        )

    percentiles = serie.quantile(
        [
            0,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1,
        ]
    )

    return pd.DataFrame(
        {
            "percentil": percentiles.index,
            "dias_desde_creacion_registro": percentiles.values,
        }
    )


def analizar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprueba si un mismo identificador aparece asociado a más de una
    fecha de creación.
    """
    resultados = []

    for clave in ["id_opp", "co_cliente"]:
        if clave not in df.columns:
            continue

        tmp = (
            df.loc[df[clave].notna()]
            .groupby(clave)[VARIABLE_FECHA]
            .nunique(dropna=True)
        )

        n_claves = len(tmp)
        n_inconsistentes = int((tmp > 1).sum())

        resultados.append(
            {
                "clave": clave,
                "n_claves": n_claves,
                "n_claves_con_mas_de_una_fecha_creacion": n_inconsistentes,
                "pct_claves_con_mas_de_una_fecha_creacion": (
                    n_inconsistentes / n_claves
                    if n_claves > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(resultados)


def obtener_valores_no_parseables(
    fecha_original: pd.Series,
    fecha_convertida: pd.Series,
) -> pd.DataFrame:
    texto = (
        fecha_original.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "NAN": pd.NA,
                "None": pd.NA,
                "NONE": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )

    mascara = texto.notna() & fecha_convertida.isna()

    if not mascara.any():
        return pd.DataFrame(
            columns=[
                "valor_original",
                "n",
            ]
        )

    return (
        texto.loc[mascara]
        .value_counts(dropna=False)
        .rename_axis("valor_original")
        .reset_index(name="n")
        .head(200)
    )


# ============================================================
# ANÁLISIS CONTRA EL TARGET
# ============================================================

def analizar_por_bucket(
    df_modelable: pd.DataFrame,
) -> pd.DataFrame:
    agregaciones = {
        "n": (TARGET, "size"),
        "ventas": (TARGET, "sum"),
        "tasa_venta": (TARGET, "mean"),
        "antiguedad_media_dias": (
            "dias_desde_creacion_registro",
            "mean",
        ),
        "antiguedad_mediana_dias": (
            "dias_desde_creacion_registro",
            "median",
        ),
    }

    if "q_rk_score" in df_modelable.columns:
        agregaciones["score_medio_q_rk"] = (
            "q_rk_score",
            "mean",
        )

    tabla = (
        df_modelable
        .groupby(
            "bucket_fe_crea_reg",
            dropna=False,
        )
        .agg(**agregaciones)
        .reset_index()
    )

    orden_buckets = [
        "00_fecha_creacion_posterior_carga",
        "01_mismo_dia",
        "02_1_7_dias",
        "03_8_30_dias",
        "04_31_90_dias",
        "05_91_180_dias",
        "06_181_365_dias",
        "07_1_2_anios",
        "08_2_5_anios",
        "09_mas_5_anios",
        "SIN_FECHA",
    ]

    tabla["orden"] = tabla["bucket_fe_crea_reg"].map(
        {
            bucket: posicion
            for posicion, bucket in enumerate(orden_buckets)
        }
    )

    tabla = (
        tabla.sort_values("orden")
        .drop(columns="orden")
        .reset_index(drop=True)
    )

    tasa_global = df_modelable[TARGET].mean()
    total_ventas = df_modelable[TARGET].sum()

    tabla["pct_poblacion"] = tabla["n"] / tabla["n"].sum()

    if total_ventas > 0:
        tabla["pct_ventas"] = tabla["ventas"] / total_ventas
    else:
        tabla["pct_ventas"] = np.nan

    if tasa_global > 0:
        tabla["lift_vs_media"] = (
            tabla["tasa_venta"] / tasa_global
        )
    else:
        tabla["lift_vs_media"] = np.nan

    return tabla


def analizar_por_mes_creacion(
    df_modelable: pd.DataFrame,
) -> pd.DataFrame:
    tmp = df_modelable.loc[
        df_modelable[VARIABLE_FECHA].notna()
    ].copy()

    if tmp.empty:
        return pd.DataFrame(
            columns=[
                "mes_creacion_registro",
                "n",
                "ventas",
                "tasa_venta",
                "antiguedad_media_dias",
                "lift_vs_media",
            ]
        )

    tmp["mes_creacion_registro"] = (
        tmp[VARIABLE_FECHA]
        .dt.to_period("M")
        .astype(str)
    )

    tabla = (
        tmp.groupby(
            "mes_creacion_registro",
            dropna=False,
        )
        .agg(
            n=(TARGET, "size"),
            ventas=(TARGET, "sum"),
            tasa_venta=(TARGET, "mean"),
            antiguedad_media_dias=(
                "dias_desde_creacion_registro",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("mes_creacion_registro")
    )

    tasa_global = tmp[TARGET].mean()

    if tasa_global > 0:
        tabla["lift_vs_media"] = (
            tabla["tasa_venta"] / tasa_global
        )
    else:
        tabla["lift_vs_media"] = np.nan

    return tabla


def analizar_por_fecha_carga(
    df_modelable: pd.DataFrame,
) -> pd.DataFrame:
    tabla = (
        df_modelable
        .groupby(
            FECHA_REFERENCIA,
            dropna=False,
        )
        .agg(
            n=(TARGET, "size"),
            ventas=(TARGET, "sum"),
            tasa_venta=(TARGET, "mean"),
            pct_nulos_fe_crea_reg=(
                VARIABLE_FECHA,
                lambda x: x.isna().mean(),
            ),
            pct_fechas_posteriores_carga=(
                "fecha_creacion_posterior_carga",
                "mean",
            ),
            antiguedad_media_dias=(
                "dias_desde_creacion_registro",
                "mean",
            ),
            antiguedad_mediana_dias=(
                "dias_desde_creacion_registro",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            FECHA_REFERENCIA,
            na_position="last",
        )
    )

    return tabla


def analizar_target_por_nulos(
    df_modelable: pd.DataFrame,
) -> pd.DataFrame:
    tmp = df_modelable.copy()

    tmp["estado_fe_crea_reg"] = np.where(
        tmp[VARIABLE_FECHA].isna(),
        "SIN_FECHA",
        "CON_FECHA",
    )

    tabla = (
        tmp.groupby(
            "estado_fe_crea_reg",
            dropna=False,
        )
        .agg(
            n=(TARGET, "size"),
            ventas=(TARGET, "sum"),
            tasa_venta=(TARGET, "mean"),
        )
        .reset_index()
    )

    tasa_global = tmp[TARGET].mean()

    if tasa_global > 0:
        tabla["lift_vs_media"] = (
            tabla["tasa_venta"] / tasa_global
        )
    else:
        tabla["lift_vs_media"] = np.nan

    return tabla


# ============================================================
# GRÁFICOS
# ============================================================

def generar_graficos(
    tabla_buckets: pd.DataFrame,
    tabla_temporal: pd.DataFrame,
) -> None:
    CARPETA_GRAFICOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not tabla_buckets.empty:
        graf = tabla_buckets.copy()

        graf["bucket_fe_crea_reg"] = (
            graf["bucket_fe_crea_reg"].astype(str)
        )

        plt.figure(figsize=(12, 6))
        plt.bar(
            graf["bucket_fe_crea_reg"],
            graf["tasa_venta"],
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Antigüedad del registro")
        plt.ylabel("Tasa de venta")
        plt.title(
            "Tasa de venta por antigüedad de fe_crea_reg"
        )
        plt.tight_layout()
        plt.savefig(
            CARPETA_GRAFICOS
            / "01_tasa_venta_por_antiguedad.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.bar(
            graf["bucket_fe_crea_reg"],
            graf["n"],
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Antigüedad del registro")
        plt.ylabel("Número de registros")
        plt.title(
            "Volumen por antigüedad de fe_crea_reg"
        )
        plt.tight_layout()
        plt.savefig(
            CARPETA_GRAFICOS
            / "02_volumen_por_antiguedad.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    if tabla_temporal.empty:
        return

    temporal = tabla_temporal.dropna(
        subset=[FECHA_REFERENCIA]
    ).copy()

    if temporal.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(
        temporal[FECHA_REFERENCIA],
        temporal["pct_nulos_fe_crea_reg"],
        marker="o",
    )
    plt.xlabel("Fecha de carga")
    plt.ylabel("% nulos fe_crea_reg")
    plt.title("Evolución de nulos de fe_crea_reg")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        CARPETA_GRAFICOS
        / "03_nulos_por_fecha_carga.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        temporal[FECHA_REFERENCIA],
        temporal["antiguedad_mediana_dias"],
        marker="o",
    )
    plt.xlabel("Fecha de carga")
    plt.ylabel("Mediana de antigüedad en días")
    plt.title(
        "Evolución de la antigüedad mediana del registro"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        CARPETA_GRAFICOS
        / "04_antiguedad_mediana_por_carga.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        temporal[FECHA_REFERENCIA],
        temporal["tasa_venta"],
        marker="o",
    )
    plt.xlabel("Fecha de carga")
    plt.ylabel("Tasa de venta")
    plt.title("Tasa de venta por fecha de carga")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        CARPETA_GRAFICOS
        / "05_tasa_venta_por_fecha_carga.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# RECOMENDACIÓN AUTOMÁTICA
# ============================================================

def generar_recomendacion(
    df: pd.DataFrame,
    df_modelable: pd.DataFrame,
    tabla_buckets: pd.DataFrame,
) -> pd.DataFrame:
    pct_nulos = df[VARIABLE_FECHA].isna().mean()

    fechas_comparables = (
        df[VARIABLE_FECHA].notna()
        & df[FECHA_REFERENCIA].notna()
    )

    if fechas_comparables.any():
        pct_fechas_posteriores = (
            df.loc[
                fechas_comparables,
                VARIABLE_FECHA,
            ]
            .gt(
                df.loc[
                    fechas_comparables,
                    FECHA_REFERENCIA,
                ]
            )
            .mean()
        )
    else:
        pct_fechas_posteriores = np.nan

    buckets_validos = tabla_buckets.loc[
        tabla_buckets["n"].ge(500)
        & tabla_buckets["tasa_venta"].notna()
        & tabla_buckets["bucket_fe_crea_reg"].ne(
            "00_fecha_creacion_posterior_carga"
        )
    ].copy()

    tasas_positivas = buckets_validos.loc[
        buckets_validos["tasa_venta"].gt(0),
        "tasa_venta",
    ]

    if len(tasas_positivas) >= 2:
        ratio_max_min = (
            tasas_positivas.max()
            / tasas_positivas.min()
        )
    else:
        ratio_max_min = np.nan

    correlacion = np.nan

    datos_correlacion = df_modelable[
        [
            "dias_desde_creacion_registro",
            TARGET,
        ]
    ].dropna()

    if (
        len(datos_correlacion) > 1
        and datos_correlacion[
            "dias_desde_creacion_registro"
        ].nunique() > 1
        and datos_correlacion[TARGET].nunique() > 1
    ):
        correlacion = datos_correlacion[
            "dias_desde_creacion_registro"
        ].corr(
            datos_correlacion[TARGET],
            method="spearman",
        )

    incidencias = []

    if pct_nulos > 0.30:
        incidencias.append(
            f"Nulos elevados: {pct_nulos:.2%}."
        )
    elif pct_nulos > 0.10:
        incidencias.append(
            f"Nulos moderados: {pct_nulos:.2%}."
        )

    if (
        pd.notna(pct_fechas_posteriores)
        and pct_fechas_posteriores > 0.001
    ):
        incidencias.append(
            "Existen fechas de creación posteriores a la carga: "
            f"{pct_fechas_posteriores:.2%}."
        )

    if (
        pd.notna(ratio_max_min)
        and ratio_max_min >= 1.5
    ):
        incidencias.append(
            "La tasa de venta cambia entre buckets de antigüedad: "
            f"ratio máximo/mínimo = {ratio_max_min:.2f}."
        )

    mala_calidad = (
        pct_nulos > 0.30
        or (
            pd.notna(pct_fechas_posteriores)
            and pct_fechas_posteriores > 0.01
        )
    )

    existe_senal = (
        pd.notna(ratio_max_min)
        and ratio_max_min >= 1.5
    )

    if mala_calidad:
        decision = "REVISAR_CALIDAD_ANTES_DE_USAR"
    elif existe_senal:
        decision = "VARIABLE_APTA_Y_CON_SEÑAL"
    else:
        decision = "VARIABLE_APTA_PERO_SEÑAL_LIMITADA"

    return pd.DataFrame(
        [
            {
                "decision": decision,
                "pct_nulos": pct_nulos,
                "pct_fechas_posteriores_carga": (
                    pct_fechas_posteriores
                ),
                "ratio_tasa_max_min_buckets": ratio_max_min,
                "correlacion_spearman_antiguedad_target": correlacion,
                "motivos": (
                    " ".join(incidencias)
                    if incidencias
                    else "Sin incidencias relevantes detectadas."
                ),
            }
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    path = Path(ARCHIVO_IN)

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el fichero: {path.resolve()}"
        )

    print("=" * 90)
    print("ANÁLISIS DE fe_crea_reg")
    print("=" * 90)
    print(f"Archivo: {path.resolve()}")

    df = leer_csv(path)

    print(f"Filas leídas: {len(df):,}")
    print(f"Columnas leídas: {len(df.columns):,}")

    # Elimina espacios accidentales de los nombres
    df.columns = df.columns.astype(str).str.strip()

    columnas_obligatorias = [
        VARIABLE_FECHA,
        FECHA_REFERENCIA,
        TARGET,
        COLUMNA_CONTACTO,
    ]

    missing = [
        columna
        for columna in columnas_obligatorias
        if columna not in df.columns
    ]

    if missing:
        print("\nColumnas disponibles similares:")

        for buscada in missing:
            similares = [
                columna
                for columna in df.columns
                if (
                    "fecha" in columna.lower()
                    or "crea" in columna.lower()
                    or "carga" in columna.lower()
                    or buscada.lower() in columna.lower()
                )
            ]

            print(f"- Para {buscada}: {similares[:30]}")

        raise ValueError(
            "Faltan columnas obligatorias para el análisis: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Copia del valor original antes de convertirlo
    # --------------------------------------------------------

    fecha_original = df[VARIABLE_FECHA].copy()

    # --------------------------------------------------------
    # Normalización
    # --------------------------------------------------------

    print("\nConvirtiendo fechas...")

    df[VARIABLE_FECHA] = convertir_fecha(
        df[VARIABLE_FECHA]
    )

    df[FECHA_REFERENCIA] = convertir_fecha(
        df[FECHA_REFERENCIA]
    )

    print("Convirtiendo target y contacto...")

    df[TARGET] = limpiar_target_binario(
        df[TARGET]
    )

    df[COLUMNA_CONTACTO] = limpiar_numero(
        df[COLUMNA_CONTACTO]
    )

    if "q_rk_score" in df.columns:
        df["q_rk_score"] = limpiar_numero(
            df["q_rk_score"]
        )

    # --------------------------------------------------------
    # Variables derivadas
    # --------------------------------------------------------

    df["dias_desde_creacion_registro"] = (
        df[FECHA_REFERENCIA]
        - df[VARIABLE_FECHA]
    ).dt.total_seconds() / 86400

    df["dias_desde_creacion_registro"] = np.floor(
        df["dias_desde_creacion_registro"]
    )

    df["fecha_creacion_posterior_carga"] = (
        df[VARIABLE_FECHA].notna()
        & df[FECHA_REFERENCIA].notna()
        & df[VARIABLE_FECHA].gt(
            df[FECHA_REFERENCIA]
        )
    )

    df["bucket_fe_crea_reg"] = crear_bucket_antiguedad(
        df["dias_desde_creacion_registro"]
    )

    # --------------------------------------------------------
    # Calidad general
    # --------------------------------------------------------

    print("Calculando métricas de calidad...")

    calidad = resumen_calidad(
        df=df,
        fecha_original=fecha_original,
    )

    antiguedad = resumen_antiguedad(df)

    duplicados = analizar_duplicados(df)

    valores_no_parseables = obtener_valores_no_parseables(
        fecha_original=fecha_original,
        fecha_convertida=df[VARIABLE_FECHA],
    )

    # --------------------------------------------------------
    # Universo modelable de venta:
    # registros contactados y target válido
    # --------------------------------------------------------

    mascara_modelable = (
        df[COLUMNA_CONTACTO].fillna(0).gt(0)
        & df[TARGET].isin([0, 1])
    )

    df_modelable = df.loc[mascara_modelable].copy()

    if df_modelable.empty:
        raise ValueError(
            "No hay registros contactados con target ganada válido."
        )

    df_modelable[TARGET] = (
        df_modelable[TARGET].astype(int)
    )

    print("\n--- UNIVERSO ---")
    print(f"Filas totales: {len(df):,}")
    print(
        "Filas contactadas con target válido: "
        f"{len(df_modelable):,}"
    )
    print(
        f"Ventas: {int(df_modelable[TARGET].sum()):,}"
    )
    print(
        "Tasa de venta post-contacto: "
        f"{df_modelable[TARGET].mean():.6%}"
    )
    print(
        "% nulos de fe_crea_reg en universo modelable: "
        f"{df_modelable[VARIABLE_FECHA].isna().mean():.2%}"
    )

    # --------------------------------------------------------
    # Relación con la venta
    # --------------------------------------------------------

    print("\nAnalizando relación con la venta...")

    tabla_buckets = analizar_por_bucket(
        df_modelable
    )

    tabla_mes_creacion = analizar_por_mes_creacion(
        df_modelable
    )

    tabla_temporal = analizar_por_fecha_carga(
        df_modelable
    )

    tabla_nulos = analizar_target_por_nulos(
        df_modelable
    )

    resumen_modelable = pd.DataFrame(
        [
            {
                "n_total_archivo": len(df),
                "n_postcontacto_target_valido": len(
                    df_modelable
                ),
                "ventas_postcontacto": int(
                    df_modelable[TARGET].sum()
                ),
                "tasa_venta_postcontacto": float(
                    df_modelable[TARGET].mean()
                ),
                "n_fe_crea_reg_valida_postcontacto": int(
                    df_modelable[
                        VARIABLE_FECHA
                    ].notna().sum()
                ),
                "pct_nulos_fe_crea_reg_postcontacto": float(
                    df_modelable[
                        VARIABLE_FECHA
                    ].isna().mean()
                ),
                "n_fechas_posteriores_carga_postcontacto": int(
                    df_modelable[
                        "fecha_creacion_posterior_carga"
                    ].sum()
                ),
                "media_antiguedad_dias_postcontacto": safe_float(
                    df_modelable[
                        "dias_desde_creacion_registro"
                    ].mean()
                ),
                "mediana_antiguedad_dias_postcontacto": safe_float(
                    df_modelable[
                        "dias_desde_creacion_registro"
                    ].median()
                ),
                "min_antiguedad_dias_postcontacto": safe_float(
                    df_modelable[
                        "dias_desde_creacion_registro"
                    ].min()
                ),
                "max_antiguedad_dias_postcontacto": safe_float(
                    df_modelable[
                        "dias_desde_creacion_registro"
                    ].max()
                ),
            }
        ]
    )

    recomendacion = generar_recomendacion(
        df=df,
        df_modelable=df_modelable,
        tabla_buckets=tabla_buckets,
    )

    # --------------------------------------------------------
    # Muestra de incidencias
    # --------------------------------------------------------

    columnas_muestra = [
        columna
        for columna in [
            "co_cliente",
            "id_opp",
            VARIABLE_FECHA,
            FECHA_REFERENCIA,
            "dias_desde_creacion_registro",
            COLUMNA_CONTACTO,
            TARGET,
        ]
        if columna in df.columns
    ]

    muestra_fechas_posteriores = (
        df.loc[
            df["fecha_creacion_posterior_carga"],
            columnas_muestra,
        ]
        .head(5000)
        .copy()
    )

    # --------------------------------------------------------
    # Exportación
    # --------------------------------------------------------

    print("\nGenerando Excel...")

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl",
    ) as writer:
        recomendacion.to_excel(
            writer,
            sheet_name="recomendacion",
            index=False,
        )

        resumen_modelable.to_excel(
            writer,
            sheet_name="resumen_modelable",
            index=False,
        )

        calidad.to_excel(
            writer,
            sheet_name="calidad_general",
            index=False,
        )

        antiguedad.to_excel(
            writer,
            sheet_name="percentiles_antiguedad",
            index=False,
        )

        tabla_buckets.to_excel(
            writer,
            sheet_name="venta_por_antiguedad",
            index=False,
        )

        tabla_nulos.to_excel(
            writer,
            sheet_name="venta_con_sin_fecha",
            index=False,
        )

        tabla_mes_creacion.to_excel(
            writer,
            sheet_name="venta_por_mes_creacion",
            index=False,
        )

        tabla_temporal.to_excel(
            writer,
            sheet_name="estabilidad_por_carga",
            index=False,
        )

        duplicados.to_excel(
            writer,
            sheet_name="consistencia_por_id",
            index=False,
        )

        valores_no_parseables.to_excel(
            writer,
            sheet_name="fechas_no_parseables",
            index=False,
        )

        muestra_fechas_posteriores.to_excel(
            writer,
            sheet_name="muestra_fechas_posteriores",
            index=False,
        )

    print("Generando gráficos...")

    generar_graficos(
        tabla_buckets=tabla_buckets,
        tabla_temporal=tabla_temporal,
    )

    # --------------------------------------------------------
    # Salida por consola
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("RESULTADO")
    print("=" * 90)

    print("\n--- RECOMENDACIÓN ---")
    print(
        recomendacion.to_string(
            index=False
        )
    )

    print("\n--- RESUMEN DEL UNIVERSO MODELABLE ---")
    print(
        resumen_modelable.to_string(
            index=False
        )
    )

    print("\n--- VENTA POR ANTIGÜEDAD ---")
    print(
        tabla_buckets.to_string(
            index=False
        )
    )

    print("\n--- VENTA CON Y SIN FECHA ---")
    print(
        tabla_nulos.to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print(f"Excel generado: {Path(OUTPUT_EXCEL).resolve()}")
    print(
        "Gráficos generados en: "
        f"{CARPETA_GRAFICOS.resolve()}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()