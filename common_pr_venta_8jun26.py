
import pandas as pd
import numpy as np


def clean_numeric(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NAN": np.nan, "NaN": np.nan, "<NA>": np.nan}),
        errors="coerce"
    )


def parse_date(s):
    return pd.to_datetime(
        s.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NAN": np.nan, "NaN": np.nan, "<NA>": np.nan}),
        errors="coerce",
        dayfirst=False
    )


def clean_bool_01(s):
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "SI": 1, "SÍ": 1, "NO": 0})
    )


def safe_str_series(s):
    return (
        s.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NAN": np.nan, "NaN": np.nan, "<NA>": np.nan})
    )


def limpiar_id(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, "<NA>": np.nan})
    )


def add_common_features(df):
    """
    Variables comunes para modelos de pr_contact, pr_venta y pr_final.

    CAMBIO IMPORTANTE:
    dias_desde_crea_reg ahora se calcula contra fe_carga_efectiva,
    no contra fe_test. fe_test solo existe desde 14/04, por lo que no sirve
    como fecha de referencia general.
    """
    df = df.copy()

    if "camp_total_descuelgues" in df.columns:
        df["camp_total_descuelgues"] = clean_numeric(df["camp_total_descuelgues"])
        df["target_descuelgue_calc"] = np.where(df["camp_total_descuelgues"].fillna(0) > 0, 1, 0)

    # Fecha de referencia correcta para antigüedad del registro
    if "fe_carga_efectiva" in df.columns:
        df["fecha_ref_registro_dt"] = parse_date(df["fe_carga_efectiva"])
        df["fecha_ref_registro_origen"] = "fe_carga_efectiva"
    elif "fe_carga" in df.columns:
        df["fecha_ref_registro_dt"] = parse_date(df["fe_carga"])
        df["fecha_ref_registro_origen"] = "fe_carga"
    elif "fe_test" in df.columns:
        df["fecha_ref_registro_dt"] = parse_date(df["fe_test"])
        df["fecha_ref_registro_origen"] = "fe_test_fallback"
    else:
        df["fecha_ref_registro_dt"] = pd.NaT
        df["fecha_ref_registro_origen"] = "sin_fecha_ref"

    if "fe_test" in df.columns:
        df["fe_test_dt"] = parse_date(df["fe_test"])
    else:
        df["fe_test_dt"] = pd.NaT

    if "fe_crea_reg" in df.columns:
        df["fe_crea_reg_dt"] = parse_date(df["fe_crea_reg"])
        df["dias_desde_crea_reg"] = (df["fecha_ref_registro_dt"] - df["fe_crea_reg_dt"]).dt.days
        df.loc[df["dias_desde_crea_reg"] < 0, "dias_desde_crea_reg"] = np.nan
    else:
        df["fe_crea_reg_dt"] = pd.NaT
        df["dias_desde_crea_reg"] = np.nan

    df["registro_nuevo_30d"] = np.where(df["dias_desde_crea_reg"].le(30), "SI", "NO")
    df["registro_nuevo_90d"] = np.where(df["dias_desde_crea_reg"].le(90), "SI", "NO")

    df["bucket_dias_crea_reg"] = pd.cut(
        df["dias_desde_crea_reg"],
        bins=[-1, 7, 30, 90, 365, 730, np.inf],
        labels=[
            "00_0_7d",
            "01_8_30d",
            "02_31_90d",
            "03_91_365d",
            "04_1_2_anios",
            "05_mas_2_anios"
        ]
    )

    if "fe_reactivacion" in df.columns:
        df["fe_reactivacion_dt"] = parse_date(df["fe_reactivacion"])
        df["dias_desde_reactivacion"] = (df["fecha_ref_registro_dt"] - df["fe_reactivacion_dt"]).dt.days
        df.loc[df["dias_desde_reactivacion"] < 0, "dias_desde_reactivacion"] = np.nan
        df["tiene_reactivacion"] = np.where(df["fe_reactivacion_dt"].notna(), "SI", "NO")
    else:
        df["fe_reactivacion_dt"] = pd.NaT
        df["dias_desde_reactivacion"] = np.nan
        df["tiene_reactivacion"] = "NO"

    if "con_web" not in df.columns:
        if "website" in df.columns:
            w = safe_str_series(df["website"])
            df["con_web"] = np.where(w.notna(), "SI", "NO")
        else:
            df["con_web"] = "DESCONOCIDO"

    if "excliente_cat" not in df.columns:
        if "excliente" in df.columns:
            e = clean_bool_01(df["excliente"])
            df["excliente_cat"] = np.where(e.eq(1), "EXCLIENTE", "NO_EXCLIENTE")
            df.loc[e.isna(), "excliente_cat"] = "DESCONOCIDO"
        else:
            df["excliente_cat"] = "DESCONOCIDO"

    if "sin_intentos_recientes" not in df.columns:
        if "intentos_ult6m" in df.columns:
            intentos = clean_numeric(df["intentos_ult6m"])
            df["sin_intentos_recientes"] = np.where(intentos.fillna(0).eq(0), "SI", "NO")
        else:
            df["sin_intentos_recientes"] = "DESCONOCIDO"

    if "algun_contacto" not in df.columns:
        if "n_contactado_ult6m" in df.columns:
            n_cont = clean_numeric(df["n_contactado_ult6m"])
            df["algun_contacto"] = np.where(n_cont.fillna(0).gt(0), "SI", "NO")
        else:
            df["algun_contacto"] = "DESCONOCIDO"

    if "origen_sc_o_no" not in df.columns:
        if "origen" in df.columns:
            origen = safe_str_series(df["origen"]).str.upper()
            df["origen_sc_o_no"] = np.where(origen.str.contains("SC", na=False), "SC", "NO_SC")
            df.loc[origen.isna(), "origen_sc_o_no"] = "DESCONOCIDO"
        else:
            df["origen_sc_o_no"] = "DESCONOCIDO"

    if "ant_empresa" not in df.columns:
        if "fe_creacion_empresa" in df.columns:
            year = clean_numeric(df["fe_creacion_empresa"])
            ref_year = df["fecha_ref_registro_dt"].dt.year
            edad = ref_year - year
            df["ant_empresa"] = pd.cut(
                edad,
                bins=[-np.inf, 0, 2, 5, 10, np.inf],
                labels=["00_mismo_anio", "01_1_2_anios", "02_3_5_anios", "03_6_10_anios", "04_mas_10_anios"]
            ).astype(object)
            df.loc[year.isna(), "ant_empresa"] = "DESCONOCIDO"
        else:
            df["ant_empresa"] = "DESCONOCIDO"

    if "sin_gmb" not in df.columns:
        if "claim_business" in df.columns:
            cb = clean_bool_01(df["claim_business"])
            df["sin_gmb"] = np.where(cb.eq(1), "NO", "SI")
            df.loc[cb.isna(), "sin_gmb"] = "DESCONOCIDO"
        else:
            df["sin_gmb"] = "DESCONOCIDO"

    if "gmb_sin_owner" not in df.columns:
        if "claim_business" in df.columns:
            cb = clean_bool_01(df["claim_business"])
            df["gmb_sin_owner"] = np.where(cb.eq(0), "SI", "NO")
            df.loc[cb.isna(), "gmb_sin_owner"] = "DESCONOCIDO"
        else:
            df["gmb_sin_owner"] = "DESCONOCIDO"

    for c in [
        "movil", "ct_merclie", "con_local", "cat_contact", "con_web",
        "excliente_cat", "origen_sc_o_no", "sin_gmb", "gmb_sin_owner",
        "sin_intentos_recientes", "ant_empresa", "algun_contacto",
        "registro_nuevo_30d", "registro_nuevo_90d", "bucket_dias_crea_reg",
        "tiene_reactivacion", "origen", "ct_sociedad", "co_actvad"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(object)

    return df
