import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from common_pr_venta_8jun26 import clean_numeric, limpiar_id

CSV_RESULTADOS = 'todo_con_resultados_17.csv'
CSV_DESCUELGUE = 'todo_con_prob_descuelgue.csv'
OUT_CSV = 'todo_con_resultados_prob_final_venta_precontacto_future.csv'
KEYS = ['co_cliente', 'id_opp']

print('Leyendo CSVs...')
df_venta = pd.read_csv(CSV_RESULTADOS, low_memory=False)
df_contacto = pd.read_csv(CSV_DESCUELGUE, low_memory=False)
print('\nFilas venta:', len(df_venta))
print('Filas contacto:', len(df_contacto))

for d in [df_venta, df_contacto]:
    for k in KEYS:
        if k in d.columns:
            d[k] = limpiar_id(d[k])

required_venta = KEYS + ['prob_venta_modelo', 'prob_venta_modelo_raw']
required_contacto = KEYS + ['prob_descuelgue_modelo', 'prob_descuelgue_modelo_raw']

faltan_v = [c for c in required_venta if c not in df_venta.columns]
faltan_c = [c for c in required_contacto if c not in df_contacto.columns]
if faltan_v:
    raise ValueError(f'Faltan columnas en venta: {faltan_v}')
if faltan_c:
    raise ValueError(f'Faltan columnas en contacto: {faltan_c}')

# El nuevo script de venta ya predice para todo el universo.
df_contacto_keep = df_contacto[required_contacto].drop_duplicates(KEYS)
df = df_venta.merge(df_contacto_keep, on=KEYS, how='left')

print('\nFilas tras merge:', len(df))
print('Match prob_descuelgue_modelo:', df['prob_descuelgue_modelo'].notna().mean())

for c in ['prob_venta_modelo', 'prob_venta_modelo_raw', 'prob_descuelgue_modelo', 'prob_descuelgue_modelo_raw', 'ganada']:
    if c in df.columns:
        df[c] = clean_numeric(df[c])

df['prob_final_venta_precontacto'] = df['prob_descuelgue_modelo'] * df['prob_venta_modelo']
df['prob_final_venta_precontacto_raw'] = df['prob_descuelgue_modelo_raw'] * df['prob_venta_modelo_raw']

# Deciles sobre probabilidad final corregida
df['decil_prob_final'] = np.nan
df_valid = df[df['prob_final_venta_precontacto'].notna()].copy().sort_values('prob_final_venta_precontacto', ascending=False)
df_valid['decil_prob_final'] = pd.qcut(np.arange(len(df_valid)), 10, labels=False, duplicates='drop') + 1
df.loc[df_valid.index, 'decil_prob_final'] = df_valid['decil_prob_final']

print('\n--- CHECK PROBABILIDADES ---')
print('N prob venta:', df['prob_venta_modelo'].notna().sum())
print('N prob contacto:', df['prob_descuelgue_modelo'].notna().sum())
print('N prob final:', df['prob_final_venta_precontacto'].notna().sum())
print('Media prob venta:', df['prob_venta_modelo'].mean())
print('Media prob contacto:', df['prob_descuelgue_modelo'].mean())
print('Media prob final:', df['prob_final_venta_precontacto'].mean())

if 'ganada' in df.columns:
    df_metricas = df[df['ganada'].isin([0, 1]) & df['prob_final_venta_precontacto'].notna()].copy()
    if len(df_metricas) > 0:
        tasa_global = df_metricas['ganada'].mean()
        tabla = df_metricas.groupby('decil_prob_final')['ganada'].agg(count='count', tasa_venta='mean', ventas='sum')
        tabla['lift_vs_media'] = tabla['tasa_venta'] / tasa_global
        tabla['ventas_acum'] = tabla['ventas'].cumsum()
        tabla['pct_ventas_acum'] = tabla['ventas_acum'] / tabla['ventas'].sum()
        tabla['pct_poblacion_acum'] = tabla['count'].cumsum() / tabla['count'].sum()
        print('\nLIFT POR DECILES')
        print(tabla)
        print('\n--- MÉTRICAS FINAL PRECONTACTO ---')
        print(f'Media real ganada: {df_metricas["ganada"].mean():.6f}')
        print(f'Media predicha:    {df_metricas["prob_final_venta_precontacto"].mean():.6f}')
        if df_metricas['ganada'].nunique() == 2:
            print('AUC ROC:', roc_auc_score(df_metricas['ganada'], df_metricas['prob_final_venta_precontacto']))
            print('AUPRC:', average_precision_score(df_metricas['ganada'], df_metricas['prob_final_venta_precontacto']))
        tabla.to_csv('lift_deciles_prob_final.csv')

df.to_csv(OUT_CSV, index=False)
print('\nCSV generado:', OUT_CSV)
