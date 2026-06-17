import pandas as pd

df = pd.read_csv("t_pr_venta_update_8jun26.csv", low_memory=False)

df["fe_carga_efectiva"] = pd.to_datetime(
    df["fe_carga_efectiva"],
    errors="coerce"
)

print(df["fe_carga_efectiva"].max())
print(
    df["fe_carga_efectiva"]
    .dt.date
    .value_counts()
    .sort_index()
    .tail(20)
)