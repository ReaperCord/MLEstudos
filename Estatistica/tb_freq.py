#%%
import pandas as pd
import sqlalchemy

df = pd.read_csv("../Data/Estatistica/points-tmw.csv")
df.head()

engine = sqlalchemy.create_engine("sqlite:///../Data/tmw.db")
df.to_sql("Points", engine, if_exists="replace", index=False)

# %%
freq_produto = (df.groupby(["descProduto"])[["idTransacao"]]
                  .count())

freq_produto["Freq. Abs Acum"] = freq_produto["idTransacao"].cumsum()

freq_produto["Freq. Relativa"] = freq_produto["idTransacao"] / freq_produto["idTransacao"].sum()

freq_produto["Freq. Relativa Acum"] = freq_produto["Freq. Relativa"].cumsum()

freq_produto


# %%
freq_desc_categoria_produtos = (df.groupby(["descCategoriaProduto"])[["idTransacao"]]
                                  .count()
                                  .rename(columns={"idTransacao":"Freq. Abs"}))

freq_desc_categoria_produtos["Freq. Abs Acum"] = freq_desc_categoria_produtos["Freq. Abs"].cumsum()

freq_desc_categoria_produtos["Freq. Relativa"] = freq_desc_categoria_produtos["Freq. Abs"] / freq_desc_categoria_produtos["Freq. Abs"].sum()

freq_desc_categoria_produtos["Freq. Relativa Acum"] = freq_desc_categoria_produtos["Freq. Relativa"].cumsum()

freq_desc_categoria_produtos


# %%
