#%%
import pandas as pd
from sklearn import tree

df = pd.read_parquet('Data/dados_clones.parquet')
df.head

#%%
print(df.columns)

# %%
features = ['Massa(em kilos)', 'Estatura(cm)', 'Tempo de existência(em meses)', 
            'Distância Ombro a ombro', 'Tamanho dos pés', 'Tamanho do crânio', ]
target = 'Status '

x = df[features]
y = df[target]


x = x.replace({
    "Tipo 1":1 , "Tipo 2":2, "Tipo 3":3, "Tipo 4":4, "Tipo 5":5,
    "Tipo 1":1 , "Tipo 2":2, "Tipo 3":3, "Tipo 4":4, "Tipo 5":5,
    "Tipo 1":1 , "Tipo 2":2, "Tipo 3":3, "Tipo 4":4, "Tipo 5":5,
})


# %%
model = tree.DecisionTreeClassifier(
    max_depth= 5,
    random_state=42
)
model.fit(X=x, y=y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=1000)

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True
               )
# %%
