#%%

import pandas as pd
from sklearn import tree

df = pd.read_excel('data/dados_frutas.xlsx')
df.head

# %%
arvore = tree.DecisionTreeClassifier(random_state=42)

caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]

y = df['Fruta']
x = df[caracteristicas]


# nessa parte é aonde o ML acontece
arvore.fit(x, y)

# %%
#saida esperada cereja
arvore.predict([[1,1,1,1]])

# %%
#Saida esperada tomate
arvore.predict([[0,1,1,0]])

# %%
#saida esperada banana
arvore.predict([[0,0,0,0]])

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True)

# %%
