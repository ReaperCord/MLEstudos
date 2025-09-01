#%%

import pandas as pd
from sklearn import tree

df = pd.read_excel('Data/dados_cerveja.xlsx')
df.head()

# %%
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

x = df[features]
y = df[target]

#Criando dummies variables para tranformar as informações do dataframe em numeros para sklearn
x = x.replace({
    "mud":1, "pint":2,
    "sim":1, "não":0,
    "clara":0, "escura":1
})


# %%
model = tree.DecisionTreeClassifier()
model.fit(X=x, y=y)


# %%

import matplotlib.pyplot as plt

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True
               )
# %%
