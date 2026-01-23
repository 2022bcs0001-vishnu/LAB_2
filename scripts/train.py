# Generated from: train.ipynb
# Converted at: 2026-01-09T05:47:07.331Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# #CSS 426 Machine Learning Operations (MLOps)
# ## Name : Vishnu Naryanan Vinodkumar
# ## Roll Number : 2022BCS0001
# ## Date : 09-01-2026
# ## Batch : 1
# ## Lab : 2


# ### Imports


import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

# ### Dataset


url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df=pd.read_csv(url,sep=";")

# ### Pre-processing and Feature Selection


x=df.drop("quality",axis=1)
y=df["quality"]

fs=["alcohol","volatile acidity","sulphates","density"]
x=x[fs]

xtr,xts,ytr,yts=train_test_split(x,y,test_size=0.2,random_state=42)

sc=StandardScaler()
xtr=sc.fit_transform(xtr)
xts=sc.transform(xts)

# ### Training


m=RandomForestRegressor(n_estimators=200,random_state=42)
m.fit(xtr,ytr)

# ### Evaluate


yp=m.predict(xts)

mse=mean_squared_error(yts,yp)
r2=r2_score(yts,yp)

print("mse:",mse)
print("r2:",r2)

# ### Saving the outputs


joblib.dump(m,"model.joblib")

res={
    "model":"RandomForestRegressor",
    "features":fs,
    "n_estimators":200,
    "mse":mse,
    "r2":r2
}

with open("results.json","w") as f:
    json.dump(res,f,indent=2)