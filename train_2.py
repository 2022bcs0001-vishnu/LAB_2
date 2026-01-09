import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

#dirs
os.makedirs("outputs/model",exist_ok=True)
os.makedirs("outputs/results",exist_ok=True)

#load data
df=pd.read_csv("dataset/winequality-red.csv",sep=";")
df.columns=df.columns.str.strip()

res=[]

#common feature sets
x_all=df.drop("quality",axis=1)
x_sel=df[["alcohol","volatile acidity","sulphates","density"]]
y=df["quality"]

#EXP-01 Linear Regression default
xtr,xts,ytr,yts=train_test_split(x_all,y,test_size=0.2,random_state=42)
m=LinearRegression()
m.fit(xtr,ytr)
yp=m.predict(xts)
res.append({
    "id":"EXP-01",
    "model":"LinearRegression",
    "params":"default",
    "preprocessing":"none",
    "features":"all",
    "split":"80/20",
    "mse":mean_squared_error(yts,yp),
    "r2":r2_score(yts,yp)
})

#EXP-02 Ridge + scaling
sc=StandardScaler()
xtr_s=sc.fit_transform(xtr)
xts_s=sc.transform(xts)
m=Ridge(alpha=1.0)
m.fit(xtr_s,ytr)
yp=m.predict(xts_s)
res.append({
    "id":"EXP-02",
    "model":"Ridge",
    "params":"alpha=1.0",
    "preprocessing":"standardscaler",
    "features":"all",
    "split":"80/20",
    "mse":mean_squared_error(yts,yp),
    "r2":r2_score(yts,yp)
})

#EXP-03 Random Forest depth=10
xtr,xts,ytr,yts=train_test_split(x_all,y,test_size=0.2,random_state=42)
m=RandomForestRegressor(n_estimators=50,max_depth=10,random_state=42)
m.fit(xtr,ytr)
yp=m.predict(xts)
res.append({
    "id":"EXP-03",
    "model":"RandomForest",
    "params":"trees=50,depth=10",
    "preprocessing":"none",
    "features":"all",
    "split":"80/20",
    "mse":mean_squared_error(yts,yp),
    "r2":r2_score(yts,yp)
})

#EXP-04 Random Forest depth=15 + selected features
xtr,xts,ytr,yts=train_test_split(x_sel,y,test_size=0.2,random_state=42)
m=RandomForestRegressor(n_estimators=100,max_depth=15,random_state=42)
m.fit(xtr,ytr)
yp=m.predict(xts)
res.append({
    "id":"EXP-04",
    "model":"RandomForest",
    "params":"trees=100,depth=15",
    "preprocessing":"none",
    "features":"selected",
    "split":"80/20",
    "mse":mean_squared_error(yts,yp),
    "r2":r2_score(yts,yp)
})

#EXP-05 Lasso + scaling
xtr,xts,ytr,yts=train_test_split(x_all,y,test_size=0.2,random_state=42)
sc=StandardScaler()
xtr_s=sc.fit_transform(xtr)
xts_s=sc.transform(xts)
m=Lasso(alpha=0.1)
m.fit(xtr_s,ytr)
yp=m.predict(xts_s)
res.append({
    "id":"EXP-05",
    "model":"Lasso",
    "params":"alpha=0.1",
    "preprocessing":"standardscaler",
    "features":"all",
    "split":"80/20",
    "mse":mean_squared_error(yts,yp),
    "r2":r2_score(yts,yp)
})

#save last model
joblib.dump(m,"outputs/model/model.joblib")

#save results
with open("outputs/results/result.json","w") as f:
    json.dump(res,f,indent=2)

#print summary
for r in res:
    print(r["id"],r["model"],"mse:",r["mse"],"r2:",r["r2"])
