import numpy as np
import pandas as pd

df = pd.read_csv("results_mse.txt", header=None, names=["Model", "POD 0.2","POD 0.5","POD 1","POD 2","POD 4","POD 8", "POFD 0.2","POFD 0.5","POFD 1","POFD 2","POFD 4","POFD 8", "MAE", "MSE"])
df['Model'] = df['Model'].map(lambda x: "lambda={},mu={}".format(str(x)[4], str(x)[5]))
df = df.drop(columns=["POD 0.2","POD 0.5","POD 2","POD 4","POD 8", "POFD 0.2","POFD 0.5","POFD 2","POFD 4","POFD 8", "MAE"])
df = df[["Model", "MSE", "POD 1", "POFD 1"]]
print(df)
print(df.to_latex(index=False))
