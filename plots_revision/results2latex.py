import numpy as np
import pandas as pd

df = pd.read_csv("results_final.csv")
df['Model'] = df['Model'].map(lambda x: "lambda={},mu={}".format(str(x)[1], str(x)[2]))
print(df.to_latex(index=False))
