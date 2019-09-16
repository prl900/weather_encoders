import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results.txt", header=None, names=["Model", "POD 0.2","POD 0.5","POD 1","POD 2","POD 4","POD 8", "POFD 0.2","POFD 0.5","POFD 1","POFD 2","POFD 4","POFD 8", "Nada"])
df = df.drop(columns=["Nada"])

df = pd.melt(df[["Model", "POD 1", "POFD 1"]].iloc[:5], id_vars="Model")
print(df)
sns.barplot(data=df, hue='variable', x="Model", y='value')
#sns.barplot(data=df_pofd[["POFD 1"]].T)
#df_pod.iloc[0:2].plot.bar(rot=0, subplots=True)
plt.show()
exit()
print(df)

df_pod = df[["POD 0.2","POD 0.5","POD 1","POD 2","POD 4","POD 8"]]
df_pofd = df[["POFD 0.2","POFD 0.5","POFD 1","POFD 2","POFD 4","POFD 8"]]
print(df_pod[["POD 1"]])

sns.barplot(data=df[["POD 1", "POFD 1"]].iloc[:5].T)
#sns.barplot(data=df_pofd[["POFD 1"]].T)
#df_pod.iloc[0:2].plot.bar(rot=0, subplots=True)
plt.show()
