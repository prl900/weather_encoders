import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results.txt", header=None, names=["Model", "POD 0.2","POD 0.5","POD 1","POD 2","POD 4","POD 8", "POFD 0.2","POFD 0.5","POFD 1","POFD 2","POFD 4","POFD 8", "Nada"])
df = df.drop(columns=["Nada"])

def show_barplot(data, threshold, reg_loss):
    df = pd.melt(data[["Model", "POD {}".format(threshold), "POFD {}".format(threshold)]].iloc[:5], id_vars="Model")
    df['Model'] = df['Model'].map(lambda x: u"λ={}, μ={}".format(str(x)[4], str(x)[5]))
    df['variable'] = df['variable'].map(lambda x: str(x).split(" ")[0])
    ax = sns.barplot(data=df, hue='variable', x="Model", y='value')#.set_title('lalala')
    ax.set(xlabel="Coefficients", ylabel='Probability')
    ax.set(ylim=(0, 1))
    plt.title('Model skill (threshold of detection = {})'.format(threshold))
    plt.legend(loc='lower right')
    plt.show()

show_barplot(df, "0.5", "MAE")
show_barplot(df, "1", "MAE")
show_barplot(df, "2", "MAE")
