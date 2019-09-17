import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results.txt", header=None, names=["Model", "POD 0.2","POD 0.5","POD 1","POD 2","POD 4","POD 8", "POFD 0.2","POFD 0.5","POFD 1","POFD 2","POFD 4","POFD 8", "MAE", "MSE"])
#df = df.drop(columns=["Nada"])

def show_barplot(data, threshold, reg_loss):
    df = pd.melt(data[["Model", "POD {}".format(threshold), "POFD {}".format(threshold), reg_loss]].iloc[:5], id_vars="Model")
    print(df)
    df['Model'] = df['Model'].map(lambda x: u"λ={}, μ={}".format(str(x)[4], str(x)[5]))
    df['variable'] = df['variable'].map(lambda x: str(x).split(" ")[0])

    df_bars = df.iloc[:10]
    df_line = df.iloc[10:]

    print(df_bars)
    print(df_line)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    sns.barplot(data=df_bars, hue='variable', x="Model", y='value', palette="pastel", ax=ax)
    ax.set(xlabel="Coefficients", ylabel='Probability')
    ax.set(ylim=(0, 1))
    
    sns.lineplot(data=df_line, hue='variable', x="Model", y='value', palette="hls", ax=ax2)
    ax2.set(ylabel=reg_loss)
    plt.legend(loc=4)

    plt.title('Model skill (threshold of detection = {})'.format(threshold))
    plt.show()

show_barplot(df, "0.5", "MAE")
show_barplot(df, "1", "MAE")
show_barplot(df, "2", "MAE")
