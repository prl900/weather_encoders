import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def show_barplot(df, mode):
   
    if mode=="0x":
        df = df.iloc[[0,1,2,3]].reset_index(drop=True)
    elif mode == "x0":
        df = df.iloc[[0,4,5,6]].reset_index(drop=True)
    elif mode == "xx":
        df = df.iloc[[0,7,8,9]].reset_index(drop=True)
    else:
        print("error")
        return

    
    df['Model'] = df['Model'].map(lambda x: u"λ={}, μ={}".format(str(x)[1], str(x)[2]))
    #df['variable'] = df['variable'].map(lambda x: str(x).split(" ")[0])

    df_bars = df[["POD","POFD"]]
    df_line = df[["MSE"]]
    
    _, ax = plt.subplots()
    ax2 = ax.twinx()
    
    df_line.plot(kind='line', style='.-', ms=10, ylim=(0, 1.2), color=['g', 'g'], ax=ax)
    ax.set(ylabel="MSE")
    ax.legend(loc=2)

    df_bars.plot(kind='bar', ax=ax2, ylim=(0, 1))
    ax2.grid(False)
    ax2.set(xlabel="Coefficients", ylabel='Probability')
    ax2.set_xticklabels(df['Model'], rotation=0)
    ax2.legend(loc=1)
   
    plt.title(r'Model Skill ($\alpha$ = 1.0)')
    plt.savefig('bars_{}.png'.format(mode))
    plt.clf()

df = pd.read_csv("results_final_bars.csv")

show_barplot(df, "0x")
show_barplot(df, "x0")
show_barplot(df, "xx")
