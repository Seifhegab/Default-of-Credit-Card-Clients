import seaborn as sns
import matplotlib.pyplot as plt


class data_exploration:
    def __init__(self, data):
        self.data = data

    # Categorical Plotting
    def Catg_plotting(self):
        temp = self.data
        temp = temp.replace({"SEX": {1: "male", 2: "female"}})
        temp = temp.replace({"EDUCATION": {1: "graduate school", 2: "university", 4: "others",
                                           3: "high school", 5: "unknown", 6: "unknown", 0: "unknown"}})
        temp = temp.replace({"MARRIAGE": {1: "married", 2: "single", 3: "others",  0: "unknown"}})
        temp['MARRIAGE'].value_counts().plot(kind='bar')
        plt.show()
        temp['EDUCATION'].value_counts().plot(kind="barh")
        plt.show()
        temp['SEX'].value_counts().plot(kind="pie", title="SEX")
        plt.show()

    # Numerical plotting
    def Num_Plotting(self, variables, n_rows, n_cols, n_bins):
        fig = plt.figure()
        for i, var_name in enumerate(variables):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            self.data[var_name].hist(bins=n_bins, ax=ax)
            ax.set_title(var_name)
        fig.tight_layout()  # Improves appearance a bit.
        plt.show()

    def Plotting(self):
        temp = self.data
        temp = temp.replace({"SEX": {1: "male", 2: "female"}})
        fig, axes = plt.subplots(ncols=2, figsize=(13, 8))
        temp['SEX'].value_counts().plot(kind="pie", ax=axes[0], subplots=True)
        sns.countplot(x=temp['SEX'], hue=temp['def_pay'], data=temp)
        plt.show()
        sns.boxplot(x=temp['def_pay'], y=temp["LIMIT_BAL"], data=temp)
        plt.show()

    def corr(self, variables):
        plt.figure(figsize=(8, 8))
        corr = self.data[variables].corr()
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                    linewidths=.1, vmin=-1, vmax=1)
        plt.show()
