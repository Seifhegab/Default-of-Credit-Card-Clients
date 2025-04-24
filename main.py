import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import Data_Exploration
import Data_Cleaning
import AI_Models

data = pd.read_csv("UCI_Credit_Card.csv")
data = data.rename(columns={'default.payment.next.month': 'def_pay',
                            'PAY_0': 'PAY_1'})

print(data.sample(10))
print("-----------------------------------------------------------------------------------------------")
print(data.info())
print("-----------------------------------------------------------------------------------------------")
print(data[['SEX', 'EDUCATION', 'MARRIAGE']].describe())
print("-----------------------------------------------------------------------------------------------")
print(data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe())
print("-----------------------------------------------------------------------------------------------")
print(data[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe())
print("-----------------------------------------------------------------------------------------------")
print(data[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].describe())
print("-----------------------------------------------------------------------------------------------")
print(data.LIMIT_BAL.describe())
print("-----------------------------------------------------------------------------------------------")
print(data['def_pay'].value_counts())
print("-----------------------------------------------------------------------------------------------")

# Data Exploration
# explore = Data_Exploration.data_exploration(data)
#
# explore.Catg_plotting()
#
# pay = data.iloc[:, 6:12]
# explore.Num_Plotting(pay, 2, 3, 10)
#
# bill_amt = data.iloc[:, 12:18]
# explore.Num_Plotting(bill_amt, 2, 3, 10)
#
# pay_amt = data.iloc[:, 18:24]
# explore.Num_Plotting(pay_amt, 2, 3, 10)
#
# explore.Plotting()
#
# explore.corr(pay.columns)
# explore.corr(bill_amt.columns)
# explore.corr(pay_amt.columns)

# Data Cleaning
x = data.iloc[:, 1:24]
y = data['def_pay']

# for imbalancimg data
smote = SMOTE()
x, y = smote.fit_resample(x, y)

data_clean = Data_Cleaning.data_cleaning()

x = data_clean.fix_marriage_education_pay_data(x)
x = data_clean.encoding(x)
x = data_clean.scaling(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


models = AI_Models.Models(x_train, x_test, y_train, y_test)
models.applied_models()


