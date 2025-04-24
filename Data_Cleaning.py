import pandas as pd
from sklearn.preprocessing import StandardScaler


class data_cleaning:

    @staticmethod
    def fix_marriage_education_pay_data(data):
        # education data
        fil = (data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)
        data.loc[fil, 'EDUCATION'] = 4

        # marriage data
        data.loc[data.MARRIAGE == 0, 'MARRIAGE'] = 3
        data.MARRIAGE.value_counts()

        # # pay data
        # fil = (data.PAY_1 == -2) | (data.PAY_1 == -1) | (data.PAY_1 == 0)
        # data.loc[fil, 'PAY_1'] = 0
        # fil = (data.PAY_2 == -2) | (data.PAY_2 == -1) | (data.PAY_2 == 0)
        # data.loc[fil, 'PAY_2'] = 0
        # fil = (data.PAY_3 == -2) | (data.PAY_3 == -1) | (data.PAY_3 == 0)
        # data.loc[fil, 'PAY_3'] = 0
        # fil = (data.PAY_4 == -2) | (data.PAY_4 == -1) | (data.PAY_4 == 0)
        # data.loc[fil, 'PAY_4'] = 0
        # fil = (data.PAY_5 == -2) | (data.PAY_5 == -1) | (data.PAY_5 == 0)
        # data.loc[fil, 'PAY_5'] = 0
        # fil = (data.PAY_6 == -2) | (data.PAY_6 == -1) | (data.PAY_6 == 0)
        # data.loc[fil, 'PAY_6'] = 0

        return data

    @staticmethod
    def encoding(data):
        data = pd.get_dummies(data, columns=['EDUCATION', 'MARRIAGE', 'SEX', 'PAY_1',
                                             'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], dtype=int)
        return data

    @staticmethod
    def scaling(data):
        scaler = StandardScaler()
        temp = scaler.fit_transform(data)
        temp = pd.DataFrame(temp, columns=data.columns)
        return temp
