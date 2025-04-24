from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt


class Models:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def applied_models(self):
        models = [DecisionTreeClassifier(criterion='gini',
                                         max_depth=10),
                  RandomForestClassifier(n_estimators=500,
                                         criterion='gini',
                                         max_depth=5,
                                         n_jobs=-1),
                  GradientBoostingClassifier(learning_rate=0.01,
                                             n_estimators=500,
                                             max_depth=5),
                  LogisticRegression(C=0.01, penalty='l2')]
        list_name = ["Decision tree", "Random forest", "gradient boost", "Logistic regression"]
        i = 0
        for model in models:
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            score = f1_score(y_true=self.y_test, y_pred=y_pred)
            accuracy = accuracy_score(y_true=self.y_test, y_pred=y_pred)
            print(list_name[i])
            print("F1 Score = ", score)
            print("Accuracy = ", accuracy)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(self.y_test, y_pred)
                                                        , display_labels=[0, 1])
            cm_display.plot()
            plt.show()
            i = i+1
