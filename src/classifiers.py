from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def naive_bayes(matrix, classes):
    nb_classifier = MultinomialNB()
    y_pred = cross_val_predict(nb_classifier, matrix, classes, cv=10)
    return y_pred


def svm(matrix, classes):
    svm_classifier = SVC(kernel='linear')
    y_pred = cross_val_predict(svm_classifier, matrix, classes, cv=10)
    return y_pred

def decision_tree(matrix, classes):
    dt_classifier = DecisionTreeClassifier()
    y_pred = cross_val_predict(dt_classifier, matrix, classes, cv=10)
    return y_pred

def random_forest(matrix, classes):
    rf_classifier = RandomForestClassifier()
    y_pred = cross_val_predict(rf_classifier, matrix, classes, cv=10)
    return y_pred