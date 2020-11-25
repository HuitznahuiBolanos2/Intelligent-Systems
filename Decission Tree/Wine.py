import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

columns = [];
total = 0;
wine = {};
x_train = {}
y_train = {}
x_test = {}
y_test = {}

def main():
    readCSV();
    splitData();
    print("Raw Tree");
    createTreeClasifier('Wine_raw.png');
    print("Raw Forest");
    randomForest();
    transformData();
    print("Transformed Tree");
    createTreeClasifier('Wine_transformed.png');
    print("Transformed Forest")
    randomForest();

def readCSV():
    global wine, total, columns;
    wine = pd.read_csv("wine.csv", header=0);
    for i in range(0,len(wine.columns)):
        columns.append(wine.columns[i])
    columns.remove('alcohol');


def splitData():
    global wine, total, columns, x_train, x_test, y_train, y_test;

    x = wine[columns];
    y = wine['alcohol'];

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


def createTreeClasifier(imageName):
    global clf, columns, x_train, y_train, x_test, y_test;
    clf = tree.DecisionTreeClassifier();
    clf.fit(x_train, y_train);

    y_pred = clf.predict(x_test);

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (48,48), dpi=300)
    tree.plot_tree(clf,
                    feature_names = columns,
                    max_depth=20,
                    fontsize=10,
                    filled = True);
    fig.savefig(imageName);
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (12,12), dpi=300)
    tree.plot_tree(clf,
                    feature_names = columns,
                    max_depth=4,
                    fontsize=10,
                    filled = True);
    fig.savefig("(2)"+imageName);

def transformData():
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

def randomForest():
    rnd_clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=16, n_jobs=-1, max_depth=5)
    rnd_clf.fit(x_train, y_train)
    y_pred_rf = rnd_clf.predict(x_test)

    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf))


main();
