import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus


columns = [];
total = 0;
mushrooms = {};
x_train = {}
y_train = {}
x_test = {}
y_test = {}

def main():
    readCSV();
    splitData();
    createTreeClasifier();
    # printTree();

def readCSV():
    global mushrooms, total, columns;
    # load dataset
    mushrooms = pd.read_csv("mushrooms.csv", header=0);
    for i in range(1,len(mushrooms.columns)):
        columns.append(mushrooms.columns[i])


def splitData():
    global mushrooms, total, columns, x_train, x_test, y_train, y_test;

    x = mushrooms[columns];
    y = mushrooms['poisonous'];

    # Transform data into numeric values OneHotEncoder
    for elem in x.columns:
        jobs_encoder = LabelBinarizer()
        jobs_encoder.fit(x[elem])
        transformed = jobs_encoder.transform(x[elem])
        ohe_df = pd.DataFrame(transformed)
        x = pd.concat([x, ohe_df], axis=1).drop([elem], axis=1)
    print(x);

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


def createTreeClasifier():
    global clf;
    clf = DecisionTreeClassifier();
    clf.fit(x_train, y_train);

    y_pred = clf.predict(x_test);

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    tree.plot_tree(clf,
                   filled = True);
    fig.savefig('mushrooms.png')
#
# def printTree():
#     global clf;
#     dot_data = StringIO()
#     export_graphviz(clf, out_file=dot_data,
#                     filled=True, rounded=True,
#                     special_characters=True,feature_names = feature_cols,class_names=['0','1'])
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('Mushrooms.png')
#     Image(graph.create_png())


main();
print(x_train);
