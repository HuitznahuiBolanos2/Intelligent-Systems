import pandas as pd;
import numpy as np;

class Mushroom:
    mushrooms = {};
    map = {};
    total = 0;
    columns = [];

    def __init__(self):
        self.readCSV();
        entropy = self.calculateEntropy(self.total, self.mushrooms);
        self.nextChoice([], self.mushrooms, entropy, self.map);

    def readCSV(self):
        # load dataset
        self.mushrooms = pd.read_csv("mushrooms.csv", header=0);
        self.total = len(self.mushrooms);
        for i in range(1,len(self.mushrooms.columns)):
            self.columns.append(self.mushrooms.columns[i])

    def calculateEntropy(self, newTotal, object):
        yes_prob = len(object[object['poisonous'] == "yes"])/newTotal;
        no_prob = len(object[object['poisonous'] == "no"])/newTotal;
        if yes_prob == 0 or no_prob == 0:
            return 0;
        entropy = -((yes_prob * np.log2(yes_prob)) + (no_prob * np.log2(no_prob)));
        return entropy;

    def nextChoice(self, passed, obj, entropy, dict):
        gains = self.calculateGains(passed,obj,entropy);
        choice = self.greaterGain(gains);
        passed.append(choice);
        dict[choice] = {};
        for var in obj[choice].unique():
            dict[choice][var] = {};
            newObj = obj[obj[choice] == var];
            newTotal = len(newObj);
            newEntropy = self.calculateEntropy(newTotal, newObj);
            if newEntropy != 0:
                newPassed = [];
                for i in passed:
                    newPassed.append(i);
                self.nextChoice(newPassed, newObj, newEntropy, dict[choice][var]);
            else:
                yes_prob = len(newObj[newObj['poisonous'] == "yes"])/newTotal;
                if yes_prob != 0:
                    dict[choice][var] = True;
                else:
                    dict[choice][var] = False;

    def calculateGains(self, passed, object, entropy):
        response = {};
        for elem in self.columns:
            if passed.count(elem) <= 0:
                gain = entropy;
                for var in object[elem].unique():
                    newObject = object[object[elem] == var];
                    newTotal = len(newObject);
                    newEntropy = self.calculateEntropy(newTotal, newObject);
                    gain -= ((newTotal / len(object)) * newEntropy);
                response[elem] = gain;
        return response

    def greaterGain(self, gains):
        column = '';
        max = 0;
        for elem in gains:
            if gains[elem] > max:
                column = elem;
                max = gains[elem];
        return column;

    def evaluate(self,object, column, newMap):
        temp = newMap[column][object[column]];
        if isinstance(temp, dict):
            pass;
            key = list(temp.keys())[0];
            return self.evaluate(object, key, temp);
        else:
            return newMap[column][object[column]];


m1 = Mushroom();
print(m1.evaluate({'odor': 'n', 'spore-print-color': 'n'}, 'odor', m1.map));
