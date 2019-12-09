from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

def encoder(df):
    for columnName, columnData in df.iteritems():
        try:
            pd.to_numeric(columnData)
        except ValueError:
            if columnName != "y":
                test = pd.get_dummies(df[columnName], prefix=columnName)
                df = pd.concat([df, test], axis=1)

    df.drop(["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"], 1, inplace=True)

    le = preprocessing.LabelEncoder()
    le.fit(df['y'])
    df['y'] = le.transform(df['y'])

    return df

df = pd.read_csv("bank.csv", delimiter=";")
df2 = pd.read_csv("bank-full.csv", delimiter=";")
df = encoder(df)

df2 = encoder(df2)
df = df.append(df2[df2['y']==1][:1200], sort=False)
df = df.astype(int)
print(df.head())
print(df['y'].value_counts())

X = np.array(df.drop(['y'], 1))
y = np.array(df['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=13)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)

# example_num = random.randint(0, len(y_test))
example_nums = []
aux = 0


for i, val in enumerate(y_test):
    if val == 1:
        aux += 1
        example_nums.append(i)
    if aux == 5:
        break

for example_num in example_nums:

    x_example = X_test[example_num]
    x_example = x_example.reshape(1, -1)
    y_example = y_test[example_num]

    prediction = clf.predict(x_example)

    print("The client data was: ")
    print(x_example)
    print("\n CLF prediction was %s and the actual result is %s" % (prediction[0], y_example))

print("Model accuracy: " + str(acc))
