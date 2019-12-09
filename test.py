from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

def encoder(df):
    for columnName, columnData in df.iteritems(): # Paso por cada columna
        try:
            pd.to_numeric(columnData) # Si puedo convertir la columna a numero quiere decir que es un int, entonces lo dejo así
        except ValueError:
            if columnName != "y": # Si tira value error y no es la columna de labels
                test = pd.get_dummies(df[columnName], prefix=columnName) # Uso función que transforma a onehot encoding (googlear)
                df = pd.concat([df, test], axis=1) # Appendeo las nuevas columnas al df

    df.drop(["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"], 1, inplace=True) # Elimino todas las columnas que ya no son onehot

    le = preprocessing.LabelEncoder()
    le.fit(df['y'])
    df['y'] = le.transform(df['y']) # Transformo los labels con labelEncoder a 0 o 1

    return df

df = pd.read_csv("bank.csv", delimiter=";") # Uso de general un dataset acortado pq sino tarda mucho
df2 = pd.read_csv("bank-full.csv", delimiter=";")
df = encoder(df) # Encodeo los features que son strings
df2 = encoder(df2)
df = df.append(df2[df2['y']==1][:1200], sort=False) # Appendeo 1200 rows con label "yes" porque sino hay mucha diferencia
df = df.astype(int)
print(df.head())
print(df['y'].value_counts())

X = np.array(df.drop(['y'], 1))
y = np.array(df['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Separo los train y test con un 20% para test

clf = neighbors.KNeighborsClassifier(n_neighbors=13) # Creo el clasificador (aguante python xd)
clf.fit(X_train, y_train) # Lo entreno

acc = clf.score(X_test, y_test) # Le paso el test para ver el accuracy

# <-------Usar esto para probal el modelo con un valor aleatorio-------------
# example_num.append(random.randint(0, len(y_test)))
#--------------------------------------------------------------------------/>

# Usar esto para probar el modelo con x valores de cierto label
#<-----------------------------------------------------------------
example_nums = []
aux = 0

for i, val in enumerate(y_test):
    if val == 1: # igualar a label
        aux += 1
        example_nums.append(i)
    if aux == 5: # igualar a x valor
        break
#------------------------------------------------------------------/>

for example_num in example_nums:

    x_example = X_test[example_num]
    x_example = x_example.reshape(1, -1) # Ni idea pq hay que reshepearlo así
    y_example = y_test[example_num]

    prediction = clf.predict(x_example)

    print("The client data was: ")
    print(x_example)
    print("\n CLF prediction was %s and the actual result is %s" % (prediction[0], y_example))

print("Model accuracy: " + str(acc))
