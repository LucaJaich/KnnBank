import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("bank.csv", delimiter=";")

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

df = encoder(df)

print(df.head().to_csv(index=False))