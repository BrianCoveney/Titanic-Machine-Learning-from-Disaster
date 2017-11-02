from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import Imputer


# Function will run each of the classifiers using cross fold validation
def runClassifiers(feature_train, label_train, feature_test):
    nearestN = KNeighborsClassifier()
    nearestN.fit(feature_train, label_train)
    results = nearestN.predict(feature_test)

    return results


def removeFeatures(df_tr, df_ts):
    df_train = df_tr.drop(['Name'], axis=1).drop(['Cabin'], axis=1).drop(['Ticket'], axis=1)
    df_test = df_ts.drop(['Name'], axis=1).drop(['Cabin'], axis=1).drop(['Ticket'], axis=1)

    df_train.dropna(subset=['Embarked'], inplace=True)
    imputeEmptyValues(df_train, 'Age')
    imputeEmptyValues(df_test, 'Age')
    imputeEmptyValues(df_train, 'Fare')
    imputeEmptyValues(df_test, 'Fare')

    encodeCategoricalVariables(df_train)
    encodeCategoricalVariables(df_test)

    return df_train, df_test


def imputeEmptyValues(df, feature):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(df[[feature]])
    df[feature] = imputer.transform(df[[feature]])
    df_copy = df.isnull().sum()
    return df_copy


def encodeCategoricalVariables(df):
    map_for_sex = {'male': 0, 'female': 1}
    df['Sex'] = df['Sex'].map(map_for_sex)
    map_embarked = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(map_embarked)


def main():
    titanic_train = pd.read_csv('res/train.csv', delimiter=",")
    titanic_test = pd.read_csv('res/test.csv', delimiter=",")

    dataframes = removeFeatures(titanic_train, titanic_test)
    feature_train = dataframes[0]
    feature_test = dataframes[1]

    # Split the training dataset into features and classes
    label_train = feature_train["Survived"]
    feature_train = feature_train.drop(["Survived"], axis=1)

    # Remove the passenger ID from training dataframe
    feature_train = feature_train.drop(['PassengerId'], axis=1)

    # Remove passenger ID from test data and store as a Series object
    passengerIDSeries = feature_test["PassengerId"]
    feature_test = feature_test.drop(['PassengerId'], axis=1)

    results = runClassifiers(feature_train, label_train, feature_test)

    resultSeries = pd.Series(data=results, name='Survived', dtype='int64')

    df = pd.DataFrame({"PassengerId": passengerIDSeries, "Survived": resultSeries})

    df.to_csv("kaggle_submission.csv", index=False, header=True)


if __name__ == '__main__':
    main()
