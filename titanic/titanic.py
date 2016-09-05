# pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )


test_df = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

titanic_df = train_df.copy().drop(["Survived"], axis=1)
titanic_df = titanic_df.append(test_df, ignore_index=True)

def get_titles():
    global titanic_df

    # we extract the title from each name
    titanic_df['Title'] = titanic_df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    titanic_df['Title'] = titanic_df.Title.map(Title_Dictionary)

get_titles()


def process_age():
    global titanic_df

    # a function that fills the missing values of the Age variable

    def fillAges(row):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

    titanic_df.Age = titanic_df.apply(lambda r: fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

process_age()


def process_names():
    global titanic_df
    # we clean the Name variable
    titanic_df.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(titanic_df['Title'], prefix='Title')
    titanic_df = pd.concat([titanic_df, titles_dummies], axis=1)

    # removing the title variable
    titanic_df.drop('Title', axis=1, inplace=True)

process_names()
