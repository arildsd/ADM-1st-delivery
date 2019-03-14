import numpy as np
import pandas

def to_categorical(value, s1, s2):
    if value < s1:
        return 0
    elif value < s2:
        return 1
    else:
        return 2


def pre_process():
    # Read the file
    df = pandas.read_csv(r"../data/StudentsPerformance.csv")

    # Get an average of all the tests of a student
    df["average_score"] = df.apply(lambda x: (x["writing score"] + x["reading score"] + x["math score"])/3, axis=1)

    average_score = df["average_score"].copy()
    average_score = np.array(average_score)
    average_score.sort()
    s1 = average_score[int(0.33333 * len(average_score))]
    s2 = average_score[int(0.66666 * len(average_score))]



    # Convert average score into numerical categories. 0: Bad, 1: Average, 2: Good
    df["student performance"] = df.apply(lambda x: to_categorical(x["average_score"], s1, s2), axis=1)

    # One hot encode genders
    df["gender"] = pandas.Categorical(df["gender"])
    dfDummies = pandas.get_dummies(df['gender'], prefix='is_gender', )
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode races
    df["race/ethnicity"] = pandas.Categorical(df["race/ethnicity"])
    dfDummies = pandas.get_dummies(df["race/ethnicity"], prefix="is_race")
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode parental level of education
    df["parental level of education"] = pandas.Categorical(df["parental level of education"])
    dfDummies = pandas.get_dummies(df['parental level of education'], prefix='is_parent_education')
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode lunch
    df["lunch"] = pandas.Categorical(df["lunch"])
    dfDummies = pandas.get_dummies(df['lunch'], prefix='is_lunch')
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode test preparation course
    df["test preparation course"] = pandas.Categorical(df["test preparation course"])
    dfDummies = pandas.get_dummies(df['test preparation course'], prefix='is_prepared')
    df = pandas.concat([df, dfDummies], axis=1)



    return df

if __name__ == '__main__':
    pre_process()