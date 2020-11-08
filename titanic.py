# Importing all the required libaries to analyse the data in hand
import os
import pandas as pd  
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

train_dataframe = pd.read_csv("dataset/train.csv")  #read csv file into a dataframe
test_dataframe = pd.read_csv("dataset/test.csv")   #read csv file into a dataframe

# joining two dataframes, train_dataframe and test_dataframe
full_dataframe = pd.concat(
    [
        # drop PassengerId and Survivied columns from the train_dataframe
        train_dataframe.drop(["PassengerId", "Survived"], axis=1), 
        # drop PassengerId columns from the test_dataframe
        test_dataframe.drop(["PassengerId"], axis=1),
    ]
)
# this line of code is to define y_train as just the Survived column from the train_dataframe
y_train = train_dataframe["Survived"].values
print(y_train)

# for debugging purposes
print(full_dataframe.isna().sum())
# print out value count of the Embarked column in the full_dataframe for debugging purposes
print(full_dataframe["Embarked"].value_counts())

# dimension of my plot
plt.figure(figsize=(10, 5))
# initializing the histogram to use the age data from the full_dataframe
plt.hist(full_dataframe["Age"], bins=20)
# plt title
plt.title("Age distribution")
# plotting a histogram graph for the age distribution
plt.show()

full_dataframe = full_dataframe.drop(["Cabin", "Name", "Ticket"], axis=1)
print(full_dataframe)
print(full_dataframe.isna().sum())

# Replace all NaN values in the Embarked column in the full_dataframe as "S"
full_dataframe["Embarked"].fillna("S", inplace=True)
# Replace all NaN values in the Fare column in the full_dataframe as the mean value of Fare
full_dataframe["Fare"].fillna(full_dataframe["Fare"].mean(), inplace=True)
# Replace all NaN values in the Age column in the full_dataframe as the mean value of Age
full_dataframe["Age"].fillna(full_dataframe["Age"].mean(), inplace=True)

# for debugging purposes
print(full_dataframe.isna().sum())

# Mapping the valuses of male and female to 1 and 2 respectively.
full_dataframe["Sex"] = full_dataframe["Sex"].map({"male": 1, "female": 0}).astype(int)    
# Mapping the valuses of S, C and Q to 1, 2 and 3 respectively.
full_dataframe["Embarked"] = full_dataframe["Embarked"].map({"S": 1, "C": 2, "Q": 3}).astype(int)  

X_train = full_dataframe[:y_train.shape[0]]
X_test = full_dataframe[y_train.shape[0]:]

print(f"Train X shape: {X_train.shape}")
print(f"Train y shape: {y_train.shape}")
print(f"Test X shape: {X_test.shape}")

# random prediction model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# fitting x and y axis with X_train and y_train respectively
model.fit(X_train,y_train)
# define the prediction as the random prediction model
predictions = model.predict(X_test)

# make a new dataframe for the submission
submission = pd.DataFrame({'PassengerId': test_dataframe.PassengerId, 'Survived': predictions})
# this is to make the dataframe a .csv file
submission.to_csv('submission/my_submission.csv', index = False)

print("Your submission was successfully saved!")





