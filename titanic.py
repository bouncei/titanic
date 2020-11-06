import pandas as pd  
import numpy as np  #Numerical python
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv("dataset/train.csv")  #read csv file into a dataframe
test_data = pd.read_csv("dataset/test.csv")   #read csv file into a dataframe


women = train_data.loc[train_data.Sex == 'female']["Survived"]  #filter out females that survived
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)   #the percentage of women who survived


# print(len(women))
# print(sum(women))



men = train_data.loc[train_data.Sex == 'male']["Survived"]  #filters out males that survived
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men) #the percentage of men who survived
print(rate_men, rate_women)

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
print(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission/my_submission.csv', index = False)
print("Your submission was successfully saved!")


