# import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# drop time
data = data.drop(columns=['time'])

# set labels
data = data.sort_values(by='DEATH_EVENT', ascending=True)
train = pd.concat([data.head(80), data.tail(80)])
for_graph = train
test = data[80:-80]
also_for_graph = test

# split into test and train
y_train = train['DEATH_EVENT']
train = train.drop(columns='DEATH_EVENT')
X_train = train

y_test = test['DEATH_EVENT']
test = test.drop(columns='DEATH_EVENT')
X_test = test

# create model
model = KNeighborsClassifier(n_neighbors=80)

# train model
model.fit(X_train, y_train)

# predict
predict = model.predict(X_test)

# validation
print('Accuracy: ', metrics.accuracy_score(y_test, predict))

# accuracy check
df = pd.DataFrame(columns=['neighbors', 'accuracy'])

for i in range(80):
    i = i + 1
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    validation = metrics.accuracy_score(y_test, prediction)
    df.loc[i] = [i] + [validation]

plt.bar(df['neighbors'], df['accuracy'])
plt.show()

# # scatter plot
# labels = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT']
# for i in labels:
#     for j in labels:
#         plt.scatter(for_graph[i][:80], for_graph[j][:80], color='c')
#         plt.scatter(for_graph[i][80:], for_graph[j][80:], color='lightpink')
#         plt.xlabel(i)
#         plt.ylabel(j)
#         plt.title('Heart Failure')
#         plt.show()
