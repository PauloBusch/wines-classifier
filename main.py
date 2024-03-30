from math import ceil
from os import system
from numpy import average
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import math

# read data
data=pd.read_csv('./dataset/data.csv')
dataFrame=pd.DataFrame(data.values, columns=data.columns.values)

# plot boxplot
# top 4 best correlation characteristcs
type='Type'
correlationed_characteristics = [
  'Proanthocyanins',
  'Total phenols', 
  'Flavanoids',
  'Proline'
]
dimension=ceil(math.sqrt(len(correlationed_characteristics)))
fig, axes=plt.subplots(dimension,dimension)
fig.suptitle('Characteristics Boxplot')
for index in range(0, dimension * dimension):
  if len(correlationed_characteristics) <= index: continue
  characteristic=correlationed_characteristics[index]
  sns.boxplot(
    x=type, 
    y=characteristic,
    ax=axes.flatten()[index], 
    data=dataFrame[[type, characteristic]],
    width=0.3,
    showfliers=False
  )
  ax=axes.flatten()[index]
  ax.set_title(characteristic)
  ax.set_xlabel(None)
  ax.set_ylabel(None)

# normalize data
input_columns=[]
for column in data.columns.values:
  if (column == type): continue
  column_average=average(dataFrame[column])
  dataFrame[column]=dataFrame[column].div(column_average)
  input_columns.append(column)

# plot dispersion matrix
plot_columns=['Type'] + correlationed_characteristics
sns.pairplot(dataFrame[plot_columns], hue=type)

# split data in train and test
train_input, test_input, train_output, test_output=train_test_split(
  dataFrame[input_columns], 
  dataFrame[type], 
  test_size=0.3,
  stratify=dataFrame[type],
  random_state=58
)

# train machine learning using Random Forest 
randomForestClassifier=RandomForestClassifier(
  n_estimators=100, 
  oob_score=True, 
  random_state=58
)
randomForestClassifier.fit(train_input, train_output)

# test machine learning using Random Forest
predicted=randomForestClassifier.predict(test_input)

# plot confusion matrix
classes=set(data[type]) 
dataFrame=pd.DataFrame(
  confusion_matrix(test_output, predicted), 
  columns=classes, 
  index=classes
)
plt.subplots(1, 1)
sns.heatmap(dataFrame, annot=True)

# collect metrics
accuracy=accuracy_score(test_output, predicted)
precision=precision_score(test_output, predicted, average='macro')
recall=recall_score(test_output, predicted, average='macro')
f1=f1_score(test_output, predicted, average='macro')

system('cls')
print(f'Accuracy Score: {accuracy:.3}')
print(f'Precision Score: {precision:.3}')
print(f'Recall Score: {recall:.3}')
print(f'F1 Score: {f1:.3}')

plt.show(block=True)
print('');