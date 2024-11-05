#linerregression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("glucose.csv")
x=data['Age X']
y=data['Glucose Y']
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x*y)
sum_x2 = np.sum(x**2)
print('No: ',len(x))
print('Sum of X:',sum_x)
print('Sum of Y:',sum_y)
print('Sum of XY:',sum_xy)
print('Sum of X sq:',sum_x2)
b0=((sum_y*sum_x2)-(sum_x*sum_xy))/((n*sum_x2)-(sum_x**2))
print('b0:',b0)
b1=((n*sum_xy)-(sum_x*sum_y))/((n*sum_x2)-(sum_x**2))
print('b1:',b1)
y_pred=b0+b1*x
y_pred
plt.scatter(x,y,color='green',label='Actual')
plt.plot(x,y_pred, color='blue',label='predicted')




#multipleregression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
data = {'X1': [1, 2, 3, 4, 5],'X2': [2, 3, 4, 5, 6],'Y': [3, 5, 7, 9, 11]}
df = pd.DataFrame(data)
X = df[['X1', 'X2']]
Y = df['Y'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Model Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
plt.scatter(X_test['X1'], Y_test, color='red', label='Actual')
plt.scatter(X_test['X1'], Y_pred, color='blue', label='Predicted')
plt.plot(X_test['X1'], Y_pred, color='green', label='Regression Line')
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('Multiple Linear Regression')
plt.legend()
plt.show()



#logisticregression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = {'X1': [1, 2, 3, 4, 5],'X2': [2, 4, 6, 8, 10],'Y': [0, 0, 0, 1, 1]}
df = pd.DataFrame(data)
X = df[['X1', 'X2']]
Y = df['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)
Y_pred = log_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")
print(f"Model Coefficients: {log_model.coef_}")
print(f"Intercept: {log_model.intercept_}")
X_test['Probabilities'] = log_model.predict_proba(X_test)[:, 1]
plt.scatter(X_test['X1'], Y_test, color='red', label='Actual')
plt.scatter(X_test['X1'], X_test['Probabilities'], color='blue', label='Predicted Probabilities')
plt.xlabel('X1')
plt.ylabel('Probability of Class 1')
plt.title('Logistic Regression')
plt.legend()
plt.show()


polynomial 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Generate sample data
# X: independent variable, y: dependent variable
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([3, 5, 7, 10, 15, 20, 30, 42, 58, 75])
# Define the degree of the polynomial
degree = 2  # You can change this to a higher number to fit a more complex curve
# Transform the features to polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)
# Fit a Linear Regression model on transformed features
model = LinearRegression()
model.fit(X_poly, y)
# Make predictions using the model
y_pred = model.predict(X_poly)
# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title(f'Polynomial Regression with Degree {degree}')
plt.show()


#gaussian
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv('creditcard.csv')
print("First 5 rows of the dataset:")
print(df.head())
print("\nBasic Information Before Dropping Null Values:")
print(df.info())
df = df.dropna()
print("\nBasic Information After Dropping Null Values:")
print(df.info())
print("\nSummary Statistics After Dropping Null Values:")
print(df.describe())
print("\nMissing Values in Each Column After Dropping Null Values:")
print(df.isnull().sum())
print("\nNumber of Unique Values in Each Column After Dropping Null Values:")
print(df.nunique())
# Separate the features (X) and the target (y)
# Replace 'target_column' with the actual name of your target column
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#Bernoulli Naive Bayes
 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import Binarizer

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Data preprocessing
df = df.dropna()

# Separate the features (X) and the target (y)
# Replace 'Class' with the actual name of your target column if different
X = df.drop(columns=['Class'])
y = df['Class']

# Binarize the data for BernoulliNB
binarizer = Binarizer()
X_binarized = binarizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_binarized, y, test_size=0.3, random_state=42)

# Initialize and train the Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("\nBernoulli Naive Bayes Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#multinomial
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Data preprocessing
df = df.dropna()

# Separate the features (X) and the target (y)
# Replace 'Class' with the actual name of your target column if different
X = df.drop(columns=['Class'])
y = df['Class']

# Ensure non-negative data for MultinomialNB by scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("\nMultinomial Naive Bayes Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))




#svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset[['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#apriori
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
basket = pd.read_csv("Groceries_dataset.csv")
display(basket.head())
basket.itemDescription = basket.itemDescription.transform(lambda x: [x])
basket = basket.groupby(['Member_number','Date']).sum()['itemDescription'].reset_index(drop=True)
encoder = TransactionEncoder()
transactions = pd.DataFrame(encoder.fit(basket).transform(basket), columns=encoder.columns_)
display(transactions.head())
frequent_itemsets = apriori(transactions, min_support= 6/len(basket), use_colnames=True, max_len = 2)
rules = association_rules(frequent_itemsets, metric="lift",  min_threshold = 1.5)
display(rules.head())
print("Rules identified: ", len(rules))
sns.set(style = "whitegrid")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection = '3d')
x = rules['support']
y = rules['confidence']
z = rules['lift']
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_zlabel("Lift")
ax.scatter(x, y, z)
ax.set_title("3D Distribution of Association Rules")
plt.show()
def draw_network(rules, rules_to_show):
  network = nx.DiGraph()
  for i in range(rules_to_show):
    network.add_nodes_from(["R"+str(i)])
    for antecedents in rules.iloc[i]['antecedents']:
        network.add_nodes_from([antecedents])
        network.add_edge(antecedents, "R"+str(i),  weight = 2)
    for consequents in rules.iloc[i]['consequents']:
        network.add_nodes_from([consequents])
        network.add_edge("R"+str(i), consequents,  weight = 2)
  color_map=[]
  for node in network:
       if re.compile("^[R]\d+$").fullmatch(node) != None:
            color_map.append('black')
       else:
            color_map.append('orange')
  pos = nx.spring_layout(network, k=16, scale=1)
  nx.draw(network, pos, node_color = color_map, font_size=8)
  for p in pos:
      pos[p][1] += 0.12
  nx.draw_networkx_labels(network, pos)
  plt.title("Network Graph for Association Rules")
  plt.show()
draw_network(rules, 10)
milk_rules = rules[rules['consequents'].astype(str).str.contains('whole milk')]
milk_rules = milk_rules.sort_values(by=['lift'],ascending = [False]).reset_index(drop = True)
display(milk_rules.head())



#KNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
df = pd.read_csv('creditcard.csv')

# Data preprocessing (drop any null values if present)
df = df.dropna()

# Separate the features (X) and the target (y)
# Replace 'Class' with the actual name of your target column
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN model (k=5 is common, but you can experiment with other values)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluation metrics
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#K-Means
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('creditcard.csv')

# Preprocessing (drop any null values if present)
df = df.dropna()

# Select the features for clustering
# Replace ['Feature1', 'Feature2'] with the names of two columns you'd like to cluster
X = df[['Feature1', 'Feature2']]

# Initialize and fit the K-Means model
# n_clusters specifies the number of clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predict the clusters
df['Cluster'] = kmeans.labels_

# Plot the clusters (only if X has 2 features)
plt.scatter(X['Feature1'], X['Feature2'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.title("K-Means Clustering")
plt.show()
This paste expires in <1 day. Public IP access. Share whatever you see with others in seconds with Context.Terms of ServiceReport this
