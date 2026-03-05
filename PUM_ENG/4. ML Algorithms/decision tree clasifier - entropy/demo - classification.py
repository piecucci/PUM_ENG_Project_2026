import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

#read csv with tab separator
ta = '\t'
df = pd.read_csv('demo - classification.csv', sep=ta)

# Encoding the 'gender' column
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'f' else 0)

# Extracting features and target
X = df[['gender', 'age']]
y = df['app']

# Creating and training the decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X, y)

# Displaying the decision tree
tree_rules = export_text(tree_clf, feature_names=['gender', 'age'])
print(tree_rules)