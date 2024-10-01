
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\91948\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional-full.csv', sep=';')



if 'duration' in df.columns:
    df = df.drop(columns=['duration'])


label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop(columns=["y"])  
y = df["y"]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


classifier = DecisionTreeClassifier(random_state=42, max_depth=3) 
classifier.fit(X_train, y_train)


plt.figure(figsize=(20,10))
plot_tree(classifier, feature_names=X.columns, class_names=['no', 'yes'], filled=True, rounded=True, fontsize=10)
plt.show()


y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
