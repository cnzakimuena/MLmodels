""" 
This code implements several machine learning models for classifiation. The Iris dataset is used 
for demonstration. The code reads the dataset, splits it into training and validation sets, 
trains multiple models, evaluates them using K-fold cross-validation, and makes predictions 
on the validation set while reporting selected performance metrics.
"""
import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# --- read data ---
EXAMPLE_DATA_PATH = r'.\Iris.csv'
example_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(EXAMPLE_DATA_PATH, names=example_names, header=0)

# --- training and validation dataset split ---
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
VALIDATION_SIZE = 0.20
seed = 7
X_TRAIN, X_VALIDATION, Y_TRAIN, Y_VALIDATION = \
	model_selection.train_test_split(X, Y, test_size=VALIDATION_SIZE, random_state=seed)

# --- models setup ---
models = [
	('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
	('LDA', LinearDiscriminantAnalysis()),
	('KNN', KNeighborsClassifier()),
	('CART', DecisionTreeClassifier()),
	('NB', GaussianNB()),
	('SVM', SVC(gamma='auto'))
]

# --- K-fold cross-validation ---
seed = 7
SCORING = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
for name, model in models:
    cv_results = \
        model_selection.cross_val_score(model, X_TRAIN, Y_TRAIN, cv=kfold, scoring=SCORING)
    msg = f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})"
    print(msg)

# --- predictions on validation dataset ---
for name, model in models:
    print(f"=== {name} Evaluation ===")
    model.fit(X_TRAIN, Y_TRAIN)
    predictions = model.predict(X_VALIDATION)
    print("Accuracy:", accuracy_score(Y_VALIDATION, predictions))
    print("Confusion Matrix:\n", confusion_matrix(Y_VALIDATION, predictions))
    print("Classification Report:\n", classification_report(Y_VALIDATION, predictions))
    print("-" * 50)
