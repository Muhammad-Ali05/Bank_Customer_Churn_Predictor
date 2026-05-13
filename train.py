import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df = pd.read_csv("Churn_Modelling.csv")
df = df.drop(['CustomerId', 'RowNumber', 'Surname'], axis=1)

X = df.drop("Exited", axis=1)
y = df["Exited"]

categorical_features = ["Geography", "Gender"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("model", GaussianNB())
])
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

pickle.dump(nb_acc, open("nb_acc.pkl", "wb"))
dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(random_state=42))
])
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

pickle.dump(dt_acc, open("dt_acc.pkl", "wb"))

svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", probability=True))
])
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

pickle.dump(svm_acc, open("svm_acc.pkl", "wb"))

pickle.dump(nb_model, open("nb_model.pkl", "wb"))
pickle.dump(dt_model, open("dt_model.pkl", "wb"))
pickle.dump(svm_model, open("svm_model.pkl", "wb"))

print("✅ Models trained and saved successfully!")