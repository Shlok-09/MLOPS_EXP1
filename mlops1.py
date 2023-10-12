import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
df = pd.read_csv("heart.csv")
df.head()

#shape and description
print(f"Shape is: {df.shape}")
print("==================================================")
print(f"Summary statistics: {df.describe()}")

#checking the maximum correlation of the output with other variables
df.corr().abs()['output'].sort_values(ascending = False)

X = df.drop('output', axis = 1)
y = df['output']

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from tensorflow import keras

model = keras.Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=[13]
        ),
        keras.layers.Dense(515, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

early_stopping = keras.callbacks.EarlyStopping( patience = 20, min_delta = 0.001,
                                               restore_best_weights =True )
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=15,
    epochs=50,
    callbacks = [early_stopping],
    verbose=1,
)

model.evaluate(X_test, y_test)

predictions =(model.predict(X_test)>0.5).astype("int32")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

acc = accuracy_score(y_test, predictions)*100
print(acc)

if acc<85:
    raise Exception("Accuracy is less! Can't Commit!")


model.save('model.h5')
# print(classification_report(y_test, predictions))

