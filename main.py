import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

#FEATURES ENGINEER
df["Class"].value_counts(normalize=True)
df["Amount_log"] = np.log1p(df["Amount"])
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

#CONFIG IA
x = df.drop("Class", axis=1)
y = df["Class"]
x_train, x_test, y_train,y_test = train_test_split(
    x,y,stratify=y,test_size=0.3,random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#ADD DATA
smote = SMOTE()
x_res, y_res = smote.fit_resample(x,y)

#XGBOOST MODEL
model_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="aucpr",
    random_state=42
)
model_xgb.fit(x_res, y_res)
y_pred_xgb = model_xgb.predict(x_test)

print("=== XGBoost com SMOTE ===")
print(classification_report(y_test, y_pred_xgb))