from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv("lip_landmarks_dataset.csv")
X = dataset.drop("label", axis=1)
Y = dataset["label"]


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) 
# split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# train the model
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

# save the trained model
joblib.dump(rfc, 'model.pkl')


#dl version of this is coming soon..