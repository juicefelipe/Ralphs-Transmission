#Using get dummies we will encode categorical data as integers
df_drop_dummy = pd.get_dummies(df_drop, drop_first = True)

#Split the data into training/validation/testing sets
from sklearn.model_selection import train_test_split
X = df_drop_dummy.drop('Buy', axis = 1)
y = df_drop_dummy['Buy']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25)

#Check the differences in scale between columns
X_train.var()

#Due to large differences in scale we scale each column to mean 0 variance 1 with StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
