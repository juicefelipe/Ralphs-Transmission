#Model with LogisticRegression and RandomForestClassifier for baseline accuracy metrics
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter = 200)
logreg.fit(X_train_scaled, y_train)
logreg_preds = logreg.predict(X_val_scaled)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_val, logreg_preds))
print(classification_report(y_val, logreg_preds))

#Not bad right out of the box. Let's convince ourselves of this score by cross validation
from sklearn.model_selection import cross_val_score
scaler_full = StandardScaler()
X_temp_scaled = scaler_full.fit_transform(X_temp)
X_test_scaled = scaler_full.transform(X_test)
logreg1 = LogisticRegression(max_iter = 200)
cross_val_score(logreg1, X_temp_scaled, y_temp).mean()

##Cross Validating shows similar accuracy. Check.

#For comparison we also build a RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_scaled, y_train)
rf_clf_preds = rf_clf.predict(X_val_scaled)
print(confusion_matrix(y_val, rf_clf_preds))
print(classification_report(y_val, rf_clf_preds))

##Excellent performance, but let's again convince ourselves of the result
rf_clf1 = RandomForestClassifier()
cross_val_score(rf_clf1, X_temp_scaled, y_temp).mean()

#Random Forest outperformed Logistic Regression by nearly 10%. Will Random Forest extend well to the test set?
rf_clf_final = RandomForestClassifier()
rf_clf_final.fit(X_temp_scaled, y_temp)
final_preds = rf_clf_final.predict(X_test_scaled)
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds))

#Given the nearly perfect accuracy on this data set we will not tune the random forest model further. However, the second project
#objective is to know the customers most likely to buy. Thus we will aslo use the predict_proba method to output purchase likelihood.
final_proba = rf_clf_final.predict_proba(X_test_scaled)
