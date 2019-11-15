from sklearn.metrics import accuracy_score,precision_score,f1_score
import lightgbm as lgbm




lgbm=lgbm.LGBMClassifier(num_leaves=60,learning_rate=0.05,n_estimators=40)
lgbm.fit(x_train,y_train)
y_pre=lgbm.predict(x_validation)

# f1=f1_score(y_validation,y_pre,average='micro')
# print("the f1 score: %.2f"%f1)
