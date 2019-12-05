from sklearn.metrics import accuracy_score,precision_score,f1_score
import lightgbm as lgbm
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


G_train_origin="E:/code/AI/dota2/data/train.csv"
G_trainx="E:/code/AI/dota2/data/train_data.csv"
G_trainy="E:/code/AI/dota2/data/train_win.csv"
G_test_origin="E:/code/AI/dota2/data/test.csv"
G_test_origin_fan="E:/code/AI/dota2/data/test_全_反.csv"
G_testx="E:/code/AI/dota2/data/test_data.csv"
G_testy="E:/code/AI/dota2/data/test_win.csv"
G_testy_fan="E:/code/AI/dota2/data/test_win_反.csv"
G_check_origin="E:/code/AI/dota2/data/check.csv"

def get_data(pos1,pos2):
    data=pd.read_csv(pos1)
    x_train=data.iloc[:,:]
    x_train=np.array(x_train)

    data=pd.read_csv(pos2)
    y_train=data.iloc[1:,]
    y_train=np.array(y_train)
    y_train=y_train.ravel()

    print(x_train.shape,y_train.shape)
    return x_train,y_train


def dispose_input(input):
    train_test = np.zeros(260)
    pos=0
    for hero in input:
        pos+=1
        if(pos<5):
            train_test[hero]=1
        else:
            train_test[hero+130]=1
    train_test=train_test.reshape(1,260)
    return train_test

x_train,y_train=get_data(G_trainx,G_trainy)
tlgbm=lgbm.LGBMClassifier(num_leaves=260,learning_rate=0.05,n_estimators=40)
tlgbm.fit(x_train,y_train)

while True:
    preinput=input("请输入测试数据")
    prelist=preinput.split(',')
    prelist=[ int(x) for x in prelist ]
    x_validation=dispose_input(prelist)
    y_pre=tlgbm.predict(x_validation)
    print(y_pre)
# f1=f1_score(y_validation,y_pre,average='micro')
# print("the f1 score: %.2f"%f1)
