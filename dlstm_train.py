from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,datasets,optimizers,Sequential,metrics
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

batch_size=600
G_train_origin="E:/code/AI/dota2/data/matches_list_ranking2.csv"
G_trainx="E:/code/AI/dota2/data/train_data.csv"
G_trainy="E:/code/AI\dota2/data/train_win.csv"
G_test_origin="E:/code/AI/dota2/data/matches_list_ranking.csv"
G_test_origin_fan="E:/code/AI/dota2/data/test_全_反.csv"
G_testx="E:/code/AI/dota2/data/test_data.csv"
G_testy="E:/code\AI/dota2/data/test_win.csv"
G_testy_fan="E:/code\AI/dota2/data/test_win_反.csv"



#parameters for LSTM
nb_lstm_outputs = 128  #神经元个数
nb_time_steps = 10  #时间序列长度
nb_input_vector = 130 #输入序列


def lstm_model(save=False,save_road='./model/drop_2_doublelstm_model.h5'):
    model = Sequential([
        layers.Bidirectional(layers.LSTM(units=nb_lstm_outputs,dropout=0.25, recurrent_dropout=0.1), input_shape=(nb_time_steps, nb_input_vector)),
        # layers.Dropout(0.2),
        # layers.Dense(100,activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
        ])
    model.compile(optimizer=keras.optimizers.Adam(),batch_size=100,
                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    model.summary()

    x,y=get_data_onehot(G_train_origin)
    history=model.fit(x,y,
          epochs=1)
    #保存网络
    if save:
        model.save(save_road)
        print("已保存在"+save_road)
    return history,model


def get_data_onehot(file_pos):
    data=pd.read_csv(file_pos)
    #提取天辉和夜魇阵容
    # radiant_train=data.iloc[1:,2:7]
    # radiant_train=np.array(radiant_train)
    # radiant_train=to_categorical(radiant_train)
    # dire_train=data.iloc[1:,7:12]
    # dire_train=np.array(dire_train)
    # dire_train=to_categorical(dire_train)

    x_train=data.iloc[1:,2:12]
    x_train=np.array(x_train)
    x_train=to_categorical(x_train,130)

    y_train=data.iloc[1:,12:13]
    y_train=np.array(y_train)
    y_train=to_categorical(y_train,2)

    print(x_train.shape,y_train.shape)

    return x_train,y_train




if __name__ == "__main__":
    load=input("是否加载模型 Y/N \n")
    if load=='Y' or load=='y':
        #加载训练好的模型
        network_result = tf.keras.models.load_model('./model/doublelstm_model.h5')
    else:
        history,network_result = lstm_model(True)

    #手动输入数据进行预测
    while True:
        pre=input("是否进行数据预测 Y/N \n")
        if pre=='Y' or pre=='y':
            preinput=input("请输入测试数据")
            prelist=preinput.split(',')
            prelist=[ int(x) for x in prelist ]
            prenp=np.array(prelist)
            prenp=to_categorical(prenp,130)
            #print(prenp)
            preinput=prenp.reshape(1,10,130)
            preresult=network_result.predict(preinput)
            print(preresult)
        else:
            break

    testx,testy=get_data_onehot(G_test_origin)
    network_result.evaluate(testx,testy)
