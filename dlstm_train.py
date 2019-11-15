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
G_train_origin="E:/code/AI/dota2/data/train.csv"
G_trainx="E:/code/AI/dota2/data/train_data.csv"
G_trainy="E:/code/AI\dota2/data/train_win.csv"
G_test_origin="E:/code/AI/dota2/data/test.csv"
G_test_origin_fan="E:/code/AI/dota2/data/test_全_反.csv"
G_testx="E:/code/AI/dota2/data/test_data.csv"
G_testy="E:/code\AI/dota2/data/test_win.csv"
G_testy_fan="E:/code\AI/dota2/data/test_win_反.csv"
G_check_origin="E:/code/AI/dota2/data/check.csv"


#parameters for LSTM
nb_lstm_outputs = 128  #神经元个数
nb_time_steps = 10  #时间序列长度
nb_input_vector = 130 #输入序列


def lstm_model(save=False,save_road='./model/sigmoid_doublelstm_model.h5'):
    model = Sequential([
        layers.Bidirectional(layers.LSTM(units=nb_lstm_outputs,dropout=0.25, recurrent_dropout=0.1), input_shape=(nb_time_steps, nb_input_vector)),
        layers.Dropout(0.2),
        layers.Dense(10,activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
        ])
    model.compile(optimizer=keras.optimizers.Adam(),batch_size=100,
                 loss='mse',
                 metrics=['accuracy'])

    model.summary()

    x,y=get_data_onehot(G_train_origin)
    history=model.fit(x,y,
          epochs=3)
    #保存网络
    if save:
        model.save(save_road)
        print("已保存在"+save_road)
    return history,model

#获取数据并进行one-hot编码
def get_data_onehot(file_pos):
    data=pd.read_csv(file_pos)
    #提取天辉和夜魇阵容
    # radiant_train=data.iloc[1:,2:7]
    # radiant_train=np.array(radiant_train)
    # radiant_train=to_categorical(radiant_train)
    # dire_train=data.iloc[1:,7:12]
    # dire_train=np.array(dire_train)
    # dire_train=to_categorical(dire_train)

    x_data=data.iloc[1:,2:12]
    x_data=np.array(x_data)
    x_data=to_categorical(x_data,130)

    y_data=data.iloc[1:,12:13]
    y_data=np.array(y_data)
    #y_data=to_categorical(y_data,2)

    print(x_data.shape,y_data.shape)

    return x_data,y_data

#获得相对正确率
def relative(model,show):
    x_test,y_test=get_data_onehot(G_test_origin)


    six_gap=0.1
    seven_gap=0.2
    eight_gap=0.3

    six_count=0
    seven_count=0
    eight_count=0

    sidata_count=0
    sedata_count=0
    eidata_count=0
    record=0

    for i in range(len(x_test)):
        rex=x_test[i].reshape(1,10,130)
        predata=model.predict(rex)

        if predata-six_gap>0.5:
            if y_test[i]==1:
                six_count+=1
            sidata_count+=1
        elif predata+six_gap<0.5:
            if y_test[i]==0:
                six_count+=1
            sidata_count+=1

        if predata-seven_gap>0.5:
            if y_test[i]==1:
                seven_count+=1
            sedata_count+=1
        elif predata+seven_gap<0.5:
            if y_test[i]==0:
                sedata_count+=1
            seven_count+=1

        if predata-eight_gap>0.5:
            if y_test[i]==1:
                eight_count+=1
            eidata_count+=1
        elif predata+eight_gap<0.5:
            if y_test[i]==0:
                eight_count+=1
            eidata_count+=1

        record+=1
        if record==100:
            record=0
            print(i)
            if show==True:
                if sidata_count!=0:
                    print("预测胜率在0.6以上准确率:"),
                    print(six_count/sidata_count)
                if sedata_count!=0:
                    print("预测胜率在0.7以上准确率:"),
                    print(seven_count/sedata_count)
                if eidata_count!=0:
                    print("预测胜率在0.8以上准确率:"),
                    print(eight_count/eidata_count)

    return six_count/sidata_count,seven_count/sedata_count,eight_count/eidata_count



if __name__ == "__main__":
    load=input("是否加载模型 Y/N \n")
    if load=='Y' or load=='y':
        #加载训练好的模型
        network_result = tf.keras.models.load_model('./model/sigmoid_doublelstm_model.h5')
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

    #进行相对准确率预测
    get_relative=input("是否获取相对准确率 Y/N \n")
    if get_relative=='Y' or get_relative=='y':
        six,seven,eight=relative(network_result,True)


    #进行平均准确率预测
    test_check=input("获取测试集或者验证集精度 t/y \n")
    if test_check=='t':
        testx,testy=get_data_onehot(G_test_origin)
    else:
        testx,testy=get_data_onehot(G_check_origin)
    network_result.evaluate(testx,testy)
