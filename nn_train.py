import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets,optimizers,Sequential,metrics
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

heroes=130
G_trainx="E:/code/AI/dota2/data/train_data.csv"
G_trainy="E:/code/AI\dota2/data/train_win.csv"
G_testx="E:/code/AI/dota2/data/test_data.csv"
G_testy="E:/code\AI/dota2/data/test_win.csv"
G_testy_fan="E:/code\AI/dota2/data/test_win_反.csv"


#数据预处理(取消)
def preprocess(x,y):
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y

#加载训练集
def get_data(pos1,pos2):
    data=pd.read_csv(pos1)
    x_test=data.iloc[:,:]
    x_test=np.array(x_test)

    data=pd.read_csv(pos2)
    y_train=data.iloc[1:,]
    y_train=np.array(y_train)
    y_train = to_categorical(y_train)

    print(x_test.shape,y_train.shape)
    db = tf.data.Dataset.from_tensor_slices((x_test,y_train))
    db = db.batch(600)
    print(db)
    return db

#构建并训练网络
def train(db,save=False,save_road='./model/model.h5'):
    network = Sequential([
        layers.Dense(260,activation=tf.nn.relu),
        layers.Dense(130,activation=tf.nn.relu),
        layers.Dense(65,activation=tf.nn.relu),
        layers.Dense(32,activation=tf.nn.relu),
        layers.Dense(2,activation='softmax')
    ])
    network.build(input_shape=[None,1*260])
    network.summary()

    #训练参数
    network.compile(optimizer=optimizers.Adam(lr=1e-2),
                    loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics = ['accuracy'])

    #运行神经网络
    network.fit(db,epochs=10,validation_freq=1)

    #保存网络
    if save:
        network.save(save_road)
        print("已保存在"+save_road)

    return network


#加载测试集
def test_data(network):
    pos1=G_testx
    pos2=G_testy
    data=pd.read_csv(pos1)
    x_test=data.iloc[:50000,]
    x_test=np.array(x_test)

    data=pd.read_csv(pos2)
    y_test=data.iloc[1:50001,:]
    y_test=np.array(y_test)

    print(x_test.shape,y_test.shape)
    db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    db_test = db.batch(500)
    network.evaluate(db_test)

#预测
def predict(network,input):
    train_test = np.zeros(heroes*2)
    pos=0
    for hero in input:
        pos+=1
        if(pos<5):
            train_test[hero]=1
        else:
            train_test[hero+130]=1
    train_test=train_test.reshape(1,260)
    result=network.predict(train_test)
    # print(result)
    # print('2:')
    # print(result[0])
    #print(result[0][0])
    #print(result[0][1])
    return result

#获得相对精确率(在相差relative的情况下的精度)
def GetRelative(data1,data2,result,relative=0):
    if (data1-data2)>relative and result==0:
        return True
    elif (data2-data1)>relative and result==1:
        return True
    return False

#人工处理验证(直观)
def my_eva(network,pos1,pos2):
    data=pd.read_csv(pos1)
    x_test=data.iloc[:,:]
    x_test=np.array(x_test)

    data=pd.read_csv(pos2)
    y_test=data.iloc[1:,]
    y_test=np.array(y_test)

    right=0
    counter=0
    hundred_num=1
    for i in range(len(x_test)):
        counter+=1
        result=network.predict(x_test[i].reshape(1,260))
        if GetRelative(result[0][0],result[0][1],y_test[i]):
            #print(result)
            right+=1
        else:
            #print(result)
            pass
        if counter>100:
            print(str(hundred_num)+":"+str(right/i))
            hundred_num+=1
            counter=0

    print(right/len(x_test))




if __name__ == "__main__":


    load=input("是否加载模型 Y/N \n")
    if load=='Y' or load=='y':
        #加载训练好的模型
        network_result = tf.keras.models.load_model('./model/model10.h5')
        #test_data(network_result)
        #input()
        #用自定义评估函数进行评估
        #my_eva(network_result,G_testx,G_testy)
    else:
        #重新训练模型
        xtrain_pos=G_trainx
        ytrain_pos=G_trainy
        db=get_data(xtrain_pos,ytrain_pos)
        network_result=train(db,True)
        

    # #测试集测试
    # eva=input("是否用测试集进行数据验证 Y/N \n")
    # if eva=='Y' or eva=='y':
    #     test_data(network_result)

    #手动输入数据进行预测
    while True:
        pre=input("是否进行数据预测 Y/N \n")
        if pre=='Y' or pre=='y':
            preinput=input("请输入测试数据")
            prelist=preinput.split(',')
            prelist=[ int(x) for x in prelist ]
            result=predict(network_result,prelist)
            if result[0][0]>result[0][1]:
                print("夜魇胜利")
            else:
                print("天辉胜利")
        else:
            break

    my_eva(network_result,G_testx,G_testy)



