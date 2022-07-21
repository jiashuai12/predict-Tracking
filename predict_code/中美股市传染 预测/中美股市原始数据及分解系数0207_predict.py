from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop
from keras.layers import LSTM,GRU,SimpleRNN,Bidirectional
import pylab as pl
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fftpack  
from scipy import interpolate
import cv2
import pywt
import os
import logging
from scipy.stats import pearsonr
import xlrd
import matplotlib
import tensorflow as tf

matplotlib.rcParams['font.family'] = 'SimHei'   #这两句都是为了正确显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] =False#显示负号

space = {
            'time' : hp.choice('time', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
            'pywt':hp.choice('pywt',['sym2','sym7','sym12','sym18',
                                     'coif1','coif5','coif10','coif15',
                                     'bior1.3','bior2.6','bior3.5','bior6.8',
                                     'db3','db9','db13','db18','db35','db25',
                                     'rbio1.1','rbio2.6','rbio3.5','rbio5.5','rbio4.4','rbio6.8'
                                    ]),
            'choice': hp.choice('num_time',
                    [
                    {'time': 5,
                       'weight1' :  hp.uniform('weight1', 0.9,1.0),
                       'weight2' :  hp.uniform('weight2', 0.2,0.5),
                       'weight3' :  hp.uniform('weight3', 0.2,0.5),
                       'weight4' :  hp.uniform('weight4', 0.2,0.5),
                       'weight5' :  hp.uniform('weight5', 0.2,0.3)
                      },
                     {'time': 6,
                       'weight6' :  hp.uniform('weight6', 0.9,1.0),
                       'weight7' :  hp.uniform('weight7', 0.2,0.5),
                       'weight8' :  hp.uniform('weight8', 0.2,0.5),
                       'weight9' :  hp.uniform('weight9', 0.2,0.5),
                       'weight10' :  hp.uniform('weight10', 0.2,0.5),
                       'weight11' :  hp.uniform('weight11', 0.2,0.3)
                     },
                     {'time': 7,
                       'weight12' :  hp.uniform('weight12', 0.9,1.0),
                       'weight13' :  hp.uniform('weight13', 0.2,0.5),
                       'weight14' :  hp.uniform('weight14', 0.2,0.5),
                       'weight15' :  hp.uniform('weight15', 0.2,0.5),
                       'weight16' :  hp.uniform('weight16', 0.2,0.5),
                       'weight17' :  hp.uniform('weight17', 0.2,0.5),
                       'weight18' :  hp.uniform('weight18', 0.2,0.3)
                     },
                     {'time': 8,
                       'weight19' :  hp.uniform('weight19', 0.9,1.0),
                       'weight20' :  hp.uniform('weight20', 0.2,0.5),
                       'weight21' :  hp.uniform('weight21', 0.2,0.5),
                       'weight22' :  hp.uniform('weight22', 0.2,0.5),
                       'weight23' :  hp.uniform('weight23', 0.2,0.5),
                       'weight24' :  hp.uniform('weight24', 0.2,0.5),
                       'weight25' :  hp.uniform('weight25', 0.2,0.5),
                       'weight26' :  hp.uniform('weight26', 0.2,0.3)
                     },
                     {'time': 9,
                       'weight27' :  hp.uniform('weight27', 0.9,1.0),
                       'weight28' :  hp.uniform('weight28', 0.2,0.5),
                       'weight29' :  hp.uniform('weight29', 0.2,0.5),
                       'weight30' :  hp.uniform('weight30', 0.2,0.5),
                       'weight31' :  hp.uniform('weight31', 0.2,0.5),
                       'weight32' :  hp.uniform('weight32', 0.2,0.5),
                       'weight33' :  hp.uniform('weight33', 0.2,0.5),
                       'weight34' :  hp.uniform('weight34', 0.2,0.5),
                       'weight35' :  hp.uniform('weight35', 0.2,0.3)
                     },
                         {'time': 1,
                       'weight36' :  hp.uniform('weight36', 0.2,1.0),
                      },
                        {'time': 2,
                       'weight37' :  hp.uniform('weight37', 0.2,1.0),
                       'weight38' :  hp.uniform('weight38', 0.2,0.5)
                      },
                        {'time': 3,
                       'weight39' :  hp.uniform('weight39', 0.9,1.0),
                       'weight40' :  hp.uniform('weight40', 0.2,0.5),
                       'weight41' :  hp.uniform('weight41', 0.2,0.5)

                      },
                        {'time': 4,
                       'weight42' :  hp.uniform('weight42', 0.9,1.0),
                       'weight43' :  hp.uniform('weight43', 0.2,0.5),
                       'weight44' :  hp.uniform('weight44', 0.2,0.5),
                       'weight45' :  hp.uniform('weight45', 0.2,0.5)
                      }
                    ]),
            'units1': hp.choice('units1', [48,36,24]),
            'dropout1': hp.uniform('dropout1', .0,.5),
            'batch_size' : hp.choice('batch_size', [10,15,20]),
            'nb_epochs' :  hp.choice('nb_epochs', [100,150,200,250,300,350,400,450,500]),
            'optimizer': hp.choice('optimizer',['adadelta','adam'])
        }


def f_nn(params):
    print ('Params testing: ', params)
    log.info('********************************************')
    log.info(' Params={}'.format(params))
    #总函数，分解，去噪，重组
    def wavelet_noising(new_df):
        data_1 = new_df   #这里传入的是原数据集的数据
        data_1 = data_1.values.T.tolist()  # 将np.ndarray()转为列表
        w = pywt.Wavelet(params['pywt'])  #定义小波对象
        # 调用小波分解函数，得到分解之后的近似值与细节的列表：coeffs
        coeffs = pywt.wavedec(data_1, w , level = params['choice']['time'])
        # print(params['choice']['time'])
        #保存地址
        path_ca = './data_decomposition_1/ca'
        path_cd = './data_decomposition_1/cd'
        data_path_ca = os.listdir(path_ca)
        for i in range(len(data_path_ca)):
            path_one_data = path_ca+"/"+data_path_ca[i]
            os.remove(path_one_data)
        data_path_cd = os.listdir(path_cd)
        for i in range(len(data_path_cd)):
            path_one_data = path_cd+"/"+data_path_cd[i]
            os.remove(path_one_data)
        #存低频细节
        ca = coeffs[0].reshape(-1)
        ca = pd.DataFrame(ca)
        ca.to_csv('./data_decomposition_1/ca/{}.csv'.format(len(coeffs)-2),index = False)
        #存高频细节
        length = len(coeffs) - 1
        for i in range(length):
            coeffs[i+1] = coeffs[i+1].reshape(-1)
            pd.DataFrame(coeffs[i+1]).to_csv('./data_decomposition_1/cd/{}.csv'.format(length-i-1),index=None)
        #usecoffs 重构列表
        data_path_ca = os.listdir(path_ca)
        for i in range(len(data_path_ca)):
            path_one_data = path_ca +"/"+data_path_ca[i]
            ca = pd.read_csv(path_one_data)
            ca = np.array(ca).reshape(1,len(ca))

        usecoeffs = []
        usecoeffs.append(ca)
        data_path_cd = os.listdir(path_cd)
        data_path_cd.sort(key=lambda x:int(x.split('.')[0]))
        #sort是进行降序排列
        #items是一个列表，里面是字典counts.items()输出的（键，值），那语句的作用相当于对items里面的键值对进行排序，
        # 并且是以“值”为关键字进行排序，最终输出倒序。
        path_one_data = path_cd +"/"+data_path_cd[0]
        cd1 = pd.read_csv(path_one_data)
        cd1 = np.array(cd1)
        length1 = len(cd1)
        Cd1 = np.array(cd1)
        abs_cd = np.abs(Cd1)
        median_cd = np.median(abs_cd)
        sigma = (1.0 / 0.6745) * median_cd
        lamda1 = sigma * math.sqrt(2.0 * math.log(float(length1), math.e))

        for i in range(len(data_path_cd)):
            path_one_data = path_cd +"/"+data_path_cd[len(data_path_cd)-(i+1)]
            cd = pd.read_csv(path_one_data)
            cd = np.array(cd)
            length1 = len(cd)
            if params['choice']['time'] == 5:
                a= params['choice']['weight{}'.format(i+1)]
            if params['choice']['time'] == 6:
                a = params['choice']['weight{}'.format(i+6)]
            if params['choice']['time'] == 7:
                a = params['choice']['weight{}'.format(i+12)]
            if params['choice']['time'] == 8:
                a = params['choice']['weight{}'.format(i+19)]
            if params['choice']['time'] == 9:
                a = params['choice']['weight{}'.format(i+27)]
            if params['choice']['time'] == 1:
                a = params['choice']['weight{}'.format(i+36)]
            if params['choice']['time'] == 2:
                a = params['choice']['weight{}'.format(i+37)]
            if params['choice']['time'] == 3:
                a = params['choice']['weight{}'.format(i+39)]
            if params['choice']['time'] == 4:
                a = params['choice']['weight{}'.format(i+42)]
            #软阈值去噪
            for k in range(length1):
                if (abs(cd[k]) >= lamda1):
                    cd[k] = np.sign(cd[k]) * (abs(cd[k]) - a * lamda1)
                else:
                    cd[k] = 0.0
            cd = cd.reshape(1,length1)
            usecoeffs.append(cd)
        recoeffs = pywt.waverec(usecoeffs, w)
        #recoffs是重构之后的信号
        recoeffs = recoeffs.reshape(-1)
        return recoeffs
# #蓝色低迷传染的读取
#     all_data =  pd.read_excel('./中美股市原始数据及分解系数0207.xlsx')
#     data = all_data.iloc[:,1:2]    #蓝色低迷传染的读取
#     # data = all_data.iloc[:,7:8]   #黄色低迷独立的数据
    data = pd.read_csv('./new_PM2.5process.csv')
    data1_denoising = wavelet_noising(data[:2400])
    # data1_denoising = wavelet_noising(data[:])
    data1 = np.array(data[:])
    data1 = data1.reshape(-1)
    print(data1[-1])
    data1_denoising = np.array(data1_denoising)
    data1_denoising = data1_denoising.reshape(-1)

    #自己定义小波分解去噪函数
    def waveletdec(s, wname=params['pywt'], level=params['time'], mode='symmetric'):
        N = len(s)
        w = pywt.Wavelet(wname)
        a = s
        ca = []
        cd = []
        for i in range(level):
            (a, d) = pywt.dwt(a, w, mode)  # 将a作为输入进行dwt分解
            ca.append(a)
            cd.append(d)
        rec_a = []
        rec_d = []
        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
        return rec_a,rec_d

    xiaobo_data_a , xiaobo_data_d = waveletdec(data1_denoising[:])
    xiaobo_data_a = np.array(xiaobo_data_a)
    xiaobo_data_d = np.array(xiaobo_data_d)
    xiaobo_data_a = xiaobo_data_a[len(xiaobo_data_a)-1]
    path = './data_decomposition'
    data_path = os.listdir(path)
    for i in range(len(data_path)):
        path_one_data = path+"/"+data_path[i]
        os.remove(path_one_data)
    xiaobo_data_a = pd.DataFrame(xiaobo_data_a)
    xiaobo_data_a.to_csv('./data_decomposition/小波_data_a_{}.csv'.format(len(xiaobo_data_d)-1),index = False)
    length = len(xiaobo_data_d)
    decomposition_add = np.zeros(shape=np.shape(xiaobo_data_d[0, :]))
    linewidth = 1
    for i in range(len(xiaobo_data_d)):
        decomposition_add = decomposition_add + xiaobo_data_d[i, :]
        pd.DataFrame(xiaobo_data_d[i, :]).to_csv('./data_decomposition/小波_data_d_{}.csv'.format(i),index=None)


    path = './data_decomposition'
    data_path = os.listdir(path)
    #os.listdir用于返回指定的文件夹包含的文件或文件夹的名字的列表
    data = np.array(data)
    s = data[:]
    real = s[-3*288:]
    real = real.reshape(len(real),1)
    predict = []
    weilai=[]
    for i in range(len(data_path)):
        path_one_data = path+"/"+data_path[i]
        s1 = pd.read_csv(path_one_data).values
        s1 = np.array(s1)
        s1 = s1.reshape(4350,1)
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        s1 = scaler1.fit_transform(s1)
#########多方法向前预测，找平均值。进行不确定性量化#######
        ###############50天预测50天，最后输出未来50天的数据################
        # s1 = s1.reshape(87,50,1)
        # X = s1[:70,:,:]
        # y = s1[1:71,:,0]
        # X_val = s1[71:87,:,:]
        # y_val = s1[72:88,:,0]
        # x_weilai=s1[86:87,:,:]
        ###############三个步长################
        s1 = s1.reshape(1450,3,1)
        X = s1[:1160,:,:]
        y = s1[1:1161,:,0]
        X_val = s1[1161:1449,:,:]
        y_val = s1[1162:1450,:,0]
        x_weilai = s1[1449:1450, :, :]
        ############五个步长###########################
        # s1 = s1.reshape(870,5,1)
        # X = s1[:696,:,:]
        # y = s1[1:697,:,0]
        # X_val = s1[697:869,:,:]
        # y_val = s1[698:870,:,0]
        # x_weilai = s1[869:870, :, :]

        print('x_val的最后一行是什么？{}'.format(X_val[-1]))
        print('x_val的格式是：{}'.format(X_val.shape))
        print('x_weilai的格式是：{}'.format(x_weilai.shape))
        model = Sequential()
        model.add(GRU(output_dim=params['units1'], input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
        model.add(Dropout(params['dropout1']))
        model.add(GRU(3))
        model.compile(loss='mae', optimizer=params['optimizer'])

        history = model.fit(X, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 2)

        predict_value = model.predict(X_val)
        predict_value = scaler1.inverse_transform(predict_value)  # 反归一化
        predict_value = predict_value.reshape(-1,1)
        predict.append(predict_value)

        ##三步五步作循环预测，得到未来一个月的预测数据。
        #x_weilai是输入GRU的数据,格式为[1,5,1]，输出30天的数据
        fu=[]
        for i in range(3):
            x_pred=model.predict(x_weilai)
            x_weilai=x_pred
            fu.append(x_weilai)
            x_weilai = x_weilai.reshape(1, 3, 1)
        fu=np.array(fu).reshape(-1,1)
        weilai_value = scaler1.inverse_transform(fu)#反归一化
        weilai_value = weilai_value.reshape(-1, 1)
        weilai.append(weilai_value)
        ################50对50的一次预测###########################
        # weilai_value = model.predict(x_weilai)
        # weilai_value = scaler1.inverse_transform(weilai_value)  # 反归一化
        # weilai_value = weilai_value.reshape(-1,1)
        # weilai.append(weilai_value)


    predict = np.array(predict)
    predict = np.sum(predict, axis=0).reshape(-1,1)
    pred_auc =model.predict(X_val, batch_size = 10, verbose = 0)

    weilai=np.array(weilai)
    weilai = np.sum(weilai, axis=0).reshape(-1, 1)
    real = real.reshape((-1,))
    predict = predict.reshape((-1,))
    acc = math.sqrt(mean_squared_error(real, predict))

    #验证集的准确度可视化
    plt.title ('验证集预测值与真实值对比图')
    plt.plot(real,color='blue',label='真实值')
    plt.plot(predict,color='green',label='预测值')
    plt.legend()
    plt.show()

    #未来数据的显示
    plt.title('未来9天预测数据')
    plt.plot(weilai,color='green',label='预测值')
    plt.legend()
    plt.show()

    print('AUC(acc):', acc)
    RMSE = math.sqrt(mean_squared_error(real, predict))
    NRMSE=((1/(max(predict)-min(predict)))*(math.sqrt(mean_squared_error(real, predict))))
    MAE=(mean_absolute_error(real, predict))
    def smape(y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    SMAPE=(smape(real, predict))
    R=(pearsonr(real, predict)[0])

    predict = pd.DataFrame(predict)
    predict.to_csv('./hyperopt3.0结果/predict({}).csv'.format(RMSE),index = False)
    weilai = pd.DataFrame(weilai)
    weilai.to_csv('./weilai_data/weilai({}).csv'.format(NRMSE), index=False)

    log.info('       RMSE={}'.format(RMSE))
    log.info('       NRMSE={}'.format(NRMSE))
    log.info('       MAE={}'.format(MAE))
    log.info('       SMAPE={}'.format(SMAPE))
    log.info('       R={}'.format(R))
    log.info('''''''''''''''''')
    print('其他指标:')
    print('NRMSE: ',NRMSE)
    print('MAE: ',MAE)
    print('SMAPE: ',SMAPE)
    print('R: ',R)

    sys.stdout.flush()
    return {'loss': acc, 'status': STATUS_OK}


log = logging.getLogger("main")
log.setLevel(logging.DEBUG)
os.remove("hyperopt3.0.log")
create_log = logging.FileHandler("hyperopt3.0.log")
create_log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
create_log.setFormatter(formatter)
log.addHandler(create_log)

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=20, trials=trials)
print('best: ')
print(best)
log.info(' best={}'.format(best))

