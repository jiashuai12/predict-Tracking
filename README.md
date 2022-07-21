# Table of Contents

1. Background
2. Data set
    *  air_pollution.csv
    *  PM2.5.csv
    *  SH600519.csv
    *  大棚气象数据.csv
    *  中美股市原始数据及分解系数0207.xlsx
    *  GPS曲线测量.csv + GPS曲线真值.csv
    *  Kaggle_2019年纽约市的房源活动和指标.csv
3. Predict_code
    *  PM2.5 predict  
    *  SH600519 predict  
    *  air_pollution predict  
    *  中美股市传染 predict  
    *  大棚气象数据 predict  
    *  GPS曲线 Path tracking  
    *  Kaggle Data Analysis  
    
## Bachground
项目代码主要针对：
*  Time series predicted + Path tracking + Data Analysis  
本项目汇总个人实践经验，共享数据集及相关算法，希望与大家共同探讨
目前包括常用的GRU、LSTM及自己设计的小波阈值去噪、贝叶斯超参数优化、可逆选择归一化层等
后续将不断补充更多、更先进的网络


## data set
*  共包含 7 组公开数据集(空气污染、PM2.5、股票、股市影响、大棚气象、GPS仿真、Kaggle数据分析)
*  air_pollution.csv：2010/1/2-2014/12/31日空气污染数据及引起污染的相关因素，共43801条数据。采样间隔1h。
*  PM2.5.csv：北京某地PM2.5监测数据，采样间隔1h，即24个数据为一天/一周期。共8785条数据
*  SH600519.csv：SH600519股票2010/4/26-2020/4/24每天开盘、收盘、最高价、最低价和成交量数据
*  大棚气象数据.csv：山东寿光温室大棚室内数据2020/8/1-2021/1/19，采样间隔30min
*  中美股市原始数据及分解系数0207.xlsx：中美股市数据和分解系数，用于研究中美股市之间的影响
*  GPS曲线测量.csv + GPS曲线真值.csv：GPS路径仿真数据，测量表示真值+粉红噪声
*  Kaggle_2019年纽约市的房源活动和指标.csv：Kaggle数据分析

## Predict_code
*  经典GRU/LSTM 的空气污染预测
*  PM2.5预测：使用小波阈值去噪+贝叶斯超参数优化+GRU进行预测
*  经典LSTM 预测股票波动
*  基于可逆自动选择归一化深度网络的(RASN)大棚气象数据预测 (https://www.mdpi.com/2073-4395/12/3/591) 
*  小波阈值去噪+贝叶斯优化+正则化GRU进行中美股市传染预测分析
*  基于可逆选择归一化编解码器模型（RSNC）的粉红噪声GPS仿真路径跟踪
*  数据分析+数据挖掘
