import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import keras
import pymysql
import torch
import torch.nn as nn

BATCH_SIZE = 2000
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 6             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 2     # it could be total point G can draw in the canvas
reportdate = "2018-07-01"

conn = pymysql.Connect(host='192.168.101.71',port=3306,user='analyst',passwd='545b4ae90d0bf01aa79a3c0d51a7f10c',db='datacenter',charset='utf8')
cursor = conn.cursor()

sqlcount = "select count(cost) from dc_stat_rev_tag where revenue >cost and cost is not null and revenue is not null and report_date ='%s'"  %reportdate
countt= pd.read_sql(sql=sqlcount, con=conn)
#print(countt.iloc[0][0])

sqlxg = "select cost_clicks,rev_clicks/rev_impr as ctr from dc_stat_rev_tag where  cost_clicks >0 and rev_clicks <> 0 and rev_impr <> 0 and revenue>cost limit "
sqlxg += str(countt.iloc[0][0])
dfxg = pd.read_sql(sql=sqlxg, con=conn)
goodx_data = preprocessing.minmax_scale(dfxg, feature_range=(-1, 1))


sqlxdate = "select trigger_words,platform,id_adpages,id_linkpair,id_account,id_trigger_words from dc_stat_rev_tag where cost_clicks >0 and rev_clicks <> 0 and rev_impr <> 0 and revenue>cost and report_date ='%s'"  %reportdate
dfxdate = pd.read_sql(sql=sqlxdate, con=conn)
#print(dfxdate[0:10][0:10])


G = nn.Sequential(                      # Generator
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)


#File = open("data/topvalue.txt", "w+",encoding=u'utf-8', errors='ignore')


autoweight = torch.load('module/G.pkl')

        # 用新加载的模型进行预测
prediction = autoweight(x)
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.show()
#tensorboard --logdir=logs
