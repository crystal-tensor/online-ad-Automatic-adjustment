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

BATCH_SIZE = 5000
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 6             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 2     # it could be total point G can draw in the canvas
input_size = 6


conn = pymysql.Connect(host='192.168.101.71',port=3306,user='analyst',passwd='545b4ae90d0bf01aa79a3c0d51a7f10c',db='datacenter',charset='utf8')
cursor = conn.cursor()
sqlxg = "select cost/cost_clicks,revenue-cost as profit from dc_stat_rev_tag where cost is not null and cost_clicks >0 and revenue is not null and revenue>cost limit "
sqlxg += str(BATCH_SIZE)
dfxg = pd.read_sql(sql=sqlxg, con=conn)
goodx_data = preprocessing.minmax_scale(dfxg, feature_range=(-1, 1))

sqlxb = "select cost/cost_clicks,revenue-cost as profit from dc_stat_rev_tag where cost is not null and cost_clicks >0 and revenue is not null and revenue<cost limit "
sqlxb += str(BATCH_SIZE)
dfxb = pd.read_sql(sql=sqlxb, con=conn)
badx_data = preprocessing.minmax_scale(dfxb, feature_range=(-1, 1))

sqlcount = "select count(cost) from dc_stat_rev_tag where revenue >cost and cost is not null and revenue is not null limit "
sqlcount += str(BATCH_SIZE)
countt= pd.read_sql(sql=sqlcount, con=conn)
#print(countt.iloc[0][0])

xs = tf.placeholder(tf.float32, [None, N_IDEAS]) # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])

from numpy import random as nr
np.set_printoptions(precision=2)
bid = nr.uniform(0.05, 3, size=(1, countt.iloc[0][0]))

def earn():
    y = goodx_data[:, 0:2]
    paintings = torch.from_numpy(y).float()
    return paintings

def loss():
    y = badx_data[:, 0:2]
    paintings = torch.from_numpy(y).float()
    return paintings

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

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
File = open("data/test.txt", "w",encoding=u'utf-8', errors='ignore')
#File = open("data/topvalue.txt", "w+",encoding=u'utf-8', errors='ignore')

for step in range(10000):
    artist_paintings = earn()           # real painting from artist
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
    G_paintings = G(G_ideas)                    # fake painting from G (random ideas)

    prob_artist0 = D(artist_paintings)          # D try to increase this prob
    prob_artist1 = D(G_paintings)               # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = -torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    lossd = sess.run(D_loss, {xs: G_ideas, ys: artist_paintings})
    if step % 50 == 0:
        #print('D_loss', "%.6f"%D_loss.data)
        # print('G_loss', G_loss)
        # print("\n")
        File.write(str("%.6f"%lossd) + "\n")
        print(lossd)
    # if step % 10000 == 0:
    #     #base_path = saver.save(sess, "module/bid_forcase.model")
    #     #print(G_paintings)
    #     for i in range(620000):
    #          File.write(str("%.6f"%G_paintings[i][0].data)+" , "+str("%.6f"%G_paintings[i][1].data)+"\n")


#tensorboard --logdir=logs