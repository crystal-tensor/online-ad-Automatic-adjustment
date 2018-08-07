# online-ad-Automatic-adjustment2
广告自动调价算法介绍
1：Bid的自动出价算法
2：Weight的自动调优算法

1：Bid的自动出价算法介绍
1、原理：我们希望最大化revenue最小化cost，但是业务员只能调整bid，budget和weight，由于sql里面目前还无法查询出bid和revenue之间对应到trigger的对应关系，我暂时先把cpc近似当成bid，然后找cpc和profit（之所以此处没有用revenue，是因为要让revenue最大化不可能不考虑cost，如果把成本考虑进来，最大化revenue更实际的情况是提高收益率profit，另外，此处没有选择提高ctr是因为提高ctr的做法放在优化weight上了）之间的关系。见图1，从图上看深度神经网络是能找到他们之间的有效相关性，
2、数据：分别抽取cpc和盈利profit之间的对应数据，以及cpc和亏损profit之间的对应数据。
3、算法：用GAN算法分别生成盈利profit对应的bid_g和亏损profit对应的bid_b。
其中bid的生成范围在0.05-3之间。
D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))  #最大化profit函数
G_loss = -torch.mean(torch.log(1. - prob_artist1))   #最小化bid函数
4、bid自动调价规则
如果bid_g - bid_b > 0,那么当前的bid能调整的幅度范围就是：bid + bid_g - bid_b。
如果bid_g - bid_b < 0,那么当前的bid能调整的幅度范围就是：bid - bid_g - bid_b。
5、模型源码及模型部署
模型源码auto-bid.py(文件我放在192.168.101.70的autobid目录下)
模型完整运行一次耗时35个小时，导入数据62万。

模型文件autobidg.model.meta和autobidb.model.meta部署在autobid/module下，这2个模型分别耗时35小时和68小时，是所有盈利数据和所有亏损数据进去后计算出来的。下一步想把bid，budget和weight合在一起出2个模型就够了,目前是分开的，不过调用的时候其实不用管模型部署文件，都已经直接写到程序里面了。
6、自动调价的计算和更新bid步骤：
6.1：每天在收到收入报告后，在导入adm之后，运行程序autobid.py。
6.2：自动调价bid的输出分为2个文件，每个文件2列，见下表
6.3：用good的价格减去对应trigger的bad价格就是每天需要调整的bid范围，然后根据自动调价规则来更新bid价格。


2：Weight的自动调优算法
2.1：理论上我们应该是谁的转换率高我们给的权重就多，但是我们的Weight是在ad-page上统计的，每个ad-page对应多个trigger。所以在计算转换率的时候我是用revenue的click除以cost的click，然后用GAN算法不断拟合weight到转换率的关系。简单说是最大化revenue的ctr，最小化cost的click。剩下的工作就跟上面一样了，但只产生一个模型。
2.2、自动调优weight的计算和更新weight步骤：
2.2.1：每天在收到收入报告后，在导入adm之后，调整参数reportdate的日期为收到报告的日期，然后运行程序autoweight.py。
2.2.2：自动调价weight的输出只有一个文件weight**.txt（**是日期），见下表
trigger_words	platform	id_adpages	id_linkpair	id_account	id_trigger_words	Weight建议值
lowering ldl cholesterol	inuvo_lexo	1230846	302984	1324	1	15.223612
lowering ldl cholesterol	inuvo_lexo	1123574	320956	1324	1	14.490099
lowering ldl cholesterol	inuvo_lexo	1168406	333796	1324	1	2.878565
lowering ldl cholesterol	inuvo_lexo	1219862	307416	1324	1	25.285267
lowering ldl cholesterol	inuvoreal	637346	190224	1016	1	27.708143
lowering ldl cholesterol	inuvo_lexo	920846	266974	208	1	28.100786
lowering ldl cholesterol	inuvo_lexo	1232120	351384	228	1	25.049885
atopic dermatitis	inuvoreal	493436	148438	381	2	13.623442
atopic dermatitis	inuvo_lexo	1274066	365812	802	2	16.44322
atopic dermatitis	parked	1026562	294370	310	2	18.391785
atopic dermatitis	inuvoreal	1253692	221414	962	2	9.028065
atopic dermatitis	inuvoreal	1244370	165926	962	2	29.078337
atopic dermatitis	inuvoreal	1257782	185636	962	2	8.735622
atopic dermatitis	inuvoreal	1273000	365604	145	2	16.769901
atopic dermatitis	inuvoreal	1273090	365608	145	2	17.396612
atopic dermatitis	inuvoreal	1269354	364070	1104	2	8.085274
atopic dermatitis	inuvoreal	1269488	364108	1104	2	24.298153
atopic dermatitis	inuvo_lexo	1265226	354156	928	2	14.363249
atopic dermatitis	inuvo_lexo	1273738	349490	928	2	11.722599

2.2.3：如果weight的值小于0代表不建议分流量到该trigger上，另外，weight的小数点可以省去。具体在哪个页面调整，看trigger对应的id_adpages及相应的账户id_account。这还需要技术部的支持把每个trigger对应到的账户和adpage分配下去。业务员根据自动调优的weight建议值调整weight。
2.3：模型源码
自动调优的源码autoweight.py(文件我放在192.168.101.70的autobid目录下)
如果需要看模型效果把注释去掉就能看到：
# if step % 50 == 0:
#     #print('D_loss', "%.6f"%D_loss.data)
#     # print('G_loss', G_loss)
#     # print("\n")
# File.write(str("%.6f"%D_loss.data)+" ,"+str("%.6f"%G_loss.data) + "\n")
