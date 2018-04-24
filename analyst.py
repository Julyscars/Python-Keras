#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
plt.style.use('ggplot')

data=pd.DataFrame(pd.read_csv('DataAnalyst.csv',encoding='gbk'))

#1、获取查看列名用data.columns,查看数据值用values,查看数据框的索引用index,查看描述性统计用describe,
#2、使用type看一下输出的描述性统计是什么样的数据类型——DataFrame数据;使用T来转置数据也就是行列转换;
#3、据进行排序，用到了sort(columns='c')参数可以指定根据哪一列数据进行排序
#4、假如我们要选择A列的数据进行操作：df['A'];还可以使用数组的切片操作，但是注意了，切片得到的是行数据df[1:3];
#5、我们还可以使用行标签来指定输出的行df[1:2];
#6、DataFrame的loc方法是帮助选择数据的，比如选择索引位置为0的一行数据注意我们是用dates[0]作为索引的）
#7、选择多列数据的写法df[:['A','B']]


city = data['city'].value_counts()
companyShortName= data['companyShortName'].value_counts()
education = data['education'].value_counts()
industryField=data['industryField'].value_counts()
positionName = data['positionName'].value_counts()
salary=data['salary'].value_counts()
workYear=data['workYear'].value_counts()
#数据去重复值subset为基准值,keep保留的值方式，first保留第一个,las保留最后一个.
data_drop_duplicates = data.drop_duplicates(subset='positionId',keep='first')
#清晰salary数据；针对｛7k-9k｝格式我们定义一个cutword(word,bottom)函数，查找word.find('-‘)
#取[:position-1]前值，作为bottomsalary,再取len(word)-1作为topsalary;
# 针对｛12k以上｝取word[word.upper().find('K')]作为bottomsalary
def cut_word(word,method):
    position = word.find('-')
    length = len(word)
    if position != -1:
        bottomsalary = word[:position-1]
        topsalary = word[position+1:length-1]
    else:
        bottomsalary = word[:word.upper().find('K')]
        topsalary = bottomsalary
    if  method == 'bottom':
        return bottomsalary
    else:
        return topsalary
#表格里添加topsalary和bottomsalary两列:apply将word_cut函数应用在salary列的所有行
data_drop_duplicates['topsalary']=data_drop_duplicates.salary.apply(cut_word,method = 'top')
data_drop_duplicates['bottomsalary']=data_drop_duplicates.salary.apply(cut_word,method='bottom')
#使用astype转化数据格式
data_drop_duplicates.topsalary=data_drop_duplicates.topsalary.astype('int')
data_drop_duplicates.bottomsalary=data_drop_duplicates.bottomsalary.astype('int')
#使用很多时候我们并不需要复杂地使用def定义函数，而用lamdba作为一次性函数。lambda x: ******* ，前面的lambda
#x:理解为输入，后面的星号区域则是针对输入的x进行运算。ambda x: ******* ，前面的lambda x:理解为输入，后面的星
#号区域则是针对输入的x进行运算。案例中，因为同时对top和bottom求平均值，所以需要加上
# x.bottomSalary和x.topSalary。word_cut的apply是针对Series，现在则是DataFrame。
#axis是apply中的参数，axis=1表示将函数用在行，axis=0则是列。
data_drop_duplicates['avgsalary']=data_drop_duplicates.apply(lambda x:(x.bottomsalary + x.topsalary)/2,axis=1)
#填充缺失值use: fillna('missingdatas') 这里，any([])函数如若里面一个为真，则返回True.即逐行扫描，发现‘missing’存在此行中
#或此行时间不落在0-24之间，或气压rpessure大于1500，或风向不落在0-360度之间，或风速大于10级，或precipitation降雨量大于10，则此行数据为异常数据进行删除。
for i in range(data.index.max()):  
    if any([  
        'missing' in data.loc[i,:].values,  
        data.loc[i,'hour'] not in range(25),  
        data.loc[i,'pressure']>1500,  
        data.loc[i,'wind_direction']<0 or data.loc[i,'wind_direction']>360,  
        data.loc[i,'wind_speed']>10,  
        data.loc[i,'precipitation']>10  
        ]):  
  
        print('已删除存在异常值 %s 行数据'%i)  
        data.drop([i],inplace=True)  
#保存清晰的数据
data_drop_duplicates.to_csv('clean.csv')  
#首先调用 DataFrame.isnull() 方法查看数据表中哪些为空值，与它相反的方法是 DataFrame.notnull()
#Pandas会将表中所有数据进行null计算，以True/False作为结果进行填充和 DataFrame.dropna() 两种方式，
#原来不加参数的情况下dropna() 会移除所有包含空值的行。如果只想移除全部为空值的列，需要加上 axis=1 和 how=all 两个参数
#使用不同分块大小来读取再调用 pandas.concat 连接DataFrame，chunkSize设置在1000万条左右速度优化比较明显。
"""
loop = True
chunkSize = 100000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print "Iteration is stopped."
df = pd.concat(chunks, ignore_index=True)

"""
data = pd.DataFrame(pd.read_csv('clean.csv',encoding='utf-8'))


#print(data.info())
#groupby可以传递一组列表，这时得到一组层次化的Series。按城市和学历分组计算了平均薪资再调用unstack方法，进行行列转置
#我们在groupby后面加一个avgSalary，说明只统计avgSalary的计数结果，不用混入相同数据
data.groupby(['city','education']).avgsalary.count().unstack()
#计算不同公司招聘的数据分析师数量，使用了agg函数，同时传入count和mean方法agg除了系统自带的几个函数，它也支持自定义函数。
data.groupby(['companyShortName']).avgsalary.agg(['count','mean']).sort_values(by='count',ascending=False)
data.groupby(['companyShortName']).avgsalary.agg([lambda x:max(x)-min(x)])
#我想计算出不同城市，招聘数据分析师需求前5的公司，应该如何处理？agg虽然能返回计数也能排序，但它返回的是所有结果，
#前五还需要手工计算。能不能直接返回前五结果？当然可以，这里再次请出apply。
def topN(zd,n=5):
    counts = zd.value_counts()
    return counts.sort_values(ascending=False)[:n]
data.groupby('city').companyShortName.apply(topN)
#如果我想知道不同城市，各职位招聘数前五，也能直接调用topN
data.groupby('city').positionName.apply(topN)
#plot相关分析
from matplotlib.font_manager import FontProperties
font_zh=FontProperties(fname='/home/killy/pythons/simples/Veras.ttf')
#箱型图

ax = data.boxplot(column='avgsalary',by='workYear',figsize=(9,6))
for label in ax.get_xticklabels():
    label.set_fontproperties(font_zh)
plt.show()
#柱状图
ax1 = data.groupby('city').avgsalary.mean().plot.bar(figsize=(9,6))
for label in ax1.get_xticklabels():
    label.set_fontproperties(font_zh)
plt.legend(['avgsalary'],prop=font_zh)
plt.show()

#多重聚合在作图上面没有太大差异，行列数据转置不要混淆即可。
ax2 = data.groupby(['city','education']).avgsalary.mean().unstack().plot.bar(figsize=(14,6))
for label in ax2.get_xticklabels():
    label.set_fontproperties(font_zh)
plt.legend(prop=font_zh)
plt.show(ax2)

#元素中的[]是无意义的，它是字符串的一部分，和数组没有关系。因为df_clean.positionLables是Series，
# 并不能直接套用replace。apply是一个好方法，但是比较麻烦。这里需要str方法。str方法允许我们针对列中的元素，
# 进行字符串相关的处理，这里的[1:-1]不再是DataFrame和Series的切片，而是对字符串截取，这里把[]都截取掉了。
# 如果漏了str，就变成选取Series第二行至最后一行的数据，切记。使用完str后，它返回的仍旧是Series，
# 当我们想要再次用replace去除空格。还是需要添加str的。现在的数据已经干净不少。
# pandas删除缺失数据(pd.dropna()方法)
word=data.positionLables.str[1:-1].str.replace('','')
#这里是重点，通过apply和value_counts函数统计标签数。因为各行元素已经转换成了列表，所以value_counts会逐行计算
# 列表中的标签，apply的灵活性就在于此，它将value_counts应用在行上，最后将结果组成一张新表。
# 用unstack完成行列转换，看上去有点怪，因为它是统计所有标签在各个职位的出现次数，绝大多数肯定是NaN。
# 将空值删除，并且重置为DataFrame，此时level_0为标签名，level_1为df_index的索引，也可以认为它对应着一个职位，
# 0是该标签在职位中出现的次数，之前我没有命名，所以才会显示0。部分职位的标签可能出现多次，这里忽略它。
word_drop=word.dropna().str.split(',').apply(pd.value_counts).unstack().dropna().reset_index()
words_count=word_drop.groupby(by=['level_0'])['level_0'].agg({"count":np.size})
word_count=words_count.reset_index().sort_values(by=["count"],ascending=False)
#replace("'","")去除单引号
word_count.level_0.replace("'","")
#print(word_count.head())
#agg(['count','mean']).sort_values(by='count',ascending=Fals
word_cloud = WordCloud(font_path='/home/killy/pythons/simples/Veras.ttf',
                      width=1900,height=1400,
                      background_color='white')

plt.subplot()
word_fre = {x[0]:x[1] for x in word_count.head(100).values}
print(word_count,'\n',word_fre)

word_clouds=word_cloud.fit_words(word_fre)
plt.imshow(word_clouds)
plt.show()
#print(word_fre)
