import os
import pickle
From Logger import logger 
file_path = 'd:/new/'

his_ls =[]
try:
    with open('his.pkl','rb') as f:
        old_ls = pickle.load(f)
        print(old_ls)
except:
    old_ls = []

new_ls = os.listdir(file_path)
new_ls = [file_path+ i for i in new_ls if str.endswith(i, '.txt') or str.endswith(i, 'spider')]

news_ls = set(new_ls).difference(set(old_ls))
news_ls = list(news_ls)
print(news_ls)
for x,i in enumerate(news_ls):
    #生成从0开始的序列和列表：(0,'string')
    his_ls.append(i)
    logger.info('reading file: {0}/{1}, {2}...'.format(x,len(new_ls),i))
    try:
        ############################
        
        
        ############################
    except Exception as err:
        ......
        ......
        continue
if len(new_ls) <=0:
    logger.infor('not new file end...')
eles:
    with open('his.pkl','wb') as d :
        a = pickle.dump(new_ls,d)
#print(i)




