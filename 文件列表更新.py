import os
import pickle
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
for i in news_ls:
    his_ls.append(i)
if len(news_ls)!=0:
    with open('his.pkl','wb') as d :
        a = pickle.dump(new_ls,d)
#print(i)




