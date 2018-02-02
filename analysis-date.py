"""
created on wednesday,january 10
@author :Asher
"""
import pandas as pd
import Levenshtein as ln,os
from itertools import combinations, permutations
print(list(combinations([1, 2, 3], 2)))
"""
def finddupl(lst):

#找出 lst 中有重复的项
#(与重复次数无关，且与重复位置无关)

    exists, dupl = set(), set()
    for item in lst:
        if item in exists:
            dupl.add(item)
        else:
            exists.add(item)
    return dupl
.....................

lst = ['1,2,3,5,6,4,7,8,9','1,2,3,5,6,4,7,8,9']
a, b ,c= [],[],[]

a = lst[0].split(',')
b = lst[1].split(',')

for i in range(0,len(a)):
    summm=a[i]+b[i]
    c.append(summm)
    print(c)

def pk_one(lst):
    exists, dupl = set(), set()
    for item in lst:
        if item is not None:
            if item in exists:
                dupl.add(item)
                #print(item)
                return False
                #print(dupl)
            else:
                exists.add(item)
        else:
            return False
    return True
def pk_tow(lst):
    num = len(lst)
    if num ==2:
        a, b, c ,d= [], [], [],[]
        a = lst[0].split(',')
        b = lst[1].split(',')
        for i in range(0, len(a)):
            summm = a[i] + b[i]
            c.append(summm)
        d = pk_one(c)
    return d
lst = ['1,2,3,5,6,4,7,8,9','1,2,3,5,6,4,7,8,9']

print(pk_tow(lst))


dir_path = []
dirname =  'd:/keras/hrds'
for name in os.listdir(dirname):
    dir_path.append(os.path.join(dirname, name))
    print(dir_path)






file = ['1,2,3,5,6,4,7,8,9','1,2,3,5,6,4,7,8,9']
def key(lst):
    exists, dupl = set(), set()
    for item in lst:
        print(item)
        if item is not None:
            if item in exists:
                dupl.add(item)
                #print(item)
                return False
                #print(dupl)
            else:
                exists.add(item)
        else:
            return False
    return True
print(key(file))
"""
"""
联合就在于主键A跟主键B形成的联合主键是唯一的。以我来看复合主键就是含有一个以上的字段组成,
如ID + name, ID + phone等, 而联合主键要同时是两个表的主题组合起来的。
这是和复合主键最大的区别!
"""