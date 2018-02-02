import pandas as pd
import Levenshtein as ln
import os
from itertools import combinations#组合不重复, permutations#排列重复

def primarykey(lst):
    exists, dupl = set(), set()
    for item in lst:
        if item is not None:
            if item in exists:
                dupl.add(item)
                # print(item)
                return False
                # print(dupl)
            else:
                exists.add(item)
        else:
            return False
    return True#主键判断


def foreighkey(lst):
    a, b, c = [], [], []
    a = lst[0].split(',')
    b = lst[1].split(',')
    for i in range(0, len(a)):
        summm = a[i] + b[i]
        c.append(summm)

    return c#外键判断


def analysis(file, file1):
    table_file, table_file1, title, title1 = [], [], [], []
    table = pd.read_csv(file)
    index_colum = table.columns.size
    file_content = pd.read_csv(file, names=range(0, index_colum))
    j = 1
    while j < index_colum:
        line = list(file_content[j])
        # print(line)
        if primarykey(line) is True:
            # print(key(line))
            table_file.append(line)
            line_name = '表' + file[11:] + '第' + str(j) + '列'
            title.append(line_name)
        else:
            print('此表没有主键')
        j = j + 1
    table2 = pd.read_csv(file1,)
    index_colum1 = table2.columns.size
    file_content1 = pd.read_csv(file1, names=range(0, index_colum1))
    p = 1
    while p < index_colum1:
        line1 = list(file_content1[p])
        if primarykey(line1) is True:
            # print(key(line1))
            table_file1.append(line1)
            line_name1 = '表' + file1[11:] + '第' + str(p) + '列'
            title1.append(line_name1)
        else:
            print('此表没有主键')
        p = p + 1
    if len(table_file) & len(table_file1) == 1:
        for colum, t in zip(table_file, title):
            # print('table_file:' + str(t) +'是主键'+ '如下所示'+ str(colum))
            for colum1, t1 in zip(table_file1, title1):
                # print('table_file:' +str(t1)+'是主键'+ '如下所示'+str(colum1))
                correct = ln.ratio(str(colum), str(colum1))
                print('table_file:' + str(t) + 'to table_file1:' + str(t1) + 'similarity is :' + str(
                    '%.2f%%' % (correct * 100)))

                while correct == 1:
                    print('table_file:' + str(t) + ' is :' + 'to table_file1:' + str(t1) + '外键')
                else:
                    print('table_file:' + str(t) + '是主键' + '如下所示' + str(colum))
                    print('table_file:' + str(t1) + '是主键' + '如下所示' + str(colum1))
                    break
    else:

            t = foreighkey(table_file)
            t1 = foreighkey(table_file1)
            w = foreighkey(title)
            w1 = foreighkey(title1)
            print()
            for colum, t in zip(t, w):
                # print('table_file:' + str(t) +'是主键'+ '如下所示'+ str(colum))
                for colum1, t1 in zip(t1, w1):
                    # print('table_file:' +str(t1)+'是主键'+ '如下所示'+str(colum1))
                    correct = ln.ratio(str(colum), str(colum1))
                    print('table_file:' + str(t) + 'to table_file1:' + str(t1) + 'similarity is :' + str(
                        '%.2f%%' % (correct * 100)))

                    while correct == 1:
                        print('table_file:' + str(t) + ' is :' + 'to table_file1:' + str(t1) + '外键')
                    else:
                        print('table_file:' + str(t) + '是主键' + '如下所示' + str(colum))
                        print('table_file:' + str(t1) + '是主键' + '如下所示' + str(colum1))
                        break
    return ""
    #两个表主外键分析
def traversal(dirname,dir_paths):
    dir_path = []
    for dire in os.listdir(dirname):
        dir_path.append(os.path.join(dirname, dire))
    dir_paths = dir_path
    return dir_paths
    #文件名字遍历
#file_table = 'd:\correct\emp'
#file_table1 = 'd:\correct\location'
#a = analysis(file_table,file_table1)
dir_name_list,dir_name_com,result=[],[],[]
dirname =  'd:/keras/hrds'
dir_name_list =traversal(dirname,dir_name_list)
dir_name_com = list(combinations(dir_name_list,2))
for i in dir_name_com:
    dir_name1 = i[0]
    dir_name2 = i[1]
    print(analysis(dir_name1,dir_name2))
"""
for i in dir_name_com:
    dir_name1 = i[0]
    dir_name2 = i[1]
    table_file, table_file1, title, title1 = [], [], [], []
    table = pd.read_csv(dir_name1)
    index_colum = table.columns.size
    file_content = pd.read_csv(dir_name1, names=range(0, index_colum))
    j = 1
    while j < index_colum:
        line = list(file_content[j])
        # print(line)
        if primarykey(line) is True:
            # print(key(line))
            table_file.append(line)

            line_name = '表' + dir_name1[14:] + '第' + str(j) + '列'

            title.append(line_name)
        else:
            print(str(dir_name1[14:])+'此表无符合得主键')
        j = j + 1
    # print(str(title)+ '是主键'+str(table_file))
    table2 = pd.read_csv(dir_name2)
    index_colum1 = table2.columns.size
    file_content1 = pd.read_csv(dir_name2, names=range(0, index_colum1))
    p = 1
    while p < index_colum1:
        line1 = list(file_content1[p])
        if primarykey(line1) is True:
            # print(key(line1))
            table_file1.append(line1)
            line_name1 = '表' + dir_name2[14:] + '第' + str(p) + '列'
            title1.append(line_name1)
        else:
            print(str(dir_name2[14:])+'此表没有符合得主键')
        p = p + 1
    # print(str(title1)+'是主键'+str(table_file1))
    for colum, t in zip(table_file, title):
        # print('table_file:' + str(t) +'是主键'+ '如下所示'+ str(colum))
        for colum1, t1 in zip(table_file1, title1):
            # print('table_file:' +str(t1)+'是主键'+ '如下所示'+str(colum1))
            correct = ln.ratio(str(colum), str(colum1))
            print('table_file:' + t + 'to table_file1:' + t1 + 'similarity is :' + str('%.2f%%' % (correct * 100)))

            while correct == 1:
                print('table_file:' + t + ' is :' + 'to table_file1:' + t1 + '外键')
            else:
                print('table_file:' + str(t) + '是主键' + '如下所示' + str(colum))
                print('table_file:' + str(t1) + '是主键' + '如下所示' + str(colum1))
                
"""
