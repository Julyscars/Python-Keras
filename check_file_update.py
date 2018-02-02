
# coding:utf-8

import fileinput
import time
import os

target_file = 'user.log'
init_flag = True  # 初次加载程序
time_kick = 5

record_count = 0

while True:
    print( '当前读到了', record_count)
    #没有日志文件，等待
    if not os.path.exists(target_file):
        print ('target_file not exist')
        time.sleep(time_kick)
        continue

    try:
        name  = 'log'
        easytime = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        file_name = '%s_user_%s.log' % (name,easytime)
        f_w = open(file_name, 'w')
        if init_flag:
            #读取整个文件
            for eachline in fileinput.input(target_file):
                print (eachline)
                f_w.write(eachline)
                record_count += 1

            init_flag = False
        else:
            #如果总行数小于当前行，那么认为文件更新了，从第一行开始读。
            total_count = os.popen('wc -l %s' % target_file).read().split()[0]
            total_count = int(total_count)
            if total_count < record_count:
                record_count = 0

            for eachline in fileinput.input(target_file):
                line_no = fileinput.filelineno()
                if line_no > record_count:
                    print (eachline)
                    f_w.write(eachline)
                    record_count += 1

        f_w.close()
    except:
        pass
    time.sleep(time_kick)
###检测文件价是否加载文件有文件更新，有的话就加载加载后分析