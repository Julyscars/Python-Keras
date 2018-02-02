import random,pyfpgrowth
import time
import os
#数据模拟
log ={'HDFS:NN:WARN','HDFS:DN:WARN','ZOOKEEPER:K:WARN','HBASE:RM:WARN','HBASE:NR:WARN','HIVE:MT:WARN','HIVE:BL:WARN','HDFS:NN:ERROR','HDFS:DN:ERROR','ZOOKEEPER:K:ERROR','HBASE:RM:ERROR','HBASE:NR:ERROR',
      'HIVE:MT:ERROR','HIVE:BL:ERROR'}
i = 0
logss = []
while i < 100:
    i = i+1
    var = random.sample(log, 14)

#检测文件
target_file = 'data.txt'
logss.append(var)
with open('data.txt','w',encoding='utf-8') as f :
    f.write(str(logss))
init_flag = True  # 初次加载程序
time_kick = 5

while True:
    print('日志存在开始分析')
    #没有日志文件，等待
    if not os.path.exists(target_file):
        print ('target_file not exist')
        time.sleep(time_kick)
        continue
    else:
        # fp-growth
        logdata = logss
        patterns = pyfpgrowth.find_frequent_patterns(logdata, 50)
        rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
        log_val = sorted(patterns.items(), key=lambda item: item[1], reverse=True)
        log_val1 = list(log_val)
        with open('patterns.txt', 'w', encoding='utf-8') as f:
            f.write(str(log_val1) + '\n ')
            f.close()
            print('patterns has saved')
        # data saving
        rule = []
        filename = 'data32.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            for i in patterns:
                j = list(i)
                rule.append(j)
                p = str(j)
                f.write(p)
                f.write('\n')
            f.close()
        pass
    time.sleep(time_kick)
