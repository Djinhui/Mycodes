import os
import sys


contents = []
count = 0
# 打开输入文件，逐行处理
with open('test.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():

        line = line.strip('\n').strip()
        if not line:
            contents.append('')
            continue

        ch, label = line.split(' ')

        if label[0] != 'M':
            contents.append(line)
            continue
        else:
            new_label = 'I' + label[1:]
            new_sent = ch + ' ' + new_label
            contents.append(new_sent)

        count += 1
        if count % 10000 == 0:
            print('count = ', count)
# 打印计数器的值
print('count = ', count)
print('--------------------------')
# 将处理后的结果写入输出文件
with open('test.txt', 'w', encoding='utf-8') as f1:
    for line in contents:
        f1.write(line + '\n')


