'''
基数排序是一种非比较型整数排序算法，其原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。
由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。
'''
def radixSort(arr):
    digit = 0
    max_digit = 1
    max_value = max(arr)
    # 找出列表中最大的位数
    while 10 ** max_digit < max_value:
        max_digit = max_digit + 1

    while digit < max_digit:
        temp = [[] for i in range(10)]:
        for i in arr:
            # 求出每个元素的个，十，百位的值
            t = int((i/10**digit) %10)
            temp[t].append(i)

        coll = []
        for bucket in temp:
            for i in bucket:
                coll.append(i)
        arr = coll
        digit = digit + 1
    return arr

