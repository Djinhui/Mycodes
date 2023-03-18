# 顺序搜索 元素无序
def sequentialSearch(alist, item):
    pos = 0
    found = False
    while pos < len(alist) and not found:
        if alist[pos] == item:
            found = True
        else:
            pos += 1

    return found

# 顺序搜索 元素有序
def orderSequentialSearch(alist, item):
    pos = 0
    found = False
    stop = False
    while pos < len(alist) and not found and not stop:
        if alist[pos] == item:
            found = True
        else:
            if alist[pos] > item:
                stop = True
            else:
                pos += 1

    return found

# 二分搜索 元素有序从小到大排列
def binarySearch(alist, item):
    first = 0
    last = len(alist) -1
    found = False
    while first <= last and not found:
        midpoint = (first + last) // 2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1
    return found

# 二分搜索 递归实现
def binarySearch2(alist, item):
    if len(alist) == 0:
        return False
    else:
        midpoint = len(alist) // 2
        if alist[midpoint] == item:
            return True
        else:
            if item < alist[midpoint]:
                return binarySearch2(alist[:midpoint], item)
            else:
                return binarySearch2(alist[midpoint:], item)


# 字符串哈希
def hash(astring, tablesize):
    sum = 0
    for pos in range(len(astring)):
        sum = sum + ord(astring[pos]) * (pos + 1)
    return sum%tablesize

