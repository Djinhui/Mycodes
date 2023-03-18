# 基本冒泡排序
def bubbleSort(alist):
    for passnum in range(len(alist)-1, 0, -1): # 第 1 轮需要比较 n -1 次，第 2 轮需要比较 n -2 次……第 n -1 轮需要比较 1 次。
        for i in range(passnum):
            if alist[i] > alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp

        print(alist)

# alist = [54,26,93,17,77,31,44,55,20]
# bubbleSort(alist)
# print(alist)

# 短路冒泡排序:在整个排序过程中没有交换，就可断定列表已经排好
def shortBubbleSort(alist):
    exchanges = True
    passnum = len(alist) - 1
    while passnum > 0 and exchanges:
        exchanges = False
        for i in range(passnum):
            if alist[i] > alist[i+1]:
                exchanges = True
                alist[i], alist[i+1] = alist[i+1], alist[i]
        passnum = passnum - 1

# alist = [2,1,3,4,5,6,7,8,9]
# shortBubbleSort(alist)
# print(alist)


# 选择排序
def selectionSort(alist):
    for fillslot in range(len(alist)-1, 0, -1):
        positonofMax = 0
        for loc in range(1, fillslot+1):
            if alist[loc] > alist[positonofMax]:
                positonofMax = loc

        alist[fillslot], alist[positonofMax] = alist[positonofMax], alist[fillslot]

# alist = [56,26,93,17,77,31,44,55,20]
# selectionSort(alist)
# print(alist)

# 插入排序
def insertSort(alist):
    for index in range(1, len(alist)):
        currentvalue = alist[index]
        position = index

        while position > 0 and alist[position-1] > currentvalue:
            alist[position] = alist[position-1]
            alist[position-1] = currentvalue
            position = position - 1

# alist = [56,26,93,17,77,31,44,55,20,1,1,93]
# insertSort(alist)
# print(alist)

def shellSort(alist):
    sublistcount = len(alist) // 2
    while sublistcount > 0:
        for startposition in range(sublistcount):
            gapInsertSort(alist, startposition,sublistcount)

            print(alist)
        sublistcount = sublistcount // 2

def gapInsertSort(alist,start, gap):
    for i in range(start+gap, len(alist), gap):
        currentvalue = alist[i]
        position = i
        while position >= gap and alist[position-gap]>currentvalue:
            alist[position] = alist[position-gap]
            alist[position-gap] = currentvalue
            position = position - gap

alist = [56,26,93,17,77,31,44,55,20]
shellSort(alist)
print(alist)