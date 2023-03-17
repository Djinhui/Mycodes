#冒泡排序：重复的遍历要排序的数列，一次比较两个元素，如果他们的顺序错误，就把他们交换过来
#对每一对相邻的元素进行同样的操作，从开始的一对到最后的一对，第一轮遍历完，最大值在数列的最后面
# 然后开始第二轮
# 轮数   比较次数
#  1       n-1
#  2       n-2
#   ........
#  n-1      1

# O(n^2) 稳定
def bubble_sort(alist):
    for i in range(len(alist)-1,0,-1):  # n-1,n-2,n-3,...1
        for j in range(i):
            if alist[j] > alist[j+1]:
                alist[j],alist[j+1] = alist[j+1],alist[j]

# 冒泡改进
def bubble_sort2(alist):
    for i in range(len(alist)-1,0,-1):
        count = 0
        for j in range(i):
            if alist[j] > alist[j+1]:
                alist[j],alist[j+1] = alist[j+1],alist[j]
                count += 1
        if count == 0: # 说明没有交换，数列本身就是有序的
            break



# 选择排序:首先在未排序序列中找到最小的元素，存放到序列的起始位置，然后从剩下的未排序序列中寻到
# 最小的元素，放在已排序序列的末尾。重复直到所有的元素均排序完毕
# O(n^2)不稳定
def select_sorted_(alist):
    n = len(alist)
    for i in range(n-1):
        minIndex = i # 假设它是当前最小值的索引
        for j in range(i+1, n):
            if alist[j] < alist[minIndex]:
                minIndex = j
        if minIndex != i:
            alist[i],alist[minIndex] = alist[minIndex],alist[i] # 交换假设的最小值与真正的最小值

# 插入排序:通过构造有序序列（开始时把第一个元素看做有序序列）,对于未排序数据，在已排序序列中从后向前
# 扫描，找到相应的位置并插入。插入排序在实现上，在从后向前扫描过程中，需要反复把已排序元素逐步后移，为
# 新元素的插入提供空间
# O(n^2)稳定
def insert_sort(alist):
    n = len(alist)
    for i in range(1,n): # 对未排序序列元素
        for j in range(i,0,-1): # 从后向前扫描已排序序列
            if alist[j] < alist[j-1]:
                alist[j],alist[j-1] = alist[j-1], alist[j]

def insert_sort2(alist):
    n = len(alist)
    for i in range(1,n):
        j=i
        while j>0:
            if alist[j] < alist[j-1]:
                alist[j],alist[j-1] = alist[j-1],alist[j]
            else:
                break
            j-=1


# 快速排序
# O（nlogn） 不稳定
'''
假设要排序的序列为A[0],A[1],...A[N-1],首先任选一个元素（通常为第一个元素）作为基准数据，然后将
所有比他小的都放在他的左边，所有比他大的都放在他的右边，称为一趟，然后对基准数据左边的序列和右边的
序列分别执行同样的步骤。
1): 设置两个变量low，high，排序开始的时候：low=0,high=N-1
2): 将序列第一个元素作为基准，赋值给mid，即mid=A[0]
3): 从high开始向前搜索，即有后开始向前搜索（high--）,找到第一个小于mid的值A[high],
将A[high]与A[low]互换
4): 从low开始后向搜索，即从前向后搜素(low++)，找到第一个大于mid的A[low]，将A[low]和A[high]交换
5): 重复3)4),直到low=high
'''
def quickSort(alist):
    if len(alist) < 2:
        return alist
    else:
        pivot = alist[0] # 基准
        less = [i for i in alist[1:] if i < pivot]
        greater = [i for i in alist[1:] if i >= pivot]
        return quickSort(less) + [pivot] + quickSort(greater)
print(quickSort([9,3,1,10]))
def quickSort2(alist,start=None,end=None):
    if start is None:
        start = 0
    if end is None:
        end = len(alist) - 1

    if start >= end:
        return
    # 设定起始元素为要寻找位置的基准元素
    mid = alist[start]
    # low 为序列左边的由左向右移动的游标
    low = start
    # high 为序列右边的由右向左移动的游标
    high = end
    while low < high:
        # 如果 low 与 high 未重合， high 指向的元素不比基准元素小， 则 high 向左移动
        while low < high and alist[high] >= mid:
            high -= 1
        #high 指向的元素比基准元素小,将 high 指向的元素放到 low 的位置上
        alist[low] = alist[high]

        # 如果 low 与 high 未重合， low 指向的元素比基准元素小， 则 low 向右移动
        while low < high and alist[low] < mid:
            low += 1
        #low 指向的元素比基准元素大,将low指向的元素放到high的位置上
        alist[high] = alist[low]

    # 退出循环后， low 与 high 重合， 此时所指位置为基准元素的正确位置
    # 将基准元素放到该位置
    alist[low] = mid

    # 对基准元素左边的子序列进行快速排序
    quick_sort(alist, start, low-1)
    # 对基准元素右边的子序列进行快速排序
    quick_sort(alist, low+1, end)

# 快排非递归
def partition(alist, start,end):
    pivot = alist[start]
    while start < end:
        while start < end and alist[end] >= pivot:
            end -= 1
        alist[start] = alist[end]
        while start < end and alist[start] < pivot:
            start += 1
        alist[end] = alist[start]
    # 此时start = end
    alist[start] = pivot
    return start
def quick_sort(alist):
    # 模拟栈操作实现非递归的快排
    if len(alist) < 2:
        return alist
    stack = []
    stack.append(len(alist)-1)
    stack.append(0)
    while stack:
        l = stack.pop()
        r = stack.pop()
        index = partition(alist,l,r)
        if l < index - 1:
            stack.append(index-1)
            stack.append(l)
        if r > index + 1:
            stack.append(r)
            stack.append(index + 1)
            




































# 归并排序:先递归分解数组，再排序合并数组
# 将数组分解最小之后，然后合并两个有序数组，基本思想是比较两个数组的最前面的数，
# 谁小就先取谁（有两个指针维持）,取了值后的那个数组的指针后移一位，然后再比较，直到一个数组为空，
# 最后把另一个数组的剩余部分复制过来即可
def merge_sort(alist):
    n = len(alist)
    # 递归退出条件
    if n <= 1:
        return alist
    
    mid = n // 2 # 拆分数组
    left_li = merge_sort(alist[:mid])
    right_li = merge_sort(alist[mid:])

    left_pointer,right_pointer = 0,0
    result = []

    while left_pointer < len(left_li) and right_pointer < len(right_li):
        if left_li[left_pointer] < right_li[right_pointer]:
            result.append(left_li[left_pointer])
            left_pointer += 1
        else:
            result.append(right_li[right_pointer])
            right_pointer += 1
        
    # 当一个数组的指针到达终点，另一个数组直接加到以排序列表的后面
    result += left_li
    result += right_li

    return result

def merge(left, right):
    left_pointer,right_pointer = 0,0
    result = []
    while left_pointer < len(left) and right_pointer < len(right):
        if left[left_pointer] < right[right_pointer]:
            result.append(left[left_pointer])
            left_pointer += 1
        else:
            result.append(right[right_pointer])
            right_pointer += 1
    result += left
    result += right
    return result


def merge_sort2(alist):
    if len(alist) <= 1:
        return alist
    num = len(alist) // 2
    left = merge_sort(alist[:num])
    right = merge_sort(alist[num:])
    return merge(left, right)
            
