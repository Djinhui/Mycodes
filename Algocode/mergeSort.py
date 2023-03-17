def mergeSort(alist):
    n = len(alist)
    if n<=1: return alist
    mid = n// 2

    left_li = mergeSort(alist[:mid])
    right_li = mergeSort(alist[mid:])

    left_pointer,right_pointer = 0,0
    result = []

    while left_pointer < len(left_li) and right_pointer < len(right_li):
        if left_li[left_pointer] <= right_li[right_pointer]:
            result.append(left_li[left_pointer])
            left_pointer += 1

        else:
            result.append(right_li[right_pointer])
            right_pointer += 1

    # 当一个数组的指针到达终点，另一个数组直接加到以排序列表的后面
    result += left_li[left_pointer:]  
    result += right_li[right_pointer:]
    return result



def merge(left, right):
    '''合并操作，将两个有序数组left和right合并成一个大的有序数组'''
    #left与right的下标指针
    l, r = 0, 0
    result = []
    while l<len(left) and r<len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += left[l:] # 若最后left列表剩余，则将其剩余部分加入到result后面
    result += right[r:] # 若最后right列表剩余，则将其剩余部分加入到result后面
    return result


def merge_sort(alist):
    if len(alist) <= 1:
        return alist
    # 二分分解
    num = len(alist)//2
    left = merge_sort(alist[:num])
    right = merge_sort(alist[num:])
    # 合并
    return merge(left,right)

