'''计数排序的核心在于将输入的数据值转化为键存储在额外开辟的数组空间中。
作为一种线性时间复杂度的排序，计数排序要求输入的数据必须是有确定范围的整数。'''


'''
桶排序是计数排序的升级版。它利用了函数的映射关系，高效与否的关键就在于这个映射函数的确定。
为了使桶排序更加高效，我们需要做到这两点：
1> 在额外空间充足的情况下，尽量增大桶的数量
2>使用的映射函数能够将输入的 N 个数据均匀的分配到 K 个桶中
'''
def countingSort(arr,maxValue):
    bucketLen = maxValue + 1
    bucket = [0] * bucketLen
    sortedIndex = 0
    arrLen = len(arr)
    for i in range(arrLen):
        if not  bucket[arr[i]]:
            bucket[arr[i]] = 0
        bucket[arr[i]] += 1
    for j in range(bucketLen):
        while bucket[j] > 0 :
            arr[sortedIndex] = j
            sortedIndex += 1
            bucket[j]-=1
    return arr

# 分桶
[1,1,2,2,2,2,2,2,2,3,4,4,6,7,7,8,8,9,9]

[1,1,  2,2,2,2,2,2,2,   3,  4,4,   6,  7,7,  8,8,  9,9]

