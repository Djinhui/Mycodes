def quickSort(array):
    if len(array) < 2: # base condition :为空或只有一个元素的列表是有序的
        return array
    else:
        pivot = array[0] # 选择一个基准值
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i >= pivot]
        return quickSort(less) + [pivot] + quickSort(greater)


alist = [54,26,93,17,77,31,44,55,20]
sortdList = quickSort(alist)
print(sortdList)
    