def shell_sort(alist):
    n = len(alist)

    # 初始步长
    gap = n // 2 
    while gap > 0:
        # 按步长进行插入排序
        for i in range(gap, n):
            j = i  
            # 插入排序
            while j >= gap and alist[j - gap] > alist[j]:
                alist[j-gap],alist[j] = alist[j],alist[j - gap]
                j -= gap

        gap = gap // 2

        