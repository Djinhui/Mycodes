def bubble_sort(alist):
    for j in range(len(alist)-1,0,-1):
        # j表示每次遍历需要比较的次数，是逐渐减小的
        for i in range(j):
            if alist[i] > alist[i+1]:
                alist[i],alist[i+1] = alist[i+1],alist[i]
                
def bubbleSort(arr):
    for i in range(1,len(arr)):
        for j in range(0,len(arr)-i):
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]

    return arr

def BubbleSort(L):
    for i in range(len(L)):
        count = 0
        for j in range(len(L)-i-1):
            if L[j] > L[j+1]:
                L[j],L[j+1] = L[j+1],L[j]
                count += 1
        if count == 0:
            break