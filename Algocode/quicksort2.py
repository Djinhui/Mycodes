def quickSort(alist,start=0,end=None):
    # base condition
    if end is None:
        end = len(alist)-1
    if start >= end:
        return

    # 设定起始元素为要寻找位置的基准元素
    mid_value = alist[start]
    # low为序列左边的由左向右移动的游标
    low = start
    # high为序列右边的由右向左移动的游标
    high = end

    while low < high:
        # 如果low与high未重合，high指向的元素不比基准元素小，则high向左移动
        while low < high and alist[high] >= mid_value:
            high -= 1
        # 将high指向的元素放到low的位置上
        alist[low] = alist[high]

        # 如果low与high未重合，low指向的元素比基准元素小，则low向右移动
        while low < high and alist[low] < mid_value:
            low += 1
        # 将low指向的元素放到high的位置上
        alist[high] = alist[low]

    # 退出循环后，low与high重合，此时所指位置为基准元素的正确位置
    # 将基准元素放到该位置
    alist[low] = mid_value

    # 对基准元素左边的子序列进行快速排序
    quick_sort(alist, start, low-1)

    # 对基准元素右边的子序列进行快速排序
    quick_sort(alist, low+1, end)



if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    quick_sort(li, 0, len(li)-1)
    print(li)


class Solution:
    def quickSort(self,L,left,right):
        stack = []
        stack.append(left)
        stack.append(right)
        while stack:
            r = stack.pop()
            l = stack.pop()
            index = self.partSort(L,l,r)
            if l<=index-1:
                stack.append(l)
                stack.append(index-1)
                if r>=index+1:
                    stack.append(index+1)
                    stack.append(r)

    def partSort(self,L,left,right):
        tail = left
        for i in range(left,right):
            if L[i] < L[right]:
                self.swap(L,i,tail)
                tail +=1 
        self.swap(L,tail,right)
        return tail

    def swap(self,L,i,j):
        L[i],L[j] = L[j],L[i]
