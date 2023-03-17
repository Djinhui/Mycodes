# -*- coding: utf-8 -*-
class Solution:
    def __init__(self):
        self.count = 0
    def InversePairs(self,data):
        """思路: 定义一个 全局变量count 或 将count作为 实例属性 记录 逆序对的数量
    主要的过程是 在 归并排序的过程中 即 在比较左右两个数组中 从小到大的元素的过程中
    如果出现了 左数组的元素 低于 右数组中的元素 那么正常进行归并排序的过程 将左数组较小的元素
    添加进归并列表中 并且 左指针加1 但是如果出现了 左数组的元素 大于 右数组的元素的情况 不仅
    需要进行归并排序的过程 将右数组中的较小的元素 添加进归并列表中 还要将 count 加上
    当前这次比较中可以知道的逆序对的个数 这个逆序对的个数就是 左列表的长度 - 左指针的值
    之所以是这样 是因为 归并排序的过程中 我们假设的是 左右两个列表 一开始就都是 升序的排列"""
        def merge_sort(data):
            if data == []:
                return 0
            n = len(data)
            if n == 1:
                return data
            mid = n // 2
            leftData = merge_sort(data[:mid])
            rightData = merge_sort(data[mid:])
            l_n = len(leftData)
            r_n = len(rightData)
            left,right = 0,0
            result = []
            while left < l_n and right < r_n:
                if leftData[left] < rightData[right]:
                    result.append(leftData[left])
                    left += 1
                else:
                    result.append(rightData[right])
                    # 左边如果大 那么 当前 left后面的元素 肯定都比 这个 right的值大
                    self.count += l_n - left
                    right += 1
            result += leftData[left:]
            result += rightData[right:]
            return result
        merge_sort(data)
        result self.count % 1000000007
