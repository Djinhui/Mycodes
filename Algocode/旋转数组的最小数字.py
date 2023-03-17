# 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
# 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
# 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
# NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        minNum = 0
        for i in range(len(rotateArray)):
            minNum = minNum if minNum < rotateArray[i] and minNum != 0 else rotateArray[i]
        return minNum

def minNumberInRotateArray(rotateArray): # 利用二分法，旋转数组部分有序
    if not  rotateArray:
        return 0
    
    low = 0
    high = len(rotateArray) - 1
    while low <= high:
        mid = (low + high) // 2
        if rotateArray[mid] < rotateArray[mid-1]:
            return rotateArray[mid]
        elif rotateArray[mid] < rotateArray[high]:
            high = mid - 1
        else:
            low = mid + 1
    return 0

