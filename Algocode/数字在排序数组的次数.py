# -*- coding:utf-8 -*-
class Solution:
    def BinarySearch(self,data,k):
        start = 0
        end = len(data)-1
        while start <= end:
            mid = start + (end-start)//2
            if data[mid] == key:
                return mid
            elif data[mid]>k:
                end = mid -1
            else:
                start = mid + 1
        return -1
    def GetNumberOfKeys(self,data,k):
        index = self.BinarySearch(data,k)
        if index == -1:
            return 0
        count = 1
        for i in range(1,len(data)):
            if index + i < len(data) and data[index+i] == key:
                count += 1
            if index -i >= 0 and data[index-i] == key:
                count +=1
        return count