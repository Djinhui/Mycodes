class Solution:

    def BinarySearch(self,data,mlen,k):
        start = 0
        end = mlen - 1
        while start <= end:
            mid = (start + end)//2
            if data[mid] == k:
                return mid
            elif data[mid] < k:
                start = mid + 1
            else:
                end = mid - 1
        return -1



    def GetNumberOgK(self,data,k):
        mlen = len(data)
        index = self.BinarySearch(data,mlen,k)
        if index == -1:
            return 0
        count = 1
        for i in range(1,mlen):
            if index+i < mlen and data[index+i] == k:
                count += 1
            if index - i >= 0 and data[index-i] == k:
                count += 1
        return count