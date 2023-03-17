class Solution:
    def maxInWindows(self,num,size):
        if size == len(num):
            return [max(num)]
        if size < 2 or size > len(num):
            return []
        res = []
        for i in range(len(num)-size+1):
            res.append(max(num[i:i+size]))
        return res