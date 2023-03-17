# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        result =[]
        plow=1
        phigh=2
        while phigh>plow:
            cur = (phigh+plow)*(phigh-plow+1)/2
            if tsum == cur:
                tmp = []
                for i in range(plow,phigh+1):
                    tmp.append(i)
                result.append(tmp)
                plow += 1
            elif cur<tsum:
                phigh += 1
            else:
                plow += 1
        return result