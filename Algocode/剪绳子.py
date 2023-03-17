class Solution:
    def cutRope(self, number):
        result = []
        for i in range(2, number+1):
            result.append(self.prod(number,i))
        return max(result)
    def prod(self,n,m):
        res = []
        while n!=0:
            tmp = n // m
            res.append(tmp) 
            n = n-tmp
            m = m-1
        prodnum = 1
        for e in res:
            prodnum = prodnum * e
        return prodnum
s = Solution()
print(s.cutRope(8))