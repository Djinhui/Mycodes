class Solution1:
    def Fibonacci(self,n): # 时间复杂度2^n ,不通过
        if n==0: 
            return 0
        if n == 1:
            return 1
        if n > 1:
            return self.Fibonacci(n-1) + self.Fibonacci(n-2)
        return None



class Solution2:
    def Fibonacci(self,n):
        if n ==0:
            return 0
        if n == 1 :
            return 1
        if n ==2 :
            return 1
        if n >2:
            s = []*n
            s.append(1)
            s.append(1)
            for i in range(2, n):
                s.append(s[i - 2] + s[i - 1])
            return s[n - 1]



class Solution3:
    def Fibonacci(self,n):
        if n ==0:
            return 0
        if n == 1 :
            return 1
        a, b = 0,1
       
        for i in range(0,n-1):          
            a,b = b,a+b
        return b





     
