# 打印前numRow行
class Solution: 
    def generate(self, numRows: int) -> List[List[int]]:
        result = []
        for i in range(numRows):
            now = [1] * (i+1)
            if i >= 2:
                for j in range(1,i):
                    now[j] = pre[j-1] + pre[j]
            result += [now]
            pre = now
        return result