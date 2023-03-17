# 四个王为0，可以看为任意牌
class Solution:
    def IsContinuous(self,numbers):
        if len(numbers):
            while min(numbers)==0:
                numbers.remove(0)
            if max(numbers) - min(numbers) <= 4 and len(numbers)==len(set(numbers)):
                return True
        return False

# 先统计王的数量，再把牌排序，如果后面一个数比前面一个数大于1以上，那么中间的差值就必须用王来补了。
# 看王的数量够不够，如果够就返回true，否则返回false    
if not numbers:
    return False
numbers.sort()    
zeroNum = numbers.count(0)

for i,v in enumerate(numbers[:-1]):
    if v != 0:
        if numbers[i+1] == v:return False
        zeroNum = zeroNum-(numbers[i+1]-v) + 1
        if zeroNum < 0:return False
return True