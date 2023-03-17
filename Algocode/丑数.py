# 丑数：只含有质因子2,3,5
# 把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
class Solution:
    def GetUglyNumber_Solution(self, index):
        if index < 1:
            return 0
        uglyList = [1]
        twoPointer = 0
        threePointer = 0
        fivePointer = 0

        count = 1
        while count != index:
            minValue = min(2*uglyList[twoPointer],3*uglyList[threePointer],5*uglyList[fivePointer])
            uglyList.append(minValue)
            count += 1

            if minValue == 2*uglyList[twoPointer]:
                twoPointer += 1
            if minValue == 3*uglyList[threePointer]:
                threePointer += 1
            if minValue == 5*uglyList[fivePointer]:
                fivePointer += 1

        return uglyList[-1]
            

        
