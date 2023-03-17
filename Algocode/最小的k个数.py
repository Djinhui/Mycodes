# 最大堆

class Solution:
    def GetLeastNumbers_Solution(self,tinput,k):
        # 创建最大堆
        # i结点的父结点下标就为(i – 1) / 2。它的左右子结点下标分别为2 * i + 1和2 * i + 2
        def createMaxHeap(num):
            maxHeap.append(num)
            currentIndex = len(maxHeap)-1
            while currentIndex != 0:
                parentIndex = (currentIndex - 1) // 2
                if maxHeap[parentIndex] < maxHeap[currentIndex]:
                    maxHeap[parentIndex],maxHeap[currentIndex] = maxHeap[currentIndex],maxHeap[parentIndex]
                else:
                    break

        # 调整最大堆
        def adjustMaxHeap(num):
            if num < maxHeap[0]:
                maxHeap[0] = num

            index = 0
            maxHeapLen = len(maxHeap)
            while index < maxHeapLen:
                leftIndex = index*2 + 1
                rightIndex = index*2 + 2
                largestIndex = 0 
                if rightIndex < maxHeapLen:
                    if maxHeap[rightIndex] < maxHeap[leftIndex]:
                        largestIndex = leftIndex
                    else:
                        largestIndex = rightIndex
                elif leftIndex < maxHeapLen:
                    largestIndex = leftIndex
                else:
                    break
                if maxHeap[index] < maxHeap[largestIndex]:# 根小于子，则调整
                    maxHeap[index],maxHeap[largestIndex] = maxHeap[largestIndex],maxHeap[index]
                index = largestIndex

        
        maxHeap = []
        if len(tinput) < k or k <= 0:
            return []
        inputLen = len(tinput)
        for i in range(inputLen):
            if i < k:
                createMaxHeap(tinput[i])
            else:
                adjustMaxHeap(tinput[i])

        maxHeap.sort()
        return maxHeap