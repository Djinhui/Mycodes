# 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
# 假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，
# 序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
# （注意：这两个序列的长度是相等的）
class Solution:
    def IsPopOrder(self, pushV, popV):
        # 按照pushV的方式压入栈
        # 弹出的时候需要循环判断是否需要弹出
        # 判断是否需要弹出的时机，刚刚压入过后就判断
        # 判断需要弹出的情况的条件，压入栈的顶部和弹出栈的顶部数据相等
        if pushV == [] or len(pushV) != len(popV):
            return None
        
        stack = [] 
        index = 0
        for item in pushV:
            stack.append(item)
            while  stack and stack[-1] == popV[index]:
                stack.pop()
                index = index + 1
        return True if stack == [] else False

s = Solution()
print(s.IsPopOrder([1,2,3,4,5],[4,5,3,2,1]))