# 第一层从左到右打，第二层从右到左打印，。。。。。。

class Solution:
    def Print(self,pRoot):
        if pRoot is None:
            return []

        stack1 = [pRoot] # left--->right
        stack2 = []       # right--->left
        ret = []

        while stack1 or stack2:
            if stack1:
                tmpRet = []
                while stack1:
                    tmpNode = stack1.pop()
                    tmpRet.append(tmpNode.val)
                    if tmpNode.left:
                        stack2.append(tmpNode.left)
                    if tmpNode.right:
                        stack2.append(tmpNode.right)
                ret.append(tmpRet)
            if stack2:
                tmpRet = []
                while stack2:
                    tmpNode = stack2.pop()
                    tmpRet.append(tmpNode.val)
                    if tmpNode.right:
                        stack1.append(tmpNode.right)
                    if tmpNode.left:
                        stack1.append(tmpNode.left)
                ret.append(tmpRet)
        return ret