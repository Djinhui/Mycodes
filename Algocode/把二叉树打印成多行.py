# 从上到下打印二叉树，同义层从左到右打印,每层一行
class Solution:
    def Print(self,pRoot):
        if pRoot is None:
            return []

        queue1 = [pRoot]
        queue2 = []
        ret = []
        while queue1 or queue2:
            if queue1:
                tmpRet = []
                while queue1:
                    tmpNode = queue1[0]
                    tmpRet.append(tmpNode.val)
                    del queue1[0]
                    if tmpNode.left:
                        queue2.append(tmpNode.left)
                    if tmpNode.right:
                        queue2.append(tmpNode.right)
                ret.append(tmpRet)
            if queue2:
                tmpRet = []
                while queue2:                   
                    tmpNode = queue2[0]
                    tmpRet.append(tmpNode.val)
                    del queue2[0]
                    if tmpNode.left:
                        queue1.append(tmpNode.left)
                    if tmpNode.right:
                        queue1.append(tmpNode.right)
                ret.append(tmpRet)
        return ret