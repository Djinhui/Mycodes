class Solution:
    def levelTravel(self,root):
        count = 0
        if root is None:
            return count
        q = []
        q.append(root)
        while len(q) != 0:
            tmp = []
            length = len(q)
            for i in range(length):
                node = q.pop(0)
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
                tmp.append(node.val)
            if tmp:
                count += 1 # 统计层数          
        return count

    def TreeDeth(self,pRoot):
        if pRoot is None:
            return 0
        count = self.levelTravel(pRoot)
        return count


# 如果该树只有一个结点，它的深度为1.
# 如果根节点只有左子树没有右子树，那么树的深度为左子树的深度加1；
# 同样，如果只有右子树没有左子树，那么树的深度为右子树的深度加1。
# 如果既有左子树也有右子树，那该树的深度就是左子树和右子树的最大值加1.
class Solution:
    def TreeDeth(self,pRoot):
        if pRoot is None:
            return 0
        count = max(self.TreeDeth(pRoot.left),self.TreeDeth(pRoot.right)) + 1
        return count