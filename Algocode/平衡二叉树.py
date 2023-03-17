# 判断一棵二叉树是不是平衡二叉树
class Solution:
    def IsBalanced_Solution(self,p):
        return self.dfs(p) != -1

    def dfs(self,p):
        if p is None:
            return 0
        left = self.dfs(p.left)
        if left == -1:
            return -1
        right = self.dfs(p.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return max(left, right) + 1
