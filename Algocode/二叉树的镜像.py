class Solution:
    def Mirror(self,root):
        if root is None:
            return None
        
        root.left,root.right = root.right,root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        