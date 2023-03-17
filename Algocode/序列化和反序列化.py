class Solution:
    def Serialize(self,root):

        retList = []
        def preOrder(root):
            if root is None:
                retList.append('#')
                return
            retList.append(str(root.val))
            preOrder(root.left)
            preOrder(root.right)
        preOrder(root)
        return ' '.join(retList)

    def Deserialize(self,s):
        retList = s.split()
        def dePreorder():
            if retList == []:
                return None
            rootVal = retList[0]
            del retList[0]
            if rootVal == '#':
                return None
            node = TreeNode(int(rootVal))

            leftNode = dePreorder()
            rightNode = dePreorder()
            node.left = leftNode
            node.right = rightNode
            return node
        pRoot = dePreorder()
        return pRoot
        
        