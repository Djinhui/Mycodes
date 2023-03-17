class TreeNode(object):
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

# 广度优先：层次遍历
# 深度优先：前序遍历，中序遍历，后序遍历

# 递归
def preOrderR(root):
    if root is None:
        return None
    print(root.val)
    preOrderR(root.left)
    preOrderR(root.right)

def midOrderR(root):
    if root is None:
        return None
    midOrderR(root.left)
    print(root.val)
    midOrderR(root.right)

def postOrderR(root):
    if root is None:
        return None
    postOrderR(root.left)
    postOrderR(root.right)
    print(root.val)

# 非递归
def preOrder(root):
    if root is None:
        return None
    stack = []
    tmpNode = root
    while tmpNode or stack:
        while tmpNode:
            print(tmpNode.val)
            stack.append(tmpNode)
            tmpNode = tmpNode.left

        node = stack.pop()
        tmpNode = node.right

def midOrder(root):
    if root is None:
        return None
    stack = []
    tmpNode = root
    while tmpNode or stack:
        while tmpNode:
            #print(tmpNode.val)
            stack.append(tmpNode)
            tmpNode = tmpNode.left

        node = stack.pop()
        print(node.val)
        tmpNode = node.right

def postOrder(root):
    if root is None:
        return None
    stack = []
    tmpNode = root
    while tmpNode or stack:
        while tmpNode:
            #print(tmpNode.val)
            stack.append(tmpNode)
            tmpNode = tmpNode.left

        #node = stack.pop()
        node = stack[-1]
        #print(node.val)
        tmpNode = node.right
        if node.right is None:
            
            node = stack.pop()
            print(node.val)
            while stack and tmpNode == stack[-1].right:
                node = stack.pop()
                print(node.val)





if __name__ == "__main__":
    t1 = TreeNode(1)
    t2 = TreeNode(2)
    t3 = TreeNode(3)
    t4 = TreeNode(4)
    t5 = TreeNode(5)
    t6 = TreeNode(6)
    t7 = TreeNode(7)
    t8 = TreeNode(8)

    t1.left = t2
    t1.right = t3
    t2.left = t4
    t2.right = t5
    t3.left = t6
    t3.right = t7
    t6.right = t8

    preOrderR(t1)
    midOrderR(t1)
    postOrderR(t1)
