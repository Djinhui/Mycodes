class TreeNode(object):
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

def preorder(root):  # root这里为根节点
    if root is None:
        return None
    print(root.val)
    preorder(val.left)
    preorder(val.right)

def midorder(root): # root
    if root is None:
        return None
    midorder(root.left)    
    print(root.val)
    midorder(root.right)

def postorder(root): # root
    if root is None:
        return None
    postorder(val.left)
    postorder(val.right)
    print(root.val)

def preOrder(root): # root
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

def midOrder(root): # root
    if root is None:
        return None
    stack = []
    tmpNode = root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode = tmpNode.left
        node = stack.pop()
        print(node.val)
        tmpNode = node.right

def postOrder(root): # root
    if root is None:
        return None
    stack = []
    tmpNode = root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode = tmpNode.left

        node = stack[-1]        
        tmpNode = node.right
        if node.right is None:
            node = stack.pop()
            print(node.val)
            while stack and tmpNode == stack[-1].right:
                node = stack.pop()
                print(node.val)