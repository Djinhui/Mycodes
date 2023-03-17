# 判断输入的数组是不是某二叉搜索树的后续遍历结果，数组中没有重复的元素
class Solution:
    def VerifySequenceOfBST(self,sequence):
        if sequence == []:
            return False

        rootNum = sequence[-1]
        del sequence[-1]

        index = None
        for i in range(len(sequence)):
            if index is None and sequence[i] > rootNum: # 左边的小于根，右边的大于根
                index = i
            if index is not None and sequence[i] < rootNum:
                return False

        if sequence[:index] == []:
            return True
        else:
            leftRet = self.VerifySequenceOfBST(sequence[:index])

        if sequence[index:]:
            return True
        else:
            rightRet = self.VerifySequenceOfBST(sequence[index:])

        return leftRet and rightRet

def VerifySequenceOfBST(sequence):
    if sequence is None or len(sequence) == 0:
        return False
    length = len(sequence)
    root = sequence[-1]
    # 在二叉搜索树中,左子树节点小于根节点
    for i in range(0, length):
        if sequence[i]> root:
            break
    # 二叉搜索树中右子树的节点都大于根节点
    for j in range(i,length):
        if sequence[j]< root:
            return False
    # 判断左子树是否为二叉树
    left=True
    if i>0:
        left = self.VerifySequenceOfBST(sequence[:i])       
    # 判断右子树是否为二叉树
    right=True
    if i < length-1:
        right = self.VerifySequenceOfBST(sequence[i:length-1])
    return left and right