'''
顺序查找也称为线形查找，属于无序查找算法。从数据结构线形表的一端开始，顺序扫描，
依次将扫描到的结点关键字与给定值k相比较，若相等则表示查找成功；
若扫描结束仍没有找到关键字等于k的结点，表示查找失败。
顺序查找的时间复杂度为O(n)'''

def order_search(key,alist):
    for i in range(len(alist)):
        if alist[i] == key:
            return i

    return -1


def binary_search(alist,key):
    low = 0
    high = len(alist) - 1
    while low <= high:
        mid = (low + high) // 2
        if alist[mid] == key:
            print(mid)
        elif alist[mid] < key:
            low = mid + 1
        else:
            high = mid - 1

def binary_search2(alist, key):
    if len(alist) == 0:
        return False
    
    mid = len(alist) // 2
    if alist[mid] == key:
        return mid
    elif alist[mid] < key:
        binary_search2(alist[mid+1:], key)
    else:
        binary_search2(alist[:mid-1], key)

# 插值查找
'''
基于二分查找算法，将查找点的选择改进为自适应选择，可以提高查找效率。当然，插值查找也属于有序查找。
对于表长较大，而关键字分布又比较均匀的查找表来说，插值查找算法的平均性能比折半查找要好的多。
反之，数组中如果分布非常不均匀，那么插值查找未必是很合适的选择。
查找成功或者失败的时间复杂度均为O(log2(log2n))
mid=low+(key-a[low])/(a[high]-a[low])*(high-low)
'''

# 哈希查找
class HashTable(object):
    def __init__(self,size):
        self.elem = [None for _ in range(size)] # 初始化哈希表
        self.count = size

    def hash(self, key): # 除留余数法
        return key % self.count

    def insert_hash(self,key):
        # 插入关键字到哈希表
        addresses = self.hash(key)
        while self.elem[addresses]: # 当前位置已有数字，发生冲突
            addresses = (addresses + 1) % self.count # 线性探测下一个位置
        self.elem[addresses] = key

    def search(self.key):
        start=addresses=self.hash(key)
        while self.elem[addresses] != key:
            addresses = (addresses + 1) % self.count
            # 没找到或者循环到了开始的位置
            if not self.elem[addresses] or addresses == start:
                return False
        return True

# 二叉树查找
class BSTNode(object):
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.left = left
        self.right = right

class BinaryTreeSearcher(object):
    def __init__(self):
        self.root = None

    def is_empty(self):
        return self.root is None

    def search(self,key):
        bt = self.root
        while bt is not None:
            entry = bt.data
            if key < entry:
                bt = bt.left
            elif key > entry:
                bt = bt.right
            else:
                return entry
        return None


    def insert(self,key):
        bt = self.root
        if bt is None:
            self.root = BSTNode(key)
            return
        while True:
            entry = bt.data
            if key < entry:
                if bt.left is None:
                    bt.left = BSTNode(key)
                    return
                bt = bt.left
            elif key > entry:
                if bt.right is None:
                    bt.right = BSTNode(key)
                    return
                bt = bt.right
            else:
                bt.data = key
                return

    def delete(self,key):
        p,q = None,self.root

        if not q:
            print("空树")
            return
        while q and q.data != key:
            p = q
            if key < q.data:
                q = q.left
            else:
                q = q.right
            if not q:
                return
        # 上面已将找到了要删除的节点，用q引用。而p则是q的父节点或者None（q为根节点时）。
        if not q.left:
            if p is None:
                self._root = q.right
            elif q is p.left:
                p.left = q.right
            else:
                p.right = q.right
            return
        # 查找节点q的左子树的最右节点，将q的右子树链接为该节点的右子树
        # 该方法可能会增大树的深度，效率并不算高。可以设计其它的方法。
        r = q.left
        while r.right:
            r = r.right
        r.right = q.right
        if p is None:
            self._root = q.left
        elif p.left is q:
            p.left = q.left
        else:
            p.right = q.left  

            
