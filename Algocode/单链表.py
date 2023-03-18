class LNode:
    """定义结点类"""
    def __init__(self, elem, next_=None) -> None:
        self.elem = elem
        self.next = next_


class LinkedListUnderflow(ValueError):
    pass


class LList:
    def __init__(self) -> None:
        self._head = None

    def is_empty(self):
        return self._head is None

    def prepend(self, elem): # 在表头插入数据
        new_node = LNode(elem, next_=self._head)
        self._head = new_node
    
    def pop(self): # 删除表头结点并返回该结点数据
        if self._head is None:
            raise LinkedListUnderflow('in pop')
        e = self._head.elem
        self._head = self._head.next
        return e