"""
优先队列与栈和队列类似，可以将数据元素保存其中，可以访问和弹出。
优先队列的特点是存入其中的每项数据都附加一个数值，表示这项数据的优先级。
优先队列应保证在任何时候访问和弹出的总是当时这个结构里保存的所有元素中优先级最高的。
"""

# 基于列表实现的优先队列
# demo：假设值较小的元素优先级更高，要求弹出最优先数据项的操作为O(1)，则要求最小值在尾端

class PrioQueueError(ValueError):
    pass

class PrioQue:
    def __init__(self, elist=[]) -> None:
        self._elems = list(elist)
        self._elems.sort(reverse=True)

    def enqueue(self, elem):
        """插入一个新数据项"""
        i = len(self._elems) - 1
        while i >= 0: # 寻找第一个大于elem的元素的下标
            if self._elems[i] <= elem: 
                i -= 1
            else:
                break
        self._elems.insert(i+1, elem)

    def is_empty(self):
        return not self._elems

    def peek(self): 
        if self.is_empty():
            raise PrioQueueError('in peek')
        return self._elems[-1]

    def dequeue(self):
        if self.is_empty():
            raise PrioQueueError('in dequeue')
        return self._elems.pop()