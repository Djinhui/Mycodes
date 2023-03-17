class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        self.items.pop()

    def peak(self):
        '''返回栈顶元素'''
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)

        