class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def hot_potato(namelist, num):
    simqueue = Queue()
    for name in namelist:
        simqueue.enqueue(name)

    while simqueue.size() > 1:
        for _ in range(num):
            simqueue.enqueue(simqueue.dequeue())

        simqueue.dequeue()

    return simqueue.dequeue()


def jsoephus_A(n,k,m):  #O(n^2 * logN)
    """n个人围一圈从第k个人开始报数,报到第m个数的人退出,然后从下一个人开始继续报数"""
    people = list(range(1, n+1)) # 编号
    i = k- 1 # 下标-1
    for num in range(n): # n次迭代
        count = 0
        while count < m:
            if people[i] > 0:
                count += 1
            if count == m:
                print(people[i], end='')
                people[i] = 0 # 置0表示已出局
            i = (i+1) % n

        if num < n-1:
            print(', ', end='')
        else:
            print('')
    return


def josephus_B(n, k, m): # O(n^2)
    people = list(range(1, n+1))

    num, i = n, k - 1
    for num in range(n, 0, -1):
        i = (i + m - 1) % num
        print(people.pop(i), end=(', ' if num >1 else "\n"))
    return

def josephus_C(n,k, m): # O(m*n)
    """基于循环单链表"""
    pass