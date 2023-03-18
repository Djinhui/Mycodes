# 要点
"""
基于栈的迷宫求解，栈内保存着一些位置，每个位置之下就是到达它的路径上的前一个位置。
这样，搜索中找到出口时，当时的栈正好保存着从入口到出口的一条路径。

对基于队列的算法，队列里保存的位置及其顺序与路径无关。

基于栈的搜索为深度优先搜索。这种搜索比较冒进，不撞南墙不回头。如果顺利，可能只探索了不多的位置
就找到了解，但也可能陷入很深的无解区域。

基于队列的搜索是宽度优先搜索。这种搜索步步为营，只有在检查完所有与入口同样距离的位置后才更多前进一步

深度和宽度优先搜索的性质：
    1. 假设搜索问题有解，深度优先可能陷入包含无穷多状态的无解子区域，
    宽度优先搜索逐步扫描，只要存在到达解的有穷长路径，宽度优先搜索一定能找到解，且是最短路径
    2. 如果找到解，如何得到相应的路径
    基于栈的搜索，在栈中就保存着路径。基于队列的搜索需要其他方法记录路径信息
    3. 搜索所有可能的解和最优解
    深度优先搜索找到一个解后进行回溯，有可能找到其他解，遍历完整个状态空间找到所有解，只有找到所有解
    才能确定最优解
    宽度优先搜索找到一个解之后也能继续搜索其他解，但是解的路径越来越长，因此第一个解就是最优解
    4. 搜索的时间开销
    无论栈还是队列，探查一个状态的开销都是O(1),总代价受限于状态空间的规模
    5. 搜索的空间开销
    深度优先搜索所需的栈空间由找到一个解(或所有解)之前遇到过的最长那条搜索路径确定
    宽度优先搜索所需的队列空间由搜索过程中可能路径分支最多的那一层确定。
"""





# 迷宫maze用矩阵表示，0代表可走，1代表无法通行
# 点(i,j)四个相邻位置相对坐标用dirs表示
dirs = [(0,1),(1,0),(0,-1),(-1,0)]

def mark(maze, pos): # 给迷宫maze的位置pos标记为2表示到访过
    maze[pos[0], maze[pos[1]]] = 2

def passabel(maze, pos): # 检查位置pos是否可行
    return maze[pos[0],pos[1]] == 0

# 递归求解 #
def find_path(maze, pos, end):
    mark(maze, pos)
    if pos == end:
        print(pos, end=' ')
        return True
    
    for i in range(4):
        nextp = pos[0] + dirs[i][0], pos[1] + dirs[i][1]
        # 考虑下一个方向
        if passabel(maze, nextp):
            if find_path(maze, nextp, end):
                print(pos, end=' ')
                return True

    return False

# 栈求解 #

# 定义一个栈
class SStack:
    def __init__(self) -> None:
        self._stack = []

    def push(self, item):
        self._stack.append(item)

    def is_empty(self):
        return len(self._stack) == 0

    def pop(self):
        return self._stack.pop()
    pass

# 打印搜索路径
def print_path(end, pos,stack):
    pass

def maze_solver(maze, start, end):
    if start == end:
        print(start)
        return True

    st = SStack()
    mark(maze, start)

    st.push((start, 0)) # 入口和方向(dirs)0的序对入栈
    while not st.is_empty(): # 走不通时回退
        pos, nxt = st.pop() # 取栈顶及其探查方向
        for i in range(nxt, 4): # 依此检查未探查方向
            nextp = (pos[0] + dirs[i][0], pos[1] + dirs[i][1]) # 下一位置
            if nextp == end: # 到达出口，打印路径
                print_path(end, pos,st)
                return True
            if passabel(maze, nextp): # 遇到未探查的新位置
                st.push((pos, i+1)) # 原位置和下以方向入栈
                mark(maze, nextp)
                st.push((nextp, 0)) # 新位置入栈
                break # 退出内层循环，下次迭代将从新栈顶为当前位置继续

    print('not found')

# 队列求解 # 
# 定义队列
class SQueue:
    def __init__(self) -> None:
        self._queue = []

    def enqueue(self, item):
        self._queue.append(item)

    def dequeue(self):
        return self._queue.pop(0)

    def is_empty(self):
        return self._queue == []

def maze_slover_queue(maze, start, end):
    if start == end:
        print('path find')
        return
    qu = SQueue()
    mark(maze, start)
    qu.enqueue(start) # start位置入队
    while not qu.is_empty(): # 还有候选位置
        pos = qu.dequeue() 
        for i in range(4): # 检查各个方向
            nextp = (pos[0] + dirs[i][0], pos[1] + dirs[i][1]) 
            if passabel(maze, nextp):
                if nextp == end:
                    print('path find')
                    return
                mark(maze, nextp)
                qu.enqueue(nextp) # 新位置入队
    print('no path')



