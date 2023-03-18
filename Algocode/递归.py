def list_sum(alist):
    if len(alist) == 1:  # 1. 基本结束条件：列表长度为1
        return alist[0]
    else:
        return alist[0] + list_sum(alist[1:]) # 2. alist[1:]缩短列表长度向基本结束条件演进 3. 调用自身


import turtle
myTurtle = turtle.Turtle()
myWin = turtle.Screen()
def drawSpiral(myTurtle, lineLen):
    if lineLen > 0:
        myTurtle.forward(lineLen)
        myTurtle.right(90)  
        drawSpiral(myTurtle,lineLen-5)

# drawSpiral(myTurtle,100)
# myWin.exitonclick()

def tree(branchLen,t):
    if branchLen > 5:
        t.forward(branchLen)
        t.right(20)
        tree(branchLen-15,t)
        t.left(40)
        tree(branchLen-15,t)
        t.right(20)
        t.backward(branchLen)

def main():
    t = turtle.Turtle()
    myWin = turtle.Screen()
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    t.color("green")
    tree(75,t)
    myWin.exitonclick()

# main()

# 将整数转化成2-16进制表示的字符串形式
def to_str(n, base):
    convert_string = '0123456789ABCDEF'
    if n < base:
        return convert_string[n]
    else:
        return to_str(n // base, base) + convert_string[n % base]

# 将整数转化成2-16进制表示的字符串形式 栈实现
class Stack:
    def __init__(self):
        self.stack = []
    
    def is_empty(self):
        return self.stack == []

    def pop(self):
        return self.stack.pop()

    def push(self, value):
        self.stack.append(value)

    def peek(self):
        return self.stack[-1]

    def size(self):
        return len(self.stack)

stack = Stack()
def to_int(n, base):
    convert_string = '0123456789ABCDEF'
    while n > 0:
        if n < base:
            stack.push(convert_string[n])
        else:
            stack.push(convert_string[n % base])
    res = ''
    while not stack.is_empty():
        res = res + stack.pop()
    return res


# 塔
def moveTower(height,a, b, c):
    if height == 1:
        print(a, '->', c)
    else:
        moveTower(height-1, a,c,b)
        moveTower(1,a,b,c)
        moveTower(height-1, b,a,c)

moveTower(3,"A","B","C")

# 阶乘的递归实现
def fact(n):
    if n == 0:
        return 1
    else:
        return n * fact(n-1)

# 任何一个递归定义的函数（程序），都可以通过引入一个栈保存中间结果的方式，翻译为
# 一个非递归的过程。
# 任何一个包含循环的程序都可以翻译为一个不包含循环的递归函数
class SStack():
    """栈定义"""
    def __init__(self) -> None:
        self.stack = []
    
    def push(self,item):
        self.stack.append(item)
    
    def pop(self):
        return self.stack.pop()
    
    def is_empty(self):
        return self.stack == []
    
# 阶乘的非递归实现
def norec_fact(n):
    res = 1
    st = SStack()
    while n < 0:
        st.push(n)
        n -= 1
    while not st.is_empty():
        res *= st.pop()
    return res

# 简单背包问题
# 一个可放入weight的背包，有n件物品重[w1,w2,...,wn],能否从中选择若干件，使得
# 重量之和等于weight
# 假设每个物品有且仅有一件
def knap_rec(weight, wlist, n):
    if weight == 0:
        return True
    if weight < 0 or (weight > 0 and n < 1):
        return False

    # 如果选择最后一件物品，那么如果knap(weight-wn,n-1)有解
    # 其解加上最后已经物品就是knap(weight,n)的解
    if knap_rec(weight-wlist[n-1], wlist, n-1):
        print('Item ' + str(n) + ':', wlist[n-1])
        return True
    
    # 如果不选择最后一件物品,那么knap(weight, n-1)的解也是knap(weight,n)的解
    # 如果找到前者的解就找到了后者的解
    if knap_rec(weight, wlist, n-1):
        return True
    else:
        return False