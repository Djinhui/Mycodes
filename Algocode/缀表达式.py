# -*- coding=utf-8 -*-
# 使用栈进行后缀表达式的计算
"""
后缀表达式的计算机求值：
A.从左至右扫描表达式；
B.遇到数字时,将数字压入堆栈,遇到运算符时,弹出栈顶的两个数,用运算符对它们做相应的计算:次顶元素 op 栈顶元素,并将结果入栈；
C.重复上述过程直到表达式最右端，最后运算得出的值即为表达式的结果
"""
class StackUnderflow(ValueError):
    pass

class SStack:
    def __init__(self) -> None:
        self._elems = []

    def is_empty(self):
        return self._elems == []

    def top(self):
        if self._elems == []:
            raise StackUnderflow('in SStack.top()')
        else:
            return self._elems[-1]

    def push(self, elem):
        self._elems.append(elem)

    def pop(self):
        if self._elems == []:
            raise StackUnderflow('in SStack.pop()')
        return self._elems.pop()

class ESStack(SStack):
    def depth(self):
        return len(self._elems)

def suf_exp_evaluator(expression):
    # experssion = '123+4×+5-' 
    operators = "+-/*" # 二元运算符
    st = ESStack()

    for x in expression:
        if x not in operators: # 遇到数字,入栈
            st.push(float(x))
            continue

        if st.depth() < 2: # x为运算符时，栈内必须有两个以上数字
            raise SyntaxError('Short of operand(s)')

        a = st.pop()
        b = st.pop()
        if x == '+':
            c = b + a
        elif x == '-':
            c = b - a
        elif x == '*':
            c = b * a
        elif x == '/':
            c = b / a
        else:
            break

        st.push(c) # 运算结果入栈

    if st.depth() == 1: # 最终栈内只有一个元素即最终结果
        return st.pop()
    else:
        raise SyntaxError('Extra operand(s).')