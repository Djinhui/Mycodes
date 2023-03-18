# 使用栈进行括号匹配

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


def check_parens(text):
    parens = r"(){}[]"
    open_parens = "({["
    opposite = {")":"(", "}":"{", "]":"["}

    def parentheses(text):
        i, text_len = 0, len(text)
        while True:
            while i < text_len and text[i] not in parens:
                i += 1
            if i >= text_len:
                return
            yield text[i], i
            i += 1

    st = SStack()
    for pr, i in parentheses(text):
        if pr in open_parens: # 开括号入栈
            st.push(pr)
        elif st.pop() != opposite[pr]: #闭括号与栈顶开括号进行匹配
            print('Unmatching is found at', i, 'for', pr)
            return False

    if st.is_empty():
        print('All parentheses are correctly matched.')
        return True
    else:
        print('There are still some open parentheses in Stack')
        return False

# text = "((123))({})[123[]]"
text = "(((((((((((()"
print(check_parens(text))
