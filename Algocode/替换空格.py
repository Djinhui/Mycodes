# 将一个字符串中的每个空格替换成“%20”。
# 例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        return s.replace(' ','%20')

print('%20'.join(['we','are','happy']))

class Solution2:
    def replaceSpace(self, s):
        