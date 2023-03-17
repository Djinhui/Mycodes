# 字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是
import re
class Solution:
    def isNumeric(self,s):
        return re.match(r"^[\+\-]?[0-9]*(\.[0-9]*)?([eE][\+\-]?[0-9]+)?$", s)

# def isNumeric(s):
#     AllowDot = True
#     AllowE = True
#     for i in range(len(s)):
#         if s[i] in '+-' and (i==0 or s[i-1] in 'eE') and i<len(s)-1:
#             continue
#         elif AllowDot and s[i] == '.': 
#             AllowDot = False
#             if i >= len(s)-1 or s[i+1] not in '0123456789':
#                 return False
#         elif AllowE and s[i] == 'eE':
#             AllowE = False
#             AllowDot = False
#             if i >= len(s)-1 or s[i+1] not in '0123456789+-':
#                 return False
#         elif s[i] not in '0123456789':
#             return False
#     return True
