# 给定两个字符串，请编写代码，输出最长公共子串（Longest Common Substring），
# 是指两个字符串中的最长的公共子串，要求子串一定是连续
# 伪DP算法
s1,s2 = input().split(',')
if len(s1) < len(s2):       # 保证长串是 s1
    s1,s2 = s2,s1
dp = 0                      # 最大公共子串的长度
for i in range(len(s1)):
    if s1[i-dp:i+1] in s2:
        dp += 1
print(dp)



# 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
class Solution:
    def lengthOfLongestSubstring(self,s):
        if not s:return 0
        lookup = [] # 保存窗口内字符串
        n = len(s)
        max_len = 0 # 最长子串长度
        cur_len = 0 # 当前窗口长度
        for i in range(n):
            val = s[i]
            if not val in lookup:
                lookup.append(val)
                cur_len = cur_len + 1 
            else:
                index = lookup.index(val)
                # 移除该位置及以前的字符
                lookup = lookup[index+1:]
                lookup.append(val)
                cur_len = len(lookup)
            if cur_len > max_len:
                max_len = cur_len
        return max_len