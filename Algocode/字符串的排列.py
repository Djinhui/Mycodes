# 输入一个字符串,按字典序打印出该字符串中字符的所有排列。
# 例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
class Solution:
    def Permutation(self,ss):
        if len(ss) <= 1: # ss = 'abc'
            return ss
        res = set()
        # 遍历字符串，固定第一个元素，依次递归
        for i in range(len(ss)):
            for j in self.Permutation(ss[:i]+ss[i+1:]):
                res.add(ss[i] + j)
        return sorted(res)

s = Solution()
print(s.Permutation('abc'))