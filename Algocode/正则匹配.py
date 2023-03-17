class Solution:
    def match(self,s,pattern):
        # s = 'avfbd'
        # pattern = 'a.fc*bd'
        m,n = len(s),len(pattern)
        match = [[False for j in range(n+1)] for i in range(m+1)]
        match[0][0] = True
        for j in range(n+1):
            if j > 1 and pattern[j-1] == '*':
                match[0][j]=match[0][j-2]
            for i in range(1,m+1):
                for j in range(1,n+1):
                    if s[i-1] == pattern[j-1] or pattern[j-1] =='.':
                        match[i][j]=match[i-1][j-1]
                    elif j > 1 and pattern[j-1] == '*':
                        if s[i-1] == pattern[j-2] or pattern[j-2] == '.':
                            match[i][j]=match[i-1][j] or match[i][j-2]
                        else:
                            match[i][j]=match[i][j-2]
        return match[m][n]

        