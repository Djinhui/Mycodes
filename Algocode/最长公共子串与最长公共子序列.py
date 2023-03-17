# 使用矩阵来记录两个子串之间各个字符之间的对应关系
# 最长公共子串：矩阵中数值最大的就是最长子串的长度。若对应位置字符相同，则从c[i][j] = c[i-1][j-1]+1,若不同则c[i][j]=0
def longSubStr(str1,str2):
    len1 = len(str1)
    len2 = len(str2)
    longest,start1,start2 = 0,0,0
    c = [[0 for i in range(len2+1)] for j in range(len1+1)] # 最左和最上加列/行0
    for i in range(len1+1):
        for j in range(len2+1):
            if i == 0 or j == 0:
                c[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                c[i][j] = c[i-1][j-1]+1
            else:
                c[i][j] = 0
            if longest < c[i][j]:
                longest = c[i][j]
                start1 = i-longest
                start2 = j-longest
    return str1[start1:start1+longest],longest

# 最长子序列：若对应位置字符相同，则从c[i][j] = c[i-1][j-1]+1,若不同则c[i][j]=max(c[i][j-1],c[i-1][j])

def printLCS(flag,str1,len1,len2): # 用于打印最长子序列
    if len1==0 or len2==0:
        return
    if flag[len1][len2] == 'OK':
        printLCS(flag,str,len1-1,len2-1)
        print(a[len1-1])
    elif flag[len1][len2] == 'Left':
        printLCS(flag,str1,len1,len2-1)
    else:
        printLCS(flag,str1,len1-1,len2)

def LongSubSeq(str1,str2):
    len1 = len(str1)
    len2 = len(str2)
    longest = 0
    c = [[0 for i in range(len2+1)] for j in range(len1+1)]
    flag = [[0 for i in range(len2+1)] for j in range(len1+1)] # 用于打印
    for i in range(len1+1):
        for j in range(len2+1):
            if i == 0 or j == 0:
                c[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                c[i][j] = c[i-1][j-1]+1
                flag[i][j] = 'OK' 
                longest = max(longest,c[i][j])
            elif c[i][j-1] >c[i-1][j]:
                c[i][j] = c[i][j-1]
                flag[i][j] = 'Left'
            else:
                c[i][j] = c[i-1][j]
                flag[i][j] = 'UP'

    printLCS(flag,str1,len1,len2)
    return longest


a='ABCBDAB'  
b='BDCABA'  
print(longSubSeq(a,b))