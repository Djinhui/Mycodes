# 变位词
# 如果一个字符串是另一个字符串的重新排列组合，那么这两个字符串互为变位词。
# 比如，”heart”与”earth”互为变位词，”python”与”typhon”也互为变位词

#  1. 检查标记法  O(n^2)
def anagram_solution(s1,s2):
    a_list = list(s2)

    pos1 = 0
    still_ok = True

    while pos1 < len(s1) and still_ok:
        pos2 = 0
        found = False
        while pos2 < len(a_list) and not found:
            if s1[pos1] == a_list[pos2]:
                found = True
            else:
                pos2 += 1

        if found:
            a_list[pos2] = None  # 如果找到，就用 None 代替以示标记
        else:
            still_ok = False
        pos1 += 1

    return still_ok

# 2. 排序比较法 O(nlogn)或O(n^2)，却决于排序算法
def anagram_solution2(s1,s2):
    a_list1 = list(s1)
    a_list2 = list(s2)

    a_list1.sort()
    a_list2.sort()

    pos = 0
    matches = True
    while pos < len(s1) and matches:
        if a_list1[pos] == a_list2[pos]:
            pos += 1
        else:
            matches = False

    return matches

# 3. 暴力排序法 O(n!) 构造由 s1 中所有字符组成的所有可能的字符串的列表，并检查 s2 是否在列表中

# 4. 计数比较法 O(n),牺牲空间换时间
def anagram_solution4(s1,s2):
    # 额外的空间来存储两个计数器列表
    C1 = [0] * 26 # 26个字母标志位
    C2 = [0] * 26

    for i in range(len(s1)):
        pos = ord(s1[i]) - ord('a')
        C1[pos] = C1[pos] + 1

    for i in range(len(s2)):
        pos = ord(s2[i]) - ord('a')
        C2[pos] = C2[pos] + 1

    j = 0
    still_ok = True
    while j < 26 and still_ok:
        if C1[j] == C2[j]:
            j += 1
        else:
            still_ok = False

    return still_ok
