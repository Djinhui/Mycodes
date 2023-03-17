class Solution:
    def PrintMinNumber(self,numbers):
        if not numbers:return ""
        numbers = list(map(str,numbers))
        numbers.sort(cmp=lambda a,b:cmp(a+b,b+a))
        return "".join(numbers).lstrip('0') or '0'

