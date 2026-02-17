class ValidParentheses:
    def is_valid(self, s):
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}

        for ch in s:
            if ch in mapping:
                if not stack or stack.pop() != mapping[ch]:
                    return False
            else:
                stack.append(ch)

        return len(stack) == 0


if __name__ == "__main__":
    vp = ValidParentheses()
    print(vp.is_valid("()[]{}"))
    print(vp.is_valid("({[)]}"))
