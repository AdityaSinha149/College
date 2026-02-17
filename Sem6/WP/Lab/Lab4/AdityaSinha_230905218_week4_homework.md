# Additional Assignment — Python Programming

**Name:** Aditya Sinha  
**Reg. No:** 230905218  
**Class & Section:** CSE-A1  
**Roll No:** 27

---

## 1. Write a python program to select smallest element from a list in an expected linear time.

### Code :
```python
def find_smallest(arr):
    if not arr:
        return None

    smallest = arr[0]
    for x in arr:
        if x < smallest:
            smallest = x
    return smallest


arr = [7, 3, 9, 2, 5]
print(find_smallest(arr))
```

### Output:
```txt
2
```

---

## 2. Write a python program to implement bubble sort.

### Code :
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


arr = [5, 1, 4, 2, 8]
print(bubble_sort(arr))
```

### Output:
```txt
[1, 2, 4, 5, 8]
```

---

## 3. Write a python program to multiply two matrices.

### Code :
```python
def multiply_matrices(A, B):
    r1, c1 = len(A), len(A[0])
    r2, c2 = len(B), len(B[0])

    if c1 != r2:
        print("Matrix multiplication not possible")
        return

    result = [[0 for _ in range(c2)] for _ in range(r1)]

    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result[i][j] += A[i][k] * B[k][j]

    return result


A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print(multiply_matrices(A, B))
```

### Output:
```txt
[[19, 22], [43, 50]]
```

---

## 4. Write a Python class to find validity of a string of parentheses.

### Code :
```python
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


vp = ValidParentheses()
print(vp.is_valid("()[]{}"))
print(vp.is_valid("({[)]}"))
```

### Output:
```txt
True
False
```

---

## 5. Write a Python class to reverse a string word by word.

### Code :
```python
class ReverseWords:
    def reverse_words(self, s):
        words = s.split()
        return " ".join(reversed(words))


rw = ReverseWords()
print(rw.reverse_words("Hello World from Python"))
```

### Output:
```txt
Python from World Hello
```

---

## 6. Write a Python class named Circle constructed by a radius and two methods to compute area and perimeter.

### Code :
```python

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius

    def perimeter(self):
        return 2 * 3.14 * self.radius


c = Circle(5)
print("Area:", c.area())
print("Perimeter:", c.perimeter())
```

### Output:
```txt
Area: 78.53981633974483
Perimeter: 31.41592653589793
```
