def pow(x, n):
    if n == 0:
        return 1
    y = pow(x, n//2)*pow(x, n//2)
    if (n&1) == 0:
        return y
    else:
        return x*y

print(pow(5, 3))