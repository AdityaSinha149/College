def binSearch( l, x, low, high ):
    if low > high :
        return -1
    mid = low + (high - low)//2
    if l[mid] < x :
        return binSearch( l, x, mid + 1, high )
    elif l[mid] == x :
        return mid
    else :
        return binSearch( l, x, low, mid - 1)

l = [1,4,5,6,8,9,23]
x = 8
y = 139


print("Index of x:", binSearch(l, x, 0, len(l) - 1))
print("Index of y:", binSearch(l, y, 0, len(l) - 1))