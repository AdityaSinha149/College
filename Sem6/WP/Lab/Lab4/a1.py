def find_smallest(arr):
    if not arr:
        return None

    smallest = arr[0]
    for x in arr:
        if x < smallest:
            smallest = x
    return smallest


if __name__ == "__main__":
    arr = [7, 3, 9, 2, 5]
    print(find_smallest(arr))
