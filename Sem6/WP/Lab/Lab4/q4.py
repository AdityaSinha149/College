def takeNotTake(arr, idx, currList, globalList):
    if idx == len(arr):
        globalList.append(currList.copy())
        return

    currList.append(arr[idx])
    takeNotTake(arr, idx + 1, currList, globalList)

    currList.pop()

    takeNotTake(arr, idx + 1, currList, globalList)


arr = [4, 5, 6]
globalList = []

takeNotTake(arr, 0, [], globalList)

print(globalList[::-1])
