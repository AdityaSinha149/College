class TwoSum:
    def __init__(self, nums, target):
        # instance variables
        self.nums = nums
        self.target = target

    def compute(self):
        numToIdx = {}

        for idx, num in enumerate(self.nums, 1):
            complement = self.target - num

            if complement in numToIdx:
                return [numToIdx[complement], idx]

            numToIdx[num] = idx

        return -1

nums = [10, 20, 10, 40, 50, 60, 70]
target = 50

obj = TwoSum(nums, target)
result = obj.compute()

print("Indices of elements whose sum is", target, ":", result)
