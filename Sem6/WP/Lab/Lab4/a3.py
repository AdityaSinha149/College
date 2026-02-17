def multiply_matrices(A, B):
    r1, c1 = len(A), len(A[0])
    r2, c2 = len(B), len(B[0])

    if c1 != r2:
        raise ValueError("Matrix multiplication not possible")

    result = [[0 for _ in range(c2)] for _ in range(r1)]

    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result[i][j] += A[i][k] * B[k][j]

    return result


if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print(multiply_matrices(A, B))
