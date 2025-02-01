#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int calculate_cost(int **cost_matrix, int *permutation, int n) {
    int total_cost = 0;
    for (int i = 0; i < n; i++) {
        total_cost += cost_matrix[i][permutation[i]];
    }
    return total_cost;
}

void generate_permutations(int *arr, int start, int end, int **cost_matrix, int n, int *min_cost, int *optimal_assignment) {
    if (start == end) {
        int cost = calculate_cost(cost_matrix, arr, n);
        if (cost < *min_cost) {
            *min_cost = cost;
            for (int i = 0; i < n; i++) {
                optimal_assignment[i] = arr[i];
            }
        }
    } else {
        for (int i = start; i <= end; i++) {
            int temp = arr[start];
            arr[start] = arr[i];
            arr[i] = temp;
            generate_permutations(arr, start + 1, end, cost_matrix, n, min_cost, optimal_assignment);
            temp = arr[start];
            arr[start] = arr[i];
            arr[i] = temp;
        }
    }
}

void brute_force_assignment(int **cost_matrix, int n) {
    int *arr = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
    
    int min_cost = INT_MAX;
    int *optimal_assignment = (int *)malloc(n * sizeof(int));
    
    generate_permutations(arr, 0, n - 1, cost_matrix, n, &min_cost, optimal_assignment);
    
    printf("Minimum Cost: %d\n", min_cost);
    printf("Optimal Assignment: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", optimal_assignment[i]);
    }
    printf("\n");
    
    free(arr);
    free(optimal_assignment);
}

int main() {
    int n = 4;
    int **cost_matrix = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        cost_matrix[i] = (int *)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            cost_matrix[i][j] = rand() % 100 + 1;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d\t",cost_matrix[i][j]);
        }
        printf("\n");
    }

    brute_force_assignment(cost_matrix, n);

    for (int i = 0; i < n; i++) {
        free(cost_matrix[i]);
    }
    free(cost_matrix);
    
    return 0;
}
