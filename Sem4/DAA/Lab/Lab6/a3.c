#include <stdio.h>
#include <stdlib.h>

int N;
int **image;

void takeInput() {
    printf("Enter the size of the image (N x N): ");
    scanf("%d", &N);

    image = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        image[i] = (int *)malloc(N * sizeof(int));
    }

    printf("Enter the %dx%d image matrix (use 1 for normal pixels and 0 for defective pixels):\n", N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &image[i][j]);
        }
    }
}

int containsDefect(int x1, int y1, int x2, int y2) {
    for (int i = x1; i < x2; i++) {
        for (int j = y1; j < y2; j++) {
            if (image[i][j] == 0) return 1;
        }
    }
    return 0;
}

void findDefect(int x1, int y1, int x2, int y2) {
    if (x1 >= x2 || y1 >= y2) return;

    if (containsDefect(x1, y1, x2, y2)) {
        if (x1 == x2 - 1 && y1 == y2 - 1) {
            printf("Defective pixel found at (%d, %d)\n", x1, y1);
            return;
        }

        int midX = (x1 + x2) / 2;
        int midY = (y1 + y2) / 2;

        findDefect(x1, y1, midX, midY);
        findDefect(x1, midY, midX, y2);
        findDefect(midX, y1, x2, midY);
        findDefect(midX, midY, x2, y2);
    }
}

void freeMemory() {
    for (int i = 0; i < N; i++) {
        free(image[i]);
    }
    free(image);
}

int main() {
    takeInput();
    printf("Defective regions:\n");
    findDefect(0, 0, N, N);
    freeMemory();
    return 0;
}
