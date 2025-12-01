#include "banker.h"

int main() {
    calculate_need();
    printf("\nNeed Matrix:\n");
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++)
            printf("%d ", need[i][j]);
        printf("\n");
    }

    if (is_safe())
        printf("\nSystem is in a safe state.\n");
    else
        printf("\nSystem is NOT in a safe state.\n");

    return 0;
}
