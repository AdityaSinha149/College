#include <stdio.h>
#include <string.h>

#define MAX 256

void badCharHeuristic(char* pattern, int badChar[]) {
    int m = strlen(pattern);
    for (int i = 0; i < MAX; i++)
        badChar[i] = m;

    for (int i = 0; i < m - 1; i++)
        badChar[pattern[i]] = m-i-1;
}

void goodSuffixHeuristic(char* pattern, int goodSuffix[]) {
    int m = strlen(pattern);
    int i, j;

    int suffix[m + 1];
    for (i = 0; i <= m; i++) {
        goodSuffix[i] = m;
        suffix[i] = 0;
    }

    // Step 1: Compute suffix[] such that suffix[i] is the length
    // of the longest suffix of pattern[0..i] which is also a prefix
    i = m;
    j = m + 1;
    suffix[m] = m + 1;

    while (i > 0) {
        while (j <= m && pattern[i - 1] != pattern[j - 1])
            j = suffix[j];
        i--;
        j--;
        suffix[i] = j;
    }

    // Step 2: Use suffix[] to fill goodSuffix[]
    for (i = 0; i < m; i++) {
        goodSuffix[i] = m - suffix[0];
    }

    for (i = 0; i < m; i++) {
        int len = m - suffix[i];
        if (goodSuffix[len] == m - suffix[0])
            goodSuffix[len] = i;
    }
}


int boyerMooreSearch(char* text, char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    int badChar[MAX];
    int goodSuffix[m + 1];

    badCharHeuristic(pattern, badChar);
    goodSuffixHeuristic(pattern, goodSuffix);

    int s = 0;
    while (s <= n - m) {
        int j = m - 1;

        while (j >= 0 && pattern[j] == text[s + j])
            j--;

        if (j < 0)
            return s; // Match found

        int badCharShift = badChar[(unsigned char)text[s + j]] - (m - 1 - j);
        if (badCharShift < 1)
            badCharShift = 1;

        int shift = (badCharShift > goodSuffix[j + 1]) ? badCharShift : goodSuffix[j + 1];
        s += shift;
    }

    return -1;
}

int main() {
    char text[100], pattern[100];
    printf("Enter the text: ");
    scanf("%s", text);
    printf("Enter the pattern: ");
    scanf("%s", pattern);

    int pos = boyerMooreSearch(text, pattern);
    if (pos != -1)
        printf("Pattern found at position %d\n", pos);
    else
        printf("Pattern not found\n");

    return 0;
}
