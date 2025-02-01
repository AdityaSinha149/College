#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int stringMatching(char* text, char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);

    if (m>n)return 0;

    for (int i=0;i<=n-m;i++) {
        int j=0;
        while(j<m && text[i+j]==pattern[j])j++;
        if (j==m)return 1;
    }
    return 0;
}

int main() {
    char text[1000];
    char pattern[1000];

    printf("Enter the text: ");
    gets(text);

    printf("Enter the pattern: ");
    gets(pattern);

    if (stringMatching(text, pattern)) {
        printf("Pattern found in text.\n");
    } else {
        printf("Pattern not found in text.\n");
    }

    return 0;
}
