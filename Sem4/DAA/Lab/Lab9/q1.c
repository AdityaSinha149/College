#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int op=0;

void makeShift(char*text,char*pattern,int shift[]){
    int m=strlen(pattern);
    int n=strlen(text);

    for(int i=0;i<256;i++)
        shift[i]=m;

    for(int i=0;i<m-1;i++)
        shift[pattern[i]]=m-i-1;
    //text never used can be removed
}

int stringMatching(char*text,char*pattern){
    int shift[256];
    int m=strlen(pattern);
    int n=strlen(text);
    makeShift(text,pattern,shift);

    int i=m-1;
    while(i<n){
        int k=0;
        while(k<m && pattern[m-1-k] == text[i-k]) k++;

        if(k==m)return i-m+1;
        i+=shift[text[i]];
    }
    return -1;
}

int main(){
    char text[100],pattern[100];
    printf("Enter the text: ");
    scanf("%s",text);
    printf("Enter the pattern: ");
    scanf("%s",pattern);

    int pos=stringMatching(text,pattern);
    if(pos!=-1)
        printf("Pattern found at position %d\n",pos);
    else
        printf("Pattern not found\n");

    return 0;
}