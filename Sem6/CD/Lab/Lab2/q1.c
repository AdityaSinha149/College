#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(){
    FILE *src,*dst;
    char file[50];
    
    printf("Enter file to remove spaces from : ");
    fgets(file, 50, stdin);
    file[strcspn(file, "\n")] = '\0';
    src = fopen(file, "r");
    if(src == NULL){
        perror("error ");
        return 1;
    }

    printf("Enter destination file : ");
    fgets(file, 50, stdin);
    file[strcspn(file, "\n")] = '\0';
    dst = fopen(file, "w");
    if(dst == NULL){
        perror("error ");
        fclose(src);
        return 1;
    }

    int ch;
    int state = 0;
    while ((ch = fgetc(src)) != EOF) {
        if(ch == ' '){
            if(state == 0) fputc(ch, dst);
            state = 1;
        }
        else state = 0;
        if(state == 0)fputc(ch, dst);
    }

    fclose(src);
    fclose(dst);

    return 0;
}