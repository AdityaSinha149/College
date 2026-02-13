#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

static void preprocess(FILE *src, FILE *dst) {
    int ch;
    // Find <?php opening tag
    while((ch = fgetc(src)) != EOF) {
        if(ch == '<') {
            int next = fgetc(src);
            if(next == '?') {
                // Skip until newline or whitespace
                while((ch = fgetc(src)) != EOF && ch != '\n' && ch != ' ' && ch != '\t');
                if(ch == '\n') fputc(ch, dst);
                break;
            }
            else {
                fseek(src, -1, SEEK_CUR);
            }
        }
    }
    
    // Copy content until ?> closing tag
    while((ch = fgetc(src)) != EOF){
        if( ch == '?') {
            int next = fgetc(src);
            if(next == '>') break;
            else {
                fseek(src, -1, SEEK_CUR);
                putc(ch, dst);
            }
        }
        else if(ch == '/') {
            int next = fgetc(src);
            if(next == '/') {
                // Single-line comment - skip entire line
                while((ch = fgetc(src)) != EOF && ch != '\n');
                if(ch == '\n') putc('\n', dst);  // Output newline to maintain line count
            }
            else if(next == '*') {
                // Multi-line comment - skip until */ and skip those lines
                int prev = 0;
                putc('\n', dst);  // Output newline for the line with comment start
                while((ch = fgetc(src)) != EOF) {
                    if(ch == '\n') putc('\n', dst);
                    if(prev == '*' && ch == '/') break;
                    prev = ch;
                }
            }
            else {
                putc('/', dst);
                if(next != EOF) {
                    fseek(src, -1, SEEK_CUR);
                }
            }
        }
        else {
            putc(ch, dst);
        }
    }
}

#endif
