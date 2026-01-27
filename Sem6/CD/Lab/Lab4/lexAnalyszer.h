#include "preprocessing.h"

typedef struct token{
    char tokenName [50];
    char tokenValue [50];
    char tokenType [50];
    char tokenReturnType [50];
    int row,col,size;
}token;

const char *keywords[] = {
    "auto", "break", "case", "char", "const",
    "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for",
    "goto", "if", "int", "long", "register",
    "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while"
};

const char *types[] = {
    "void", "char", "short", "int", "long",
    "float", "double", "signed", "unsigned",
    "_Bool", "_Complex", "_Imaginary"
};

//Token identifying
token isKeyword(int ch, FILE *src, int *row, int *col);
token isIdentifier(int ch, FILE *src, int *row, int *col);

token isOperator(int ch, FILE *src, int *row, int *col);

token isRelationalOperator(int ch, FILE *src, int *row, int *col);
token isArithmeticOperator(int ch, FILE *src, int *row, int *col);
token isLogicalOperator(int ch, FILE *src, int *row, int *col);
token isBitwiseOperator(int ch, FILE *src, int *row, int *col);
token isConditionalOperator(int ch, int *row, int *col);
token isAssignmentOperator(int ch, FILE *src, int *row, int *col);

token isStringLiteral(int ch, FILE *src, int *row, int *col);

token isNumber(int ch, FILE *src, int *row, int *col);

//Token Server
token getNextToken(FILE *src, int *row, int *col);

//Helpers
void PrintToken(token t, FILE *dst);
void copyFile(FILE *src, FILE *dst);
void postprocess(FILE *src, FILE *dst);

int isType(char *s);

//Token Server
char type [50];

token getNextToken(FILE *src, int *row, int *col){
    token curr;
    memset(&curr, 0, sizeof(curr));

    int ch;
    ch = fgetc(src);

    while (ch != EOF) {

        if (isalpha(ch)) {
            curr = isKeyword(ch, src, row, col);
            if (curr.tokenName[0]) return curr;

            curr = isIdentifier(ch, src, row, col);
            if (curr.tokenName[0]) return curr;

        }
        
        else if (isdigit(ch)) {
            curr = isNumber(ch, src, row, col);
            return curr;
        }
        
        else if (ch == '"') {
            curr = isStringLiteral(ch, src, row, col);
            return curr;
        }

        else if (strchr("+-&|*/%<>!^?", ch)) {
            curr = isOperator(ch, src, row, col);
            return curr;
        }

        else {//Symbol
            curr.col = *col;
            curr.row = *row;
            curr.tokenName[0] = ch;
            (*col)++;
            return curr;
        }
        ch = fgetc(src);
    }
    return curr;
}

//Token identifying

token isKeyword(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));
    int c = 1;
    curr.col = *col;
    curr.row = *row;
    char word[50];
    int i = 0;
    word[i++] = (char)ch;

    while ((ch = fgetc(src)) != EOF) {
        c++;
        if (!isalpha(ch)) {
            fseek(src, -1, SEEK_CUR);
            c--;
            break;
        }
        if (i < (int)sizeof(word) - 1)
            word[i++] = ch;
    }

    word[i] = '\0';
    if ( strcmp( word, "int" ) == 0 ) 
        strcpy( type, word );

    for (int k = 0; k < (int)(sizeof(keywords)/sizeof(keywords[0])); k++) {
        if (strcmp(word, keywords[k]) == 0) {
            strcpy(curr.tokenName, word);
            *col += c;
            return curr;
        }
    }
    fseek(src, -c+1, SEEK_CUR);
    return curr;
}

token isIdentifier(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    curr.col = *col;
    curr.row = *row;
    char word[50];
    word[0] = ch;
    (*col)++;
    int i = 1;
    while (ch != EOF) {
        ch = fgetc(src);
        (*col)++;
        if(ch != '_' && !isalnum(ch)){
            fseek(src, -1, SEEK_CUR);
            (*col)--;
            strcpy(curr.tokenName, "id");
            return curr;
        }
        word[i++] = ch;
    }

    word[i] = 0;

    strcpy ( curr.tokenValue, word );
    if ( isType ( word ) )
        strcmp( type, word);
    else {
        if ( strcmp ( word, "printf" ) == 0 ||
                strcmp ( word, "scanf" ) == 0 )
            return curr;
        ch = fgetc( src );
        if( ch == '(' ) {
            strcpy ( curr.tokenReturnType, type );
            fseek ( src, -1, SEEK_CUR );
        }
        else {
            strcpy ( curr.tokenType, type );
        }
    }
    return curr;
}

token isOperator(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    curr = isRelationalOperator(ch, src, row, col);
    if (curr.tokenName[0]) return curr;

    curr = isArithmeticOperator(ch, src, row, col);
    if (curr.tokenName[0]) return curr;

    curr = isLogicalOperator(ch, src, row, col);
    if (curr.tokenName[0]) return curr;

    curr = isBitwiseOperator(ch, src, row, col);
    if (curr.tokenName[0]) return curr;

    curr = isConditionalOperator(ch, row, col);
    if (curr.tokenName[0]) return curr;

    curr = isAssignmentOperator(ch, src, row, col);
    if (curr.tokenName[0]) return curr;

    return curr;
}

token isRelationalOperator(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));
    if(ch != '=' && ch != '<' && ch != '>' && ch != '!')
        return curr;
    curr.col = *col;
    curr.row = *row;
    int prev = ch;
    ch = fgetc(src);

    if (ch == '='){
        strcpy(curr.tokenName, "relOp");
        (*col) += 2;
        return curr;
    }
    else if (prev == '<' || prev == '>'){
        strcpy(curr.tokenName, "relOp");
        (*col)++;
        return curr;
    }
    else
        fseek(src, -1, SEEK_CUR);

    return curr;
}

token isArithmeticOperator(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));
    if(ch != '+' && ch != '-' && ch != '*' && ch != '/' && ch != '%')
        return curr;
    curr.col = *col;
    curr.row = *row;
    int prev = ch;
    ch = fgetc(src);

    if (prev == '+' && ch == '+' || 
        prev == '-' && ch == '-'){
        strcpy(curr.tokenName, "ariOp");
        (*col) += 2;
        return curr;
    }
    else{
        strcpy(curr.tokenName, "ariOp");
        (*col)++;
        fseek(src, -1, SEEK_CUR);
    }

    return curr;
}

token isLogicalOperator(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));
    if(ch != '&' && ch != '|' && ch != '!')
        return curr;
    curr.col = *col;
    curr.row = *row;
    int prev = ch;
    ch = fgetc(src);

    if (prev != '!' && ch == prev){
        strcpy(curr.tokenName, "logOp");
        (*col) += 2;
        return curr;
    }
    else if (ch == '!') {
        strcpy(curr.tokenName, "logOp");
        (*col)++;

        fseek(src, -1, SEEK_CUR);

        return curr;
    }
    else
        fseek(src, -1, SEEK_CUR);

    return curr;
}

token isBitwiseOperator(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));
    if(ch != '<' && ch != '>' && ch != '|' && ch != '&' && ch != '^' && ch != '~')
        return curr;
    curr.col = *col;
    curr.row = *row;
    int prev = ch;
    ch = fgetc(src);

    if (prev == '^' || prev == '~') {
        strcpy(curr.tokenName, "bitOp");
        (*col)++;
        return curr;
    }
    else if ((ch == '<' || ch == '>') && prev == ch){
        strcpy(curr.tokenName, "bitOp");
        (*col) += 2;
        return curr;
    }
    if ((prev == '&' || prev == '|') && prev != ch) {
        strcpy(curr.tokenName, "bitOp");
        (*col)++;

        fseek(src, -1, SEEK_CUR);
        return curr;
    }
    else
        fseek(src, -1, SEEK_CUR);

    return curr;
}

token isConditionalOperator(int ch, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    if (ch == '?' || ch == ':') {
        strcpy(curr.tokenName, "condOp");
        curr.row = *row;
        curr.col = *col;
        (*col)++;
    }

    return curr;
}

token isAssignmentOperator(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    if (ch != '=' && ch != '+' && ch != '-' && ch != '*' &&
        ch != '/' && ch != '%' && ch != '<' && ch != '>' &&
        ch != '&' && ch != '|' && ch != '^')
        return curr;

    curr.col = *col;
    curr.row = *row;

    int next = fgetc(src);

    if (ch == '=' && next != '=') {
        strcpy(curr.tokenName, "assignOp");
        (*col)++;

        if (next != EOF)
            fseek(src, -1, SEEK_CUR);

        return curr;
    }

    if (next == '=') {
        strcpy(curr.tokenName, "assignOp");
        (*col) += 2;
        return curr;
    }

    if ((ch == '<' || ch == '>') && next == ch) {
        int next2 = fgetc(src);

        if (next2 == '=') {
            strcpy(curr.tokenName, "assignOp");
            (*col) += 3;
            return curr;
        }
        fseek(src, -2, SEEK_CUR);

        return curr;
    }

    if (next != EOF)
        fseek(src, -1, SEEK_CUR);

    return curr;
}

token isStringLiteral(int ch, FILE *src, int *row, int *col) {
    token curr;
    curr.col = *col;
    curr.row = *row;
    strcpy(curr.tokenName, "stringLit");
    ch = fgetc(src);
    (*col)++;
    while (ch != '"') {
        ch = fgetc(src);
        if (ch == '\n') {
            *col = 1;
            (*row)++;
        }
        else (*col)++;
    }
    (*col)++;
    return curr;
}

token isNumber(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));
    curr.col = *col;
    curr.row = *row;
    int i = 0;

    int state = 1;
    int prev;

    while (state != 4) {
        prev = ch;

        if (ch != EOF) {
            ch = fgetc(src);
            (*col)++;
        }

        if (state == 1) {
            curr.tokenName[i++] = prev;
            if (isdigit(ch)) continue;
            else if (ch == 'e' || ch =='E') state = 3;
            else if (ch == '.') state = 2;
            else state = 4;
        }
        else if (state == 2) {
            curr.tokenName[i++] = prev;
            if (isdigit(ch)) state = 5;
            else state = 4;
        }
        else if (state == 3) {
            curr.tokenName[i++] = prev;
            if (isdigit(ch)) state = 6;
            else if (ch == '+' || ch == '-') state = 7;
            else state = 4;
        }
        else if (state == 5) {
            curr.tokenName[i++] = prev;
            if (isdigit(ch)) continue;
            if (ch == 'e' || ch =='E') state = 3;
            else state = 4;
        }
        else if (state == 6) {
            curr.tokenName[i++] = prev;
            if (isdigit(ch)) continue;
            else state = 4;
        }
        else {
            curr.tokenName[i++] = prev;
            if (isdigit(ch)) state = 6;
            else state = 4;
        }
    }

    fseek(src, -1, SEEK_CUR);

    curr.tokenName[i] = '\0';
    return curr;
}

void PrintToken(token t, FILE *dst) {
    fprintf(dst, "<%s,%d,%d>", t.tokenName, t.row, t.col);
}

void copyFile(FILE *src, FILE *dst) {
    int ch;
    while((ch = fgetc(src)) != EOF) {
        putc(ch, dst);
    }
}

void postprocess(FILE *src, FILE *dst) {
    int ch;
    int newLine = 1;
    while((ch = fgetc(src)) != EOF){
        if(newLine && ch == '\n') continue;
        fputc(ch, dst);

        if (ch == '\n')
            newLine = 1;
        else
            newLine = 0;
    }
}

int isType(char *s) {
    for ( int i = 0; i < sizeof( types ); i++ )
        if ( strcmp( s, types[i] ) == 0 )
            return 1;
    return 0;
}