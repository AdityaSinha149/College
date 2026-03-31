/*
Lex file for q1_declaration.y
Recognizes int, float, char, identifiers, and semicolon.
*/

%{
#include "q1.tab.h"
%}

%%
"," return COMMA;
int     { return INT; }
float   { return FLOAT; }
char    { return CHAR; }
[a-zA-Z_][a-zA-Z0-9_]* { return ID; }
;       { return SEMI; }
[ \t\r]+   ;
\n      ;
.       { return 0; }
%%
