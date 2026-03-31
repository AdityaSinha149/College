/*
Lex file for q3_arithmetic.y
Recognizes numbers, operators, and newline.
*/

%{
#include "q3.tab.h"
%}

%%
[0-9]+      { yylval = atoi(yytext); return NUMBER; }
\+          { return '+'; }
\-          { return '-'; }
\*          { return '*'; }
\/          { return '/'; }
\n          { return NL; }
[ \t\r]+    ;
.           { return 0; }
%%
