%{
#include "q2.tab.h"
#include <stdlib.h>
%}

%%

"if"            { return IF; }
"else"          { return ELSE; }

"=="|"!="|"<="|">="|"<"|">"   { return RELOP; }

"="             { return ASSIGN; }
";"             { return SEMI; }

"("             { return LPAREN; }
")"             { return RPAREN; }

[0-9]+          { yylval.num = atoi(yytext); return NUMBER; }

[a-zA-Z_][a-zA-Z0-9_]*  { return ID; }

[ \t\r]+        ;   /* ignore whitespace */
\n              ;   /* ignore newline */

.               { return yytext[0]; }

%%