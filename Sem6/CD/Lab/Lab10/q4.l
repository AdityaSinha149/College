%{
#include "q4.tab.h"
#include <stdlib.h>
%}

%%

[0-9]+      { yylval = atoi(yytext); return NUM; }

\+          { return '+'; }
\-          { return '-'; }
\*          { return '*'; }
\/          { return '/'; }
\^          { return '^'; }

\n          { return NL; }

[ \t\r]+    ;   /* ignore whitespace */

.           { return yytext[0]; }

%%