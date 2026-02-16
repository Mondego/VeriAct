
/* astgen.l -- FLEX definition of ast generation program */

/* This program generates the boilerplate for fields of AST nodes */

%{

#include <stdio.h>
#include <assert.h>
#include "astactions.h"

#define ECHO astecho(yytext, yyleng)
/* #define ECHO do { \
	fwrite( yytext, yyleng, 1, yyout ); fflush(yyout); \
  } while(0) */

static int lineno;

#define PUSH(m) do { modes_push(m); BEGIN(m); } while (0)
#define POP() do { modes_pop(); BEGIN(modes_top()); } while (0)
#define GOTO(m) do { modes_goto(m); BEGIN(m); } while(0)

/* Local procedures */
static void modes_push(int m);
static void modes_pop();
static void modes_goto(int m);
static int modes_top();
static void modes_init();
static int in_comment();

static int nesting_depth;

#define YY_DECL \
  int astgen(FILE *yyin_i)

%}

WSP [ \t]
FSP [ \t\n]
NAME ([a-zA-Z][a-zA-Z0-9_]*)
PNAME ([a-zA-Z][a-zA-Z0-9_.]*)

%option noyywrap

%s INIT
%s ABOVECLASS
%s LOOKINGFORNAME
%s HAVENAME
%s LOOKINGFORSUPER
%s INCLASS
%s COMMENT

%%


 yyrestart(yyin_i);
 init();
 nesting_depth = 0;
 modes_init();
 PUSH(INIT);

[\n]+ { lineno += yyleng; ECHO; }
{WSP}* { ECHO; }
"//".* { ECHO; }
"/*" { ECHO; if (! in_comment()) PUSH(COMMENT); }
<<EOF>> { endastfile(); return 0; }

<COMMENT>"*/" { ECHO; POP(); }

<INIT>^{WSP}*"//#"{WSP}*"VisitorRoot"{WSP}.*\n { visitorroot(yytext, yyleng); ECHO; }
<INIT>^{WSP}*"//#"{WSP}*"TagBase"{WSP}.*\n { tagbase(yytext, yyleng); ECHO; }
<INIT>^{WSP}*"//#"{WSP}*"EndHeader".*\n { endheader(); GOTO(ABOVECLASS); }

<ABOVECLASS>"abstract" { abstract(); ECHO; }
<ABOVECLASS>"class" { ECHO; PUSH(LOOKINGFORNAME); }

<LOOKINGFORNAME>{NAME} { classname(yytext, yyleng); ECHO; GOTO(HAVENAME); }
<LOOKINGFORNAME>. { ECHO; YY_FATAL_ERROR("Can't find class name."); }

<HAVENAME>"{" { nesting_depth++; beginclass(); ECHO; GOTO(INCLASS); }
<HAVENAME>"extends" { ECHO; PUSH(LOOKINGFORSUPER); }

<LOOKINGFORSUPER>{NAME} { supername(yytext, yyleng); ECHO; POP(); }
<LOOKINGFORSUPER>. { ECHO; YY_FATAL_ERROR("Can't find superclass name."); }


<INCLASS>"{" { nesting_depth++; ECHO; }
<INCLASS>"}"{WSP}* {
  if (--nesting_depth == 0) { endclass(yytext, yyleng); POP(); } else ECHO;
 }
<INCLASS>"}"{WSP}*\n {
  lineno++;
  if (--nesting_depth == 0) { endclass(yytext, yyleng); POP(); } else ECHO;
 }
<INCLASS>^{WSP}*"//#".*\n { expand(yytext, yyleng); }
<INCLASS>\"([^"\n]|\\\")*\" { ECHO; /* Strings */ }
<INCLASS>\"([^"\n]|\\\")*\n { ECHO; lineno++; /* Unterminated string */ }
<INCLASS>"'"([^'\n]|\\"'")*"'" { ECHO; /* Character */ }
<INCLASS>"'"([^'\n]|\\"'")*\n { lineno++; ECHO; /* Unterminated character */ }
<INCLASS>([^"'{}\n/]|"/"[^/*])* { ECHO; }

. { ECHO; }

%%

int main(int argc, char **argv)
{
  /* jrk: return added; is it necessary/correct? */
  return astgen(stdin);
}

static int modes_size, modes_next_slot;
static int *modes = NULL;

static void modes_init()
{
  if (DEBUGPRINT) fprintf(stderr, "void modes_init()\n");
  if (modes == NULL) {
	modes_size = 8;
	modes = (int *)malloc(modes_size * sizeof(int));
	if (modes == NULL) YY_FATAL_ERROR("out of memory");
  }
  modes[0] = INITIAL;
  modes_next_slot = 1;
}

static void modes_push(int m)
{
  if (DEBUGPRINT) fprintf(stderr, "Pushing %d\n", m);
  if (modes_next_slot >= modes_size) {
	modes_size *= 2;
	modes = (int *)realloc(modes, modes_size*sizeof(int));
	if (modes == NULL) YY_FATAL_ERROR("out of memory");
  }
  modes[modes_next_slot++] = m;
}

static void modes_pop()
{
  assert(modes_next_slot > 1);
  if (DEBUGPRINT)
	fprintf(stderr, "Popping %d, re-entering %d\n", modes[modes_next_slot-1],
		 modes[modes_next_slot-2]);
  modes_next_slot--;
}

static void modes_goto(int m)
{
  assert(modes_next_slot > 0);
  if (DEBUGPRINT) fprintf(stderr, "Goto %d\n", m); 
  modes[modes_next_slot-1] = m;
}

static int modes_top()
{ return(modes[modes_next_slot-1]); }

static int in_comment()
{
  int i;
  for(i = 0; i < modes_next_slot; i++)
	if (modes[i] == COMMENT) return 1;
  return 0;
}

