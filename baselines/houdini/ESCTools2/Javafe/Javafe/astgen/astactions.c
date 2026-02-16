/* Copyright 2000, 2001, Compaq Computer Corporation */


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "astutil.h"
#include "astoutput.h"
#include "astactions.h"

extern void *memcpy(void *dst, const void *src, size_t len);

/*
   Constants defining the various states
*/

#define UNINITIALIZED 0
#define INIT 1
#define ABOVECLASS 2
#define SUPERLESS 3
#define SUPERFULL 4
#define INCLASS 5
#define DONE 6


/*
   State of this module
*/

static int state = UNINITIALIZED;

static const char *tagBase;
static const char *visitorRoot;
static char *headerText;
static int headerTextAlloc, headerTextLen;
static ClassListNode *classes;
static int classesCount;
static FILE *currentOutputFile;
static Class *currentClass;

static char *tmpname = "ASTGENtmp";

/* static FILE* visitorOutputFile; */

/* Invariants:

P0: currentClass == currentOutputFile == NULL
    and file named "tmpname" can be created
P1: each member of "classes" is valid and "classesCount" = len of "classes"
    and "headerText != NULL"
    and "headerTextLen <= headerTextAlloc ==" allocated size of "headerText"
P2: "classCount > 0" and "currentClass" points to last element of classes
    and "currentOutputFile" points to a file open for writing
P3: "currentClass->name == NULL"
    and "tmpname" is path to "currentOutputFile"
P4: "currentClass->name != NULL"
    and "tmpname" is path to "currentOutputFile"

state = INIT ==> P0, P1, "classesCount == 0", headerTextLen == 0,
state = ABOVECLASS ==> P1, P2, P3
state = SUPERLESS ==> P1, P2, P4
state = SUPERFULL ==> P1, P2, P4
state = INCLASS ==> P1, P2, P4
state = DONE ==> P0, P1,
                 and "classes" is the list, in order, of the class info
                     read by the state machine
*/

static void check(int expectedState)
{
  assert(headerText); assert(headerTextAlloc > 0); assert(headerTextLen >= 0);
  assert(classesCount >= 0);

  assert(state == expectedState);
  switch(state) {
  case INIT:
    assert(! currentClass); assert(! currentOutputFile);
    assert(classesCount == 0); assert(! classes);
    break;
  case DONE:
    assert(classesCount == 0 || classes);
    assert(! currentClass); assert(! currentOutputFile);
    break;
  case ABOVECLASS:
    assert(classesCount > 0); assert(classes);
    assert(currentClass); assert(currentOutputFile);
    assert(! currentClass->name);
    break;
  case SUPERLESS:
    assert(classesCount > 0); assert(classes);
    assert(currentClass); assert(currentOutputFile);
    assert(currentClass->name);
    break;
  case SUPERFULL:
    assert(classesCount > 0); assert(classes);
    assert(currentClass); assert(currentOutputFile);
    assert(currentClass->name);
    break;
  case INCLASS:
    assert(classesCount > 0); assert(classes);
    assert(currentClass); assert(currentOutputFile);
    assert(currentClass->name);
    break;
  default: assert(0);
  }
}

static void freeResources()
{
  while(classes) {
    DirectiveListNode *d = classes->c->directives;
    while(d) {
      DirectiveListNode *n = d->next;
      free(d->text);
      if (d->tag == FIELDDIRECTIVE) {
	free(d->i.f.type);
	free(d->i.f.name);
      } else if (d->tag == MAKERSPECDIRECTIVE) {
	free(d->i.ms.pragma);
      }
      free(d);
      d = n;
    }
    {
      ClassListNode *cl = classes;
      classes = classes->next;
      free(cl->c);
      free(cl);
    }
  }
  if (headerText) { free(headerText); headerText = NULL; }
  if (currentOutputFile)
    { fclose(currentOutputFile); currentOutputFile = NULL; }
}





/*
   Adding to the header text buffer
*/

static void addHeaderText(const char *text, int len)
{
  while (headerTextLen + len >= headerTextAlloc) {
    headerTextAlloc *= 2;
    headerText = (char *)realloc(headerText, headerTextAlloc);
  }
  memcpy(headerText + headerTextLen, text, len);
  headerTextLen += len;
}



/*
   Opening files
*/

static void openOutputFile()
{
  assert(headerText);
  assert(state == INIT || (state == INCLASS && currentClass->name));
  if (state == INCLASS) {
    char *fname = catstr(currentClass->name, ".java");
    fclose(currentOutputFile);
    unlink(fname); /* In case it already exists */
    link(tmpname, fname);
    unlink(tmpname);
    free(fname);
  }
  if ((currentOutputFile = fopen(tmpname, "w")) == NULL) {
    perror("astgen output file");
    assert(0);
  }
  outputStartOfFile(currentOutputFile, headerText, headerTextLen);
}



/*
   Operations on "classes" (the class list)
*/

static void addClass()
{
  ClassListNode **cl = &classes;
  assert(state == INIT || state == INCLASS);
  while (*cl) cl = &((*cl)->next);
  *cl = (ClassListNode *)malloc(sizeof(ClassListNode));
  (*cl)->next = NULL;
  (*cl)->c = currentClass = (Class *)calloc(1, sizeof(Class));
  classesCount++;
}

static void printClasses(FILE *out)
{
  ClassListNode *cl;
  
  if (headerText)
    fwrite(headerText, headerTextLen, 1, out);

  for(cl = classes; cl; cl = cl->next) {
    DirectiveListNode *d;

    if (cl->c->superclass)
      fprintf(out, "\nclass %s extends %s\n",
	      cl->c->name, cl->c->superclass->name);
    else fprintf(out, "\nclass %s\n", cl->c->name);

    for(d = cl->c->directives; d; d = d->next)
      switch (d->tag) {
      case FIELDDIRECTIVE:
	fprintf(out, "  Directive: %s%s%s\n",
		d->i.f.type, (d->i.f.sequence ? " * " : " "), d->i.f.name);
	break;
      case NOMAKERDIRECTIVE:
	fprintf(out, "  Directive: %s\n", NOMAKERTOK);
	break;
      case MANUALTAGDIRECTIVE:
	fprintf(out, "  Directive: %s\n", MANUALTAGTOK);
	break;
      case POSTMAKECALLDIRECTIVE:
	fprintf(out, "  Directive: %s\n", POSTMAKECALLTOK);
	break;
      case POSTCHECKCALLDIRECTIVE:
	fprintf(out, "  Directive: %s\n", POSTCHECKCALLTOK);
	break;
      case MAKERSPECDIRECTIVE:
	fprintf(out, "  Directive: %s %s\n", MAKERSPECTOK, d->i.ms.pragma);
	break;
      default: assert(0);
      }
  }
  fflush(out);
}



/*
   State transition routines
*/


void init()
{
  if (DEBUGPRINT) fprintf(stderr, "void init()\n");

  if (! headerText) {
    headerTextAlloc = 512;
    headerText = (char *)malloc(headerTextAlloc);
  }
  state = INIT;
  headerTextLen = 0;
  tagBase = "0";
  visitorRoot = NULL;
  classes = NULL;
  classesCount = 0;
  currentOutputFile = NULL;
  currentClass = NULL;
  check(INIT);
}

void visitorroot(const char *text, int len)
{
  const char *end = text + len - 1;
  assert(*end == '\n');
  while(text < end && isspace(*text)) text++;
  assert(text+3 < end && *text++ == '/' && *text++ == '/' && *text++ == '#');
  while(text < end && isspace(*text)) text++;
  while(text < end && isalpha(*text)) text++;
  assert(text+1 < end);
  assert(isspace(*text++));
  visitorRoot = mkstr(text, end - text);
}

void tagbase(const char *text, int len)
{
  const char *end = text + len - 1;
  assert(*end == '\n');
  while(text < end && isspace(*text)) text++;
  assert(text+3 < end && *text++ == '/' && *text++ == '/' && *text++ == '#');
  while(text < end && isspace(*text)) text++;
  while(text < end && isalpha(*text)) text++;
  assert(text+1 < end);
  assert(isspace(*text++));
  while(text < end && isspace(*text)) text++;
  tagBase = mkstr(text, end - text);
}

void endheader()
{
  check(INIT);
  if (DEBUGPRINT) fprintf(stderr, "void endheader()\n");
  openOutputFile(); /* Important that state == INIT here! */
  addClass();
  state = ABOVECLASS;
  check(ABOVECLASS);
}

void abstract()
{
  if (DEBUGPRINT) fprintf(stderr, "void abstract()\n");
  check(ABOVECLASS);
  assert(! currentClass->abstract);
  currentClass->abstract = 1;
}

void classname(const char *text, int len)
{
  char *s = mkstr(text, len);

  if (DEBUGPRINT) fprintf(stderr, "void classname(%s, %d)\n", s, len);

  check(ABOVECLASS);
  assert(! findClass(classes, s));
  currentClass->name = s;

  state = SUPERLESS;
  check(SUPERLESS);
}

void supername(const char *text, int len)
{
  Class *sc;
  char *s = mktmpstr(text, len);

  check(SUPERLESS);
  if (DEBUGPRINT) fprintf(stderr, "void supername(%s, %d)\n", s, len);
  assert(currentClass && !currentClass->superclass);
  sc = currentClass->superclass = findClass(classes, s);
  while(sc != NULL) { assert(sc != currentClass); sc = sc->superclass; }
  state = SUPERFULL;
  check(SUPERFULL);
}

void beginclass()
{
  assert(state == SUPERLESS || state == SUPERFULL);
  check(state);
  if (DEBUGPRINT) fprintf(stderr, "void beginclass()\n");
  state = INCLASS;
  check(INCLASS);
}

void endclass(const char *text, int len)
{
  check(INCLASS);
  if (DEBUGPRINT) fprintf(stderr, "void endclass()\n");

  /* Put appropriate default visitor method in Visitor.java */
/*
  if( currentClass->superclass != NULL ) { 
    fprintf( visitorOutputFile, "  public void visit%s(%s x) {\n",
            currentClass->name, currentClass->name );
    fprintf( visitorOutputFile, "    visit%s(x);\n",
            currentClass->superclass->name );
    fprintf( visitorOutputFile, "  }\n\n");
  } else {
    fprintf( visitorOutputFile, "  public abstract void visit%s(%s x);\n\n",
            currentClass->name, currentClass->name );
  }
*/
#ifndef TESTPARSING
  outputEndClass(currentOutputFile, currentClass, text, len, visitorRoot);
#else
  fputc('}', currentOutputFile);

#endif
  openOutputFile(); /* Important that state == INCLASS here */
  addClass();
  state = ABOVECLASS;
  check(ABOVECLASS);
}

void endastfile()
{
  assert(state == ABOVECLASS || state == INIT);
  check(state);
  if (DEBUGPRINT) fprintf(stderr, "void endastfile()\n");
  if (DEBUGPRINT) printClasses(stderr);

  if (state == ABOVECLASS) {
    fclose(currentOutputFile);
    currentOutputFile = NULL;
    unlink(tmpname);
    if (classes->next) {
      ClassListNode *cl = classes;
      while(cl->next->c != currentClass) cl = cl->next;
      cl->next = NULL; classesCount--;
      free(currentClass);
      currentClass = NULL;
    } else {
      assert(classesCount == 1 && classes->c == currentClass);
      classesCount = 0;
      free(classes);
      classes = NULL;
    }
    currentClass = NULL;
  } else currentClass = NULL;

  outputEndOfAstFile(headerText, headerTextLen, classes, tagBase, visitorRoot);
  
/*
  fprintf( visitorOutputFile,"}\n");
  fclose( visitorOutputFile );
*/
}



/*
   Text output routines
*/

void astecho(const char *text, int len)
{
  assert(state != UNINITIALIZED && state != DONE);
  check(state);
  if (state == INIT) addHeaderText(text, len);
  else {
    fwrite(text, len, 1, currentOutputFile);
    if (DEBUGPRINT) fflush(currentOutputFile);
  }
}

void expand(const char *text, int len)
{
  DirectiveListNode *d;
  char *s = mkstr(text, len);

  check(INCLASS);
  if (DEBUGPRINT) fprintf(stderr, "void expand(%s, %d)\n", s, len);
  d = addDirective(currentClass, s);

#ifndef TESTPARSING
  outputExpansion(currentOutputFile, currentClass, d);
#else
  astecho(text, len);
#endif
}

