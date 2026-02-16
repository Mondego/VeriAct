/* Copyright 2000, 2001, Compaq Computer Corporation */


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "astutil.h"
#include "astoutput.h"
#include "astactions.h"


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

static const char *visitorRoot;

static void check(int expectedState)
{ }


/*
   State transition routines
*/


void init()
{
  state = INIT;
  check(INIT);
}

void visitorroot(const char *text, int len)
{ }

void tagbase(const char *text, int len)
{ }

void endheader()
{
  if (DEBUGPRINT) fprintf(stderr, "void endheader()\n");
  check(INIT);
  state = ABOVECLASS;
  check(ABOVECLASS);
}

void abstract()
{
  if (DEBUGPRINT) fprintf(stderr, "void abstract()\n");
  check(ABOVECLASS);
}

void classname(const char *text, int len)
{
  char *s = mkstr(text, len);
  if (DEBUGPRINT) fprintf(stderr, "void classname(%s, %d)\n", s, len);
  check(ABOVECLASS);
  s = catstr(s, ".java");
  printf("%s\n", s);
  free(s);
  state = SUPERLESS;
  check(SUPERLESS);
}

void supername(const char *text, int len)
{
  Class *sc;
  char *s = mktmpstr(text, len);
  if (DEBUGPRINT) fprintf(stderr, "void supername(%s, %d)\n", s, len);
  check(SUPERLESS);
  state = SUPERFULL;
  check(SUPERFULL);
}

void beginclass()
{
  if (DEBUGPRINT) fprintf(stderr, "void beginclass()\n");
  assert(state == SUPERLESS || state == SUPERFULL);
  check(state);
  state = INCLASS;
  check(INCLASS);
}

void endclass(const char *text, int len)
{
  if (DEBUGPRINT) fprintf(stderr, "void endclass()\n");
  check(INCLASS);
  state = ABOVECLASS;
  check(ABOVECLASS);
}

void endastfile()
{
  if (DEBUGPRINT) fprintf(stderr, "void endastfile()\n");
  assert(state == ABOVECLASS || state == INIT);
  check(state);
  printf(VISITORCLASS ".java\n");
  printf(TAGSBASECLASS ".java\n");
}



/*
   Text output routines
*/

void astecho(const char *text, int len)
{
  assert(state != UNINITIALIZED && state != DONE);
  check(state);
}

void expand(const char *text, int len)
{
  char *s = mkstr(text, len);
  if (DEBUGPRINT) fprintf(stderr, "void expand(%s, %d)\n", s, len);
  check(INCLASS);
}

