/* Copyright 2000, 2001, Compaq Computer Corporation */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "astutil.h"


/*
   Operations on class lists
*/

Class *findClass(ClassListNode *cl, const char *name)
{
  for( ; cl; cl = cl->next)
    if (cl->c->name && strcmp(cl->c->name, name) == 0)
      return cl->c;
  return NULL;
}

ClassListNode *superclassList(Class *c)
{
  ClassListNode *result = NULL;

  for ( ; c; c = c->superclass) {
    ClassListNode *head = (ClassListNode*)malloc(sizeof(ClassListNode));
    head->c = c;
    head->next = result;
    result = head;
  }
  return result;
}

void freeClassList(ClassListNode *cl)
{
  while (cl) {
    ClassListNode *tmp = cl;
    cl = cl->next;
    free(tmp);
  }
}


/*
   Operations on directives
*/

static void eatSpace(char **p)
{ while (**p != '\0' && isspace(**p)) (*p)++; }

/* Leaves *p pointing to one character after the token returned. */
static char *nextToken(char **p)
{
  char *start;
  if (**p == '\0') return NULL;
  eatSpace(p);
  start = *p;
  if (isalpha(**p)) do { (*p)++; } while (isalnum(**p));
  else if (**p == '[' || **p == ']' || **p == '?' || **p == '*' || **p == '.')
    (*p)++;
  else return NULL;
  return mktmpstr(start, (*p) - start);
}

DirectiveListNode *addDirective(Class *c, char *text)
{
  DirectiveListNode **d;
  for(d = &c->directives; *d != NULL; d = &((*d)->next)) ;
  *d = (DirectiveListNode *)malloc(sizeof(DirectiveListNode));
  (*d)->text = text;
  parseDirective(*d);
  (*d)->next = NULL;
  return (*d);
}

void parseDirective(DirectiveListNode *d)
{
  char *p, *start, *tok;
  assert(p = strchr(d->text, '#'));
  d->indent = 1 + (p - d->text) - strlen("//#");
  p++;

  eatSpace(&p);
  start = p;
  tok = nextToken(&p);
  assert(tok != NULL);
  if (strcmp(tok, NOMAKERTOK) == 0) d->tag = NOMAKERDIRECTIVE;
  else if (strcmp(tok, MANUALTAGTOK) == 0) d->tag = MANUALTAGDIRECTIVE;
  else if (strcmp(tok, POSTMAKECALLTOK) == 0) d->tag = POSTMAKECALLDIRECTIVE;
  else if (strcmp(tok, POSTCHECKCALLTOK) == 0) d->tag = POSTCHECKCALLDIRECTIVE;
  else if (strcmp(tok, MAKERSPECTOK) == 0) {
    d->tag = MAKERSPECDIRECTIVE;
    // copy the rest of the line
    eatSpace(&p);
    d->i.ms.pragma = cpstr(p);
  } else {
    char *end = p;

    d->tag = FIELDDIRECTIVE;

    /* Read entire type, including brackets */
    for(;;) {
      tok = nextToken(&p);
      if (*tok != '[' && *tok != ']') break;
      end = p;
    }
    d->i.f.type = mkstr(start, end - start);

    if (tok != NULL && *tok == '*') {
      d->i.f.sequence = TRUE;
      tok = nextToken(&p);
    } else d->i.f.sequence = FALSE;

    assert(tok != NULL && isalpha(*tok));
    d->i.f.name = cpstr(tok);
    tok = nextToken(&p);

    d->i.f.notnull = TRUE;
    d->i.f.checkfield = TRUE;
    d->i.f.notnullloc = FALSE;
    d->i.f.syntax = FALSE;
    while (tok != NULL) {
      if (strcmp(tok, NULLOKTOK) == 0) d->i.f.notnull = FALSE;
      else if (strcmp(tok, NOCHECKTOK) == 0) d->i.f.checkfield = FALSE;
      else if (strcmp(tok, NOTNULLLOCTOK) == 0) d->i.f.notnullloc = TRUE;
      else if (strcmp(tok, SYNTAXTOK) == 0) d->i.f.syntax = TRUE;
      tok = nextToken(&p);
    }
  }
}


/*
   String functions
*/

char *mkstr(const char *s, int len)
{
  char *result = (char *)malloc(len+1);
  memcpy(result, s, len);
  result[len] = '\0';
  return result;
}

static int tmpstrlen;
static char *tmpstr = NULL;
char *mktmpstr(const char *s, int len)
{
  if (!tmpstr) {
    tmpstrlen = 4*len;
    tmpstr = (char *)malloc(tmpstrlen);
  } else if (len + 1 > tmpstrlen) {
    tmpstrlen = 4*len;
    tmpstr = (char *)realloc(tmpstr, tmpstrlen);
  }
  memcpy(tmpstr, s, len);
  tmpstr[len] = '\0';
  return tmpstr;
}
/* jrk: added return to this function.  is it necessary? */
char *cpstr(const char *s)
{ return mkstr(s, strlen(s)); }

char *catstr(const char *s1, const char *s2)
{
  int s1l = strlen(s1);
  char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
  strcpy(result, s1);
  strcat(result + s1l, s2);
  return result;
}



/*
   Misc functions
*/

void indent(FILE *out, int indent)
{ while(indent--) fputc(' ', out); }
