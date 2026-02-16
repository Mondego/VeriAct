/* Copyright 2000, 2001, Compaq Computer Corporation */


#ifndef ASTUTIL_H
#define ASTUTIL_H

typedef int boolean;
#ifndef TRUE
#define TRUE -1
#endif
#ifndef FALSE
#define FALSE 0
#endif

typedef struct Class_s Class;
typedef struct ClassListNode_s ClassListNode;
typedef struct DirectiveListNode_s DirectiveListNode;
typedef struct FieldDirective_s FieldDirective;
typedef struct MakerSpecDirective_s MakerSpecDirective;

/* There different variants of DirectiveListNodes distinguished
   by the following tags: */
#define FIELDDIRECTIVE 1
#define NOMAKERDIRECTIVE 2
#define MANUALTAGDIRECTIVE 3
#define POSTMAKECALLDIRECTIVE 4
#define POSTCHECKCALLDIRECTIVE 5
#define MAKERSPECDIRECTIVE 6

/* The following are non-field directives. */
#define NOMAKERTOK "NoMaker"
#define MANUALTAGTOK "ManualTag"
#define POSTMAKECALLTOK "PostMakeCall"
#define POSTCHECKCALLTOK "PostCheckCall"
#define MAKERSPECTOK "MakerSpec"

/* The following are field qualifiers */
#define NULLOKTOK "NullOK"
#define NOCHECKTOK "NoCheck"
#define NOTNULLLOCTOK "NotNullLoc"
#define SYNTAXTOK "Syntax"


/* Function prototypes */

extern Class *findClass(ClassListNode *cl, const char *name);
/* Effects: Returns the first element of "cl" whose name is "name."
    Returns NULL if no such class exists. */

extern ClassListNode *superclassList(Class *c);
/* Effects: Returns a newly-allocated list of classes containing "c"
    and all its superclasses in order starting from the top-most and
    ending with c. */

extern void freeClassList(ClassListNode *cl);
/* Effects: Frees all ClassListNodes reachable from "cl" (including
    cl) (but doesn't free the classes pointed to by the list). */

extern DirectiveListNode *addDirective(Class *c, char *text);
/* Modifies: "c"
   Effects: Malloc and returns new DirectiveListNode.  This
   new node is appended to the end of the direective list of "c".
   The "text" field of the result points to the "text" argument,
   and the rest of the fields of the result are initialized by
   parsing "text". */

extern void parseDirective(DirectiveListNode *d);
/* Modifes: "d"
   Requires: "d->text" points to a null-terminated string.
   Effects: sets the rest of the fields of "d" to values
   determined by parsing "d->text".  Syntax errors are
   "reported" via assertions failing. */


extern char *mkstr(const char *s, int len);
/* Effects: Malloc and return a string consisting of the first "len"
   characters of "s" plus an extra '\0' at the end (the first
   "len" characters of "s" are copied even if one of them is '\0'). */

extern char *mktmpstr(const char *s, int len);
/* Effects: Same as above except returns an internal buffer that is reused
   on each call to mktmpstr. */

extern char *cpstr(const char *s);
/* Requires: "s" is a zero-terminated string.
   Effects: Makes a copy of "s", allocating space for it using malloc. */

extern char *catstr(const char *s1, const char *s2);
/* Requires: "s1" and "s2" are a zero-terminated strings.
   Effects: Returns the concatenation of its arguments, allocating space
   for the result using malloc. */


extern void indent(FILE *out, int indent);
/* Modifies: "out"
   Effects: Print "indent" spaces to "out" */


/*
   Details of structure definitions
*/

struct Class_s {
  boolean abstract;
  const char *name;
  Class *superclass;
  DirectiveListNode *directives;
};

struct ClassListNode_s {
  Class *c;
  ClassListNode *next;
};

struct FieldDirective_s {
  char *type;
  boolean sequence, notnull, checkfield, notnullloc, syntax;
  char *name;
};

struct MakerSpecDirective_s {
  char *pragma;
};

struct DirectiveListNode_s {
  char *text;
  int indent;
  int tag; /* Must be one of the tags from above */
  union {
    FieldDirective f;
    MakerSpecDirective ms;
    /* Invariants:
       tag == FIELDDIRECTIVE ==> "i.f" is parsed version of "text"
       tag == NOMAKERDIRECTIVE ==> no elements of this union are valid
       tag == MANUALTAGDIRECTIVE ==> no elements of this union are valid
       tag == POSTMAKECALLDIRECTIVE ==> no elements of this union are valid
       tag == POSTCHECKCALLDIRECTIVE ==> no elements of this union are valid
       tag == MAKERSPECDIRECTIVE ==> "i.ms" is parsed version of "text"
       */
  } i;
  DirectiveListNode *next;
};

#endif /* ASTUTIL_H */
