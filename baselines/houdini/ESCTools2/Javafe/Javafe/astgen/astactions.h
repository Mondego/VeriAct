/* Copyright 2000, 2001, Compaq Computer Corporation */


#ifndef ASTACTIONS_H
#define ASTACTIONS_H

/*
   The input to astgen must have the same lexical language as Java and
   must follow the following grammar:

    astfile ::= PackageDeclaration_opt ImportDeclarations_opt ClassDeclaration*

   where the non-terminals on the right are those defined in the Java
   Language Specification.  Before the first ClassDeclaration there
   must be a line with the following lexical structure:

     ^{white-space}*"//#"{white-space}*"EndHeader".*\n

   Roughly speaking, astgen does the following: all text (including
   comments and whitespace) before the EndHeader directive is read
   into a buffer.  Then, for each ClassDeclaration, the text of the
   declaration (again including comments and whitespace) is appended
   to an "expanded" version of each generated class decaration.
*/


/*
   astactions is a stateful module that implements a state machine.
   The state transitions of this machine are:

     <anystate> -(init)-> INIT
     INIT -(visitorroot)-> INIT
     INIT -(tagbase)-> INIT
     INIT -(endheader)-> ABOVECLASS
     INIT -(endastfile)-> DONE
     ABOVECLASS -(abstract)-> ABOVECLASS
     ABOVECLASS -(endastfile)-> DONE
     ABOVECLASS -(classname)-> SUPERLESS -(supername)-> SUPERFULL
     SUPERCLESS,SUPERFULL -(beginclass)-> INCLASS
     INCLASS -(endclass)-> ABOVECLASS

   The state machine starts in the special state UNINITIALIZED.
   init is the only routine that may be called when the module is
   in this state.
*/

/*
   The following procedures must implement state transitions as
   indicated above.  They may have other side effects, such as
   outputing text.  The implementors may infer appropriate
   pre-conditions from the state transition diagram above, for
   example, "abstract" will not be called unless the current state is
   "ABOVECLASS".
*/
   
extern void init();
extern void visitorroot(const char *text, int len);
extern void tagbase(const char *text, int len);
extern void endheader();
extern void abstract();
extern void classname(const char *text, int len);
extern void supername(const char *text, int len);
extern void beginclass();
extern void endclass(const char *text, int len);
extern void endastfile();

/*
   With two exceptions, every piece of text in the input file (except
   the EOF character) is sent through (exactly) one of the following
   echo routines.  This includes text that is also passed to state
   transition procedures such as "classname" and "supername"; in these
   cases, astecho is called first, then the state transition routine.
   One exception to the rule that every piece of text is sent to an
   echo routine is the line containing the "//# EndHeader" section,
   which triggers the call to "endheader".  The other exception is the
   '}' character that ends a (top-level) class declaration; this
   triggers the call to "endclass", to which the closing "}" plus any
   characters matching "{white-space}\n{0,1}" is passed.

   astecho is called in most situations; it may be called in any
   state.  expand is called only in state INCLASS, and is called only
   on (and on every) piece of text that matches the pattern

     {whitespace}*"//#".*\n
*/

extern void astecho(const char *text, int len);
extern void expand(const char *text, int len);

#ifndef DEBUGPRINT
#define DEBUGPRINT 0
#endif

#endif /* ASTACTIONS_H */
