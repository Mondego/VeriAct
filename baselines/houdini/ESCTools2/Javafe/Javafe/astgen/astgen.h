/* Copyright 2000, 2001, Compaq Computer Corporation */


#ifndef ASTGEN_H
#define ASTGEN_H

/*
   Module implements a state machine.

   The state transitions are:

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

*/

extern void init();

extern void astecho(const char *text, int len);

extern void classname(const char *text, int len);
extern void supername(const char *text, int len);
extern void beginclass();
extern void expand(const char *text, int len);
extern void endclass();
extern void endastfile();

#ifndef DEBUGPRINT
#define DEBUGPRINT 0
#endif

#endif /* ASTGEN_H */
