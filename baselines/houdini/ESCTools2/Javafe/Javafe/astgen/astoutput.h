/* Copyright 2000, 2001, Compaq Computer Corporation */


#ifndef ASTOUTPUT_H
#define ASTOUTPUT_H

/* Must include stdio.h and astutil.h first! */

#define VECTORSUFFIX "Vec"
#define TAGSBASECLASS "GeneratedTags"
#define VISITORCLASS "Visitor"
#define VISITORARGRESULTCLASS "VisitorArgResult"

extern void outputStartOfFile(FILE *o, const char *text, int len);
extern void outputExpansion(FILE *o, Class *c, DirectiveListNode *d);
extern void outputEndOfFile(FILE *o, Class *c, const char *text, int len,
			    const char *visitorRoot);
extern void outputEndOfAstFile(const char *text, int len,
			       ClassListNode *classes,
			       const char *tagBase,
			       const char *visitorRoot);

static boolean isJavaPrimitiveType(const char *s);

#endif /* ASTOUTPUT_H */
