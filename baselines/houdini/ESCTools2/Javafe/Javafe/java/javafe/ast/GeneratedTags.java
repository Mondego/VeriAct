// -*- mode: java -*-
/* Copyright 2000, 2001, Compaq Computer Corporation */

/* IF THIS IS A JAVA FILE, DO NOT EDIT IT!  

   Most Java files in this directory which are part of the Javafe AST
   are automatically generated using the astgen comment (see
   ESCTools/Javafe/astgen) from the input file 'hierarchy.h'.  If you
   wish to modify AST classes or introduce new ones, modify
   'hierarchy.j.'
 */

package javafe.ast;

import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.ErrorSet;
import javafe.tc.TagConstants;

// Convention: unless otherwise noted, integer fields named "loc" refer
// to the location of the first character of the syntactic unit

public class GeneratedTags {
   static public final int COMPILATIONUNIT = 0;
   static public final int SINGLETYPEIMPORTDECL = COMPILATIONUNIT + 1;
   static public final int ONDEMANDIMPORTDECL = SINGLETYPEIMPORTDECL + 1;
   static public final int CLASSDECL = ONDEMANDIMPORTDECL + 1;
   static public final int INTERFACEDECL = CLASSDECL + 1;
   static public final int CONSTRUCTORDECL = INTERFACEDECL + 1;
   static public final int METHODDECL = CONSTRUCTORDECL + 1;
   static public final int INITBLOCK = METHODDECL + 1;
   static public final int LOCALVARDECL = INITBLOCK + 1;
   static public final int FIELDDECL = LOCALVARDECL + 1;
   static public final int FORMALPARADECL = FIELDDECL + 1;
   static public final int BLOCKSTMT = FORMALPARADECL + 1;
   static public final int SWITCHSTMT = BLOCKSTMT + 1;
   static public final int ASSERTSTMT = SWITCHSTMT + 1;
   static public final int VARDECLSTMT = ASSERTSTMT + 1;
   static public final int CLASSDECLSTMT = VARDECLSTMT + 1;
   static public final int WHILESTMT = CLASSDECLSTMT + 1;
   static public final int DOSTMT = WHILESTMT + 1;
   static public final int SYNCHRONIZESTMT = DOSTMT + 1;
   static public final int EVALSTMT = SYNCHRONIZESTMT + 1;
   static public final int RETURNSTMT = EVALSTMT + 1;
   static public final int THROWSTMT = RETURNSTMT + 1;
   static public final int BREAKSTMT = THROWSTMT + 1;
   static public final int CONTINUESTMT = BREAKSTMT + 1;
   static public final int LABELSTMT = CONTINUESTMT + 1;
   static public final int IFSTMT = LABELSTMT + 1;
   static public final int FORSTMT = IFSTMT + 1;
   static public final int SKIPSTMT = FORSTMT + 1;
   static public final int SWITCHLABEL = SKIPSTMT + 1;
   static public final int TRYFINALLYSTMT = SWITCHLABEL + 1;
   static public final int TRYCATCHSTMT = TRYFINALLYSTMT + 1;
   static public final int CONSTRUCTORINVOCATION = TRYCATCHSTMT + 1;
   static public final int CATCHCLAUSE = CONSTRUCTORINVOCATION + 1;
   static public final int ARRAYINIT = CATCHCLAUSE + 1;
   static public final int THISEXPR = ARRAYINIT + 1;
   static public final int ARRAYREFEXPR = THISEXPR + 1;
   static public final int NEWINSTANCEEXPR = ARRAYREFEXPR + 1;
   static public final int NEWARRAYEXPR = NEWINSTANCEEXPR + 1;
   static public final int CONDEXPR = NEWARRAYEXPR + 1;
   static public final int INSTANCEOFEXPR = CONDEXPR + 1;
   static public final int CASTEXPR = INSTANCEOFEXPR + 1;
   static public final int PARENEXPR = CASTEXPR + 1;
   static public final int AMBIGUOUSVARIABLEACCESS = PARENEXPR + 1;
   static public final int VARIABLEACCESS = AMBIGUOUSVARIABLEACCESS + 1;
   static public final int FIELDACCESS = VARIABLEACCESS + 1;
   static public final int AMBIGUOUSMETHODINVOCATION = FIELDACCESS + 1;
   static public final int METHODINVOCATION = AMBIGUOUSMETHODINVOCATION + 1;
   static public final int CLASSLITERAL = METHODINVOCATION + 1;
   static public final int EXPROBJECTDESIGNATOR = CLASSLITERAL + 1;
   static public final int TYPEOBJECTDESIGNATOR = EXPROBJECTDESIGNATOR + 1;
   static public final int SUPEROBJECTDESIGNATOR = TYPEOBJECTDESIGNATOR + 1;
   static public final int ERRORTYPE = SUPEROBJECTDESIGNATOR + 1;
   static public final int TYPENAME = ERRORTYPE + 1;
   static public final int ARRAYTYPE = TYPENAME + 1;
   static public final int SIMPLENAME = ARRAYTYPE + 1;
   static public final int COMPOUNDNAME = SIMPLENAME + 1;
   static public final int LAST_TAG = COMPOUNDNAME;


    static public /*@ non_null @*/ String toString(int tag) {
      switch (tag) {
        case COMPILATIONUNIT: return "COMPILATIONUNIT";
        case SINGLETYPEIMPORTDECL: return "SINGLETYPEIMPORTDECL";
        case ONDEMANDIMPORTDECL: return "ONDEMANDIMPORTDECL";
        case CLASSDECL: return "CLASSDECL";
        case INTERFACEDECL: return "INTERFACEDECL";
        case CONSTRUCTORDECL: return "CONSTRUCTORDECL";
        case METHODDECL: return "METHODDECL";
        case INITBLOCK: return "INITBLOCK";
        case LOCALVARDECL: return "LOCALVARDECL";
        case FIELDDECL: return "FIELDDECL";
        case FORMALPARADECL: return "FORMALPARADECL";
        case BLOCKSTMT: return "BLOCKSTMT";
        case SWITCHSTMT: return "SWITCHSTMT";
        case ASSERTSTMT: return "ASSERTSTMT";
        case VARDECLSTMT: return "VARDECLSTMT";
        case CLASSDECLSTMT: return "CLASSDECLSTMT";
        case WHILESTMT: return "WHILESTMT";
        case DOSTMT: return "DOSTMT";
        case SYNCHRONIZESTMT: return "SYNCHRONIZESTMT";
        case EVALSTMT: return "EVALSTMT";
        case RETURNSTMT: return "RETURNSTMT";
        case THROWSTMT: return "THROWSTMT";
        case BREAKSTMT: return "BREAKSTMT";
        case CONTINUESTMT: return "CONTINUESTMT";
        case LABELSTMT: return "LABELSTMT";
        case IFSTMT: return "IFSTMT";
        case FORSTMT: return "FORSTMT";
        case SKIPSTMT: return "SKIPSTMT";
        case SWITCHLABEL: return "SWITCHLABEL";
        case TRYFINALLYSTMT: return "TRYFINALLYSTMT";
        case TRYCATCHSTMT: return "TRYCATCHSTMT";
        case CONSTRUCTORINVOCATION: return "CONSTRUCTORINVOCATION";
        case CATCHCLAUSE: return "CATCHCLAUSE";
        case ARRAYINIT: return "ARRAYINIT";
        case THISEXPR: return "THISEXPR";
        case ARRAYREFEXPR: return "ARRAYREFEXPR";
        case NEWINSTANCEEXPR: return "NEWINSTANCEEXPR";
        case NEWARRAYEXPR: return "NEWARRAYEXPR";
        case CONDEXPR: return "CONDEXPR";
        case INSTANCEOFEXPR: return "INSTANCEOFEXPR";
        case CASTEXPR: return "CASTEXPR";
        case PARENEXPR: return "PARENEXPR";
        case AMBIGUOUSVARIABLEACCESS: return "AMBIGUOUSVARIABLEACCESS";
        case VARIABLEACCESS: return "VARIABLEACCESS";
        case FIELDACCESS: return "FIELDACCESS";
        case AMBIGUOUSMETHODINVOCATION: return "AMBIGUOUSMETHODINVOCATION";
        case METHODINVOCATION: return "METHODINVOCATION";
        case CLASSLITERAL: return "CLASSLITERAL";
        case EXPROBJECTDESIGNATOR: return "EXPROBJECTDESIGNATOR";
        case TYPEOBJECTDESIGNATOR: return "TYPEOBJECTDESIGNATOR";
        case SUPEROBJECTDESIGNATOR: return "SUPEROBJECTDESIGNATOR";
        case ERRORTYPE: return "ERRORTYPE";
        case TYPENAME: return "TYPENAME";
        case ARRAYTYPE: return "ARRAYTYPE";
        case SIMPLENAME: return "SIMPLENAME";
        case COMPOUNDNAME: return "COMPOUNDNAME";
        default: return "Unknown javafe GeneratedTag " + tag; 
      }
    }
}
