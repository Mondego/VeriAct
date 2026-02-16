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


/**
 * Represents a Literal.
 *
 * <p> The tag of a LiteralExpr should be one of the *LIT (eg INTLIT)
 * constants defined in TagConstants.  The value fields is a
 * Character/String/Long/Double/Boolean/null, as appropriate.
 */

public class LiteralExpr extends Expr
{
  //@ invariant isValidTag(tag);
  public int tag;


  //@ invariant isValidValue(tag, value);
  public /*@nullable*/ Object value;


  final
  public int loc;

  public final int getTag() { return this.tag; }

  private void postCheck() {
    Assert.notFalse(isValidTag(tag));
    Assert.notFalse(isValidValue(tag, value));
  }
  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  //@ also public normal_behavior
  //@ ensures \result == loc;
  public /*@ pure @*/ int getEndLoc() { return loc; }

  static public LiteralExpr cast(LiteralExpr lit, int t) {
	if (!(lit.value instanceof Number)) return lit;
	Number num = (Number)lit.value;
	if (t == TagConstants.BYTELIT) {
	    return LiteralExpr.make(t,new Integer(num.byteValue()),lit.getStartLoc());
	} else if (t == TagConstants.SHORTLIT) {
	    return LiteralExpr.make(t,new Short(num.shortValue()),lit.getStartLoc());
	} else if (t == TagConstants.INTLIT) {
	    return LiteralExpr.make(t,new Integer(num.intValue()),lit.getStartLoc());
	} else if (t == TagConstants.LONGLIT) {
	    return LiteralExpr.make(t,new Long(num.longValue()),lit.getStartLoc());
	} else if (t == TagConstants.FLOATLIT) {
	    return LiteralExpr.make(t,new Float(num.floatValue()),lit.getStartLoc());
	} else if (t == TagConstants.DOUBLELIT) {
	    return LiteralExpr.make(t,new Double(num.doubleValue()),lit.getStartLoc());
	} else return lit;
	// FIXME - what about casts to character values ???
  }

  // Note: Java does not actually have byte and short literals.
  // We include them here because we pre-compute literals that have been
  // cast to other types, e.g. (short)0, as a literal of the target type
  // FIXME - short and byte are included for completeness, but I'm not sure they actually ever get used that way

     /*@ ensures \result <==> tag==TagConstants.NULLLIT    ||
       @                      tag==TagConstants.BOOLEANLIT ||
       @                      tag==TagConstants.BYTELIT    ||
       @                      tag==TagConstants.SHORTLIT   ||
       @                      tag==TagConstants.CHARLIT    ||
       @                      tag==TagConstants.INTLIT     ||
       @                      tag==TagConstants.LONGLIT    ||
       @                      tag==TagConstants.FLOATLIT   ||
       @                      tag==TagConstants.DOUBLELIT  ||
       @                      tag==TagConstants.STRINGLIT;
       @*/
      public static boolean isValidTag(int tag)
      {
          return tag==TagConstants.NULLLIT    ||
                 tag==TagConstants.BOOLEANLIT ||
                 tag==TagConstants.BYTELIT    ||
                 tag==TagConstants.SHORTLIT   ||
                 tag==TagConstants.CHARLIT    ||
                 tag==TagConstants.INTLIT     ||
                 tag==TagConstants.LONGLIT    ||
                 tag==TagConstants.FLOATLIT   ||
                 tag==TagConstants.DOUBLELIT  ||
                 tag==TagConstants.STRINGLIT;
      }
     /*@ requires isValidTag(tag);
       @ ensures \result <==> (tag==TagConstants.NULLLIT    ==> value == null) &&
       @                      (tag==TagConstants.BOOLEANLIT ==> value instanceof Boolean) &&
       @                      (tag==TagConstants.BYTELIT    ==> value instanceof Byte) &&
       @                      (tag==TagConstants.SHORTLIT   ==> value instanceof Short) &&
       @                      (tag==TagConstants.CHARLIT    ==> value instanceof Integer) &&
       @                      (tag==TagConstants.INTLIT     ==> value instanceof Integer) &&
       @                      (tag==TagConstants.LONGLIT    ==> value instanceof Long) &&
       @                      (tag==TagConstants.FLOATLIT   ==> value instanceof Float) &&
       @                      (tag==TagConstants.DOUBLELIT  ==> value instanceof Double) &&
       @                      (tag==TagConstants.STRINGLIT  ==> value instanceof String);
       @*/
      public static boolean isValidValue(int tag, Object value)
      {
          if (tag == TagConstants.NULLLIT)    return value == null;
          if (tag == TagConstants.BOOLEANLIT) return value instanceof Boolean;
          if (tag == TagConstants.BYTELIT)    return value instanceof Byte;
          if (tag == TagConstants.SHORTLIT)   return value instanceof Short;
          if (tag == TagConstants.CHARLIT)    return value instanceof Integer;
          if (tag == TagConstants.INTLIT)     return value instanceof Integer;
          if (tag == TagConstants.LONGLIT)    return value instanceof Long;
          if (tag == TagConstants.FLOATLIT)   return value instanceof Float;
          if (tag == TagConstants.DOUBLELIT)  return value instanceof Double;
          if (tag == TagConstants.STRINGLIT)  return value instanceof String;
	  return false; //should be unreachable
      } 

  //@ requires isValidTag(tag);
  //@ requires isValidValue(tag, value);
  //@ requires loc != javafe.util.Location.NULL;
  public static /*@non_null*/ LiteralExpr make(int tag, Object value, int loc) {
     LiteralExpr result = new LiteralExpr(tag,value,loc);
     return result;
  }

  //@ requires isValidTag(tag);
  //@ requires isValidValue(tag, value);
  public static /*@non_null*/ LiteralExpr makeNonSyntax(int tag, Object value) {
     LiteralExpr result = new LiteralExpr(tag,value,Location.NULL);
     return result;
  }

  public /*@ non_null @*/ String getInfoNewTree(){
	  return (value==null ? "null" : value.toString());
  }


// Generated boilerplate constructors:

  //@ ensures this.tag == tag;
  //@ ensures this.value == value;
  //@ ensures this.loc == loc;
  protected LiteralExpr(int tag, /*@nullable*/Object value, int loc) {
     super();
     this.tag = tag;
     this.value = value;
     this.loc = loc;
  }

// Generated boilerplate methods:

  public final int childCount() {
     return 1;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.value;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[LiteralExpr"
        + " tag = " + this.tag
        + " value = " + this.value
        + " loc = " + this.loc
        + "]";
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitLiteralExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitLiteralExpr(this, o); }

  public void check() {
     super.check();
     postCheck();
  }

}
