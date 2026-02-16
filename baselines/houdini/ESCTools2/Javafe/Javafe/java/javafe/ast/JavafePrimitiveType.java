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
 * Represents a Java Front End PrimitiveType syntactic unit. 
 * The tag should be one of the *TYPE constants defined in TagConstants
 * (e.g., INTTYPE).
 *
 * @warning This AST node has associated locations only if syntax is
 * true.
 */

public class JavafePrimitiveType extends PrimitiveType
{
    /*@ public normal_behavior
      @   ensures  \result == (tag == TagConstants.BOOLEANTYPE || tag == TagConstants.INTTYPE
      @  || tag == TagConstants.LONGTYPE || tag == TagConstants.CHARTYPE
      @  || tag == TagConstants.FLOATTYPE || tag == TagConstants.DOUBLETYPE
      @  || tag == TagConstants.VOIDTYPE || tag == TagConstants.NULLTYPE
      @  || tag == TagConstants.BYTETYPE || tag == TagConstants.SHORTTYPE);
      @*/
    public static /*@pure*/ boolean isValidTag(int tag) {
	return (tag == TagConstants.BOOLEANTYPE || tag == TagConstants.INTTYPE
		|| tag == TagConstants.LONGTYPE || tag == TagConstants.CHARTYPE
		|| tag == TagConstants.FLOATTYPE || tag == TagConstants.DOUBLETYPE
		|| tag == TagConstants.VOIDTYPE || tag == TagConstants.NULLTYPE
		|| tag == TagConstants.BYTETYPE || tag == TagConstants.SHORTTYPE);
    }

    /*@ also
      @ public normal_behavior
      @   ensures  \result == JavafePrimitiveType.isValidTag(tag);
      @*/
    public /*@pure*/ boolean isValidTag() {
	return JavafePrimitiveType.isValidTag(tag);
    }
  /**
   * Normal maker that produces syntax, but requires a non-NULL location.
   */
    /*@ public normal_behavior
      @   requires  JavafePrimitiveType.isValidTag(tag);
      @   requires loc != javafe.util.Location.NULL;
      @   ensures \result.syntax;
      @*/
  public static /*@ non_null */ JavafePrimitiveType make(TypeModifierPragmaVec tmodifiers,
				   int tag, int loc) {
     JavafePrimitiveType result = new JavafePrimitiveType(
                                tmodifiers,
                                tag,
                                loc);
     return result;
  }

  /**
   * Special maker for producing non-syntax, which does not require a
   * location.
   */
    /*@ public normal_behavior
      @   requires  JavafePrimitiveType.isValidTag(tag);
      @   ensures !\result.syntax;
      @*/
  public static /*@non_null*/ JavafePrimitiveType makeNonSyntax(int tag) {
     JavafePrimitiveType result = new JavafePrimitiveType(
                                 null,
                                 tag,
                                 Location.NULL);
     return result;
  }

  // manual override for backwards compatibility
    /*@ public normal_behavior
      @   requires  JavafePrimitiveType.isValidTag(tag);
      @   requires loc != javafe.util.Location.NULL;
      @   ensures \result.syntax;
      @*/
  static public /*@non_null*/ JavafePrimitiveType make(int tag, int loc) {
    return JavafePrimitiveType.make(null, tag, loc);
  }

  private void postCheck() {
    boolean goodtag = JavafePrimitiveType.isValidTag(tag);
    Assert.notFalse(goodtag); 
  }


// Generated boilerplate constructors:

  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.tag == tag;
  //@ ensures this.loc == loc;
  protected JavafePrimitiveType(/*@nullable*/TypeModifierPragmaVec tmodifiers, int tag, int loc) {
     super(tmodifiers, tag, loc);
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null) sz += tmp_tmodifiers.size();
     return sz + 0;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     sz = (tmp_tmodifiers == null ? 0 : tmp_tmodifiers.size());
     if (0 <= index && index < sz)
        return tmp_tmodifiers.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[JavafePrimitiveType"
        + " tmodifiers = " + this.tmodifiers
        + " tag = " + this.tag
        + " loc = " + this.loc
        + "]";
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitJavafePrimitiveType(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitJavafePrimitiveType(this, o); }

  public void check() {
     super.check();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
     postCheck();
  }

}
