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
 * Represents a PrimitiveType syntactic unit. 
 * Subtypes determing the range of a valid tags
 *
 * @warning This AST node has associated locations only if syntax is
 * true.
 *
 * for backwards compatibility, the PrimitiveType makers make calls to the JavafePrimitiveType
 */
public abstract class PrimitiveType extends Type
{

    //@ invariant isValidTag();
    public abstract /*@pure*/ boolean isValidTag();
  public int tag;


  public int loc;


    //@ also ensures \result == this.tag;
    public final /*@pure*/ int getTag() { return this.tag; }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  protected PrimitiveType() {throw new RuntimeException("this sould never be called");}
 


// Generated boilerplate constructors:

  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.tag == tag;
  //@ ensures this.loc == loc;
  protected PrimitiveType(/*@nullable*/TypeModifierPragmaVec tmodifiers, int tag, int loc) {
     super(tmodifiers);
     this.tag = tag;
     this.loc = loc;
  }
  public void check() {
     super.check();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
  }

}
