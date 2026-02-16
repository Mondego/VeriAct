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


/* ---------------------------------------------------------------------- */

/**
 * Represents a Type syntactic unit. <p>
 *
 * WARNING: unlike other AST nodes, Type and it's subtypes may
 * not have associated locations!  Locations exist only if syntax is
 * true.
 */

public abstract class Type extends ASTNode
{
    /**
     * Does this AST Node have associated locations?  True if yes.
     */
    //@ public model boolean syntax;
    //@ public represents syntax <- !isInternal;

    public /*@nullable*/ TypeModifierPragmaVec tmodifiers;

    protected Type() {}



// Generated boilerplate constructors:

    //@ ensures this.tmodifiers == tmodifiers;
    protected Type(/*@nullable*/TypeModifierPragmaVec tmodifiers) {
       super();
       this.tmodifiers = tmodifiers;
    }
    public void check() {
       super.check();
       TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
       if (tmp_tmodifiers != null)
          for(int i = 0; i < tmp_tmodifiers.size(); i++)
             tmp_tmodifiers.elementAt(i).check();
    }

}
