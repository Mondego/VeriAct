/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;

import javafe.util.Assert;
import javafe.util.Set;
import javafe.util.ErrorSet;
import javafe.ast.*;

import escjava.ast.*;
import escjava.ast.TagConstants;



public class Helper {

  /** Provides support for appropriately handling the 'helper' pragma
      during translation.
  **/

    /* The following decoration lets us cache the fact that a given
       RoutineDecl is known to be not helper recursive, so that we don't
       have to recompute this fact.  In particular, the helperDecoration of
       a RoutineDecl is non-null precisely when the RoutineDecl is known
       to be not helper recursive.
       
       FIXME - this is not currently used, but should be?
    
    static {
        ASTDecoration helperDecoration = 
            new ASTDecoration("helperDecoration");
    }
    */


    /* Returns true iff the given RoutineDecl has a 'helper' modifier
       pragma.
    */
    public static boolean isHelper(RoutineDecl r) {
	if (r != null && r.pmodifiers != null) {
	    int pmodSize = r.pmodifiers.size();
	    for (int i = 0; i < pmodSize; i++) {
		ModifierPragma mp = r.pmodifiers.elementAt(i);
		if (mp.getTag() == TagConstants.HELPER) {
		    return true;
		}
	    }
	}
	return false;
    }


    // Abort the program because the given RoutineDecl is helper recursive.
    public static void abort(/*@ non_null */ RoutineDecl rd) {
	ErrorSet.fatal(rd.locId, "helper routine is recursive");
    }

}
