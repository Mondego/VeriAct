/* Copyright Hewlett-Packard, 2002 */

package escjava.tc;

import javafe.ast.ModifierPragmaVec;
import escjava.ast.TagConstants;

import javafe.tc.TypeSig;
import javafe.util.Info;


public class TypeCheck extends javafe.tc.TypeCheck {

    /**
     ** Creates a instance of TypeCheck, and sets the <code>inst</code>
     ** field to this instance. Only one instance should be created. 
     ** Also initializes PrepTypeDeclaration.
     **/
    public TypeCheck() {
	inst = this;
    }


    /**
     ** Called to obtain the algorithm for performing name resolution
     ** and type checking.  Returns an instance of
     ** <code>escjava.tc.FlowInsensitiveChecks</code>.
     **/
    public javafe.tc.FlowInsensitiveChecks makeFlowInsensitiveChecks() {
	return new escjava.tc.FlowInsensitiveChecks();
    }


    /**
     ** Override canAccess to account for spec_public
     **/
    public boolean canAccess(TypeSig from, TypeSig target,
				    int modifiers,
				    ModifierPragmaVec pmodifiers) {
	if (super.canAccess(from, target, modifiers, pmodifiers))
	    return true;

	if (!escjava.tc.FlowInsensitiveChecks.inAnnotation)
	    return false;

	if (pmodifiers==null)
	    return false;

	for (int i=0; i<pmodifiers.size(); i++) {
	    if (pmodifiers.elementAt(i).getTag()==TagConstants.SPEC_PUBLIC)
		return true;
	}

	return false;
    }
}



