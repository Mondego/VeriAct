/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.tc;

import javafe.ast.ModifierPragmaVec;
import escjava.ast.TagConstants;

import javafe.tc.TypeSig;
import javafe.util.Info;

public class TypeCheck extends javafe.tc.TypeCheck
{
    /**
     * Creates a singleton instance of this class, and sets the <code>inst</code>
     * field to this instance. Only one instance should be created.  Also initializes
     * {@link javafe.tc.PrepTypeDeclaration}.
     */
    public TypeCheck() {
	inst = this;
	new escjava.tc.PrepTypeDeclaration();
    }

    /**
     * Called to obtain the algorithm for performing name resolution and type
     * checking.
     *
     * @return an instance of <code>escjava.tc.FlowInsensitiveChecks</code>.
     */
    public javafe.tc.FlowInsensitiveChecks makeFlowInsensitiveChecks() {
	return new escjava.tc.FlowInsensitiveChecks();
    }

    /**
     * Override {@link javafe.tc.TypeCheck#canAccess} to account for
     * <code>spec_public</code>.
     */
    public boolean canAccess(/*@non_null*/TypeSig from, /*@non_null*/TypeSig target,
                             int modifiers,
                             ModifierPragmaVec pmodifiers) {
	if (super.canAccess(from, target, modifiers, pmodifiers))
	    return true;

	if (!escjava.tc.FlowInsensitiveChecks.inAnnotation)
	    return false;

	if (pmodifiers == null)
	    return false;

	for (int i = 0; i < pmodifiers.size(); i++) {
	    if (pmodifiers.elementAt(i).getTag() == TagConstants.SPEC_PUBLIC)
		return true;
	}

	return false;
    }
} // end of class TypeCheck

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
