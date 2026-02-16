/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.tc;

import java.util.Enumeration;
import java.util.Hashtable;

import javafe.ast.*;
import escjava.ast.GhostDeclPragma;
import escjava.ast.ModelDeclPragma;
import escjava.ast.TagConstants;
import escjava.ast.Utils;

import javafe.tc.*;
import javafe.tc.TypeSig;

/**
 * This class overrides {@link EnvForTypeSig} so that it "sees" ghost and model
 * fields if {@link escjava.tc.FlowInsensitiveChecks#inAnnotation} is
 * <code>true</code>.
 */

public class GhostEnv extends EnvForTypeSig
{
    // Creation

    public GhostEnv(/*@ non_null */ Env parent,
		    /*@ non_null */ TypeSig peervar,
		    boolean staticContext) {
	super(parent, peervar, staticContext);
    }

    // Current/enclosing instances

    /**
     * Returns a new {@link Env} that acts the same as us, except that its current
     * instance (if any) is not accessible.
     *
     * @note This routine is somewhat inefficient and should be avoided unless an
     * unknown environment needs to be coerced in this way.
     */
    public Env asStaticContext() {
	return new GhostEnv(parent, peervar, true);
    }

    // Debugging functions

    /**
     * Display information about an {@link Env} to {@link System#out}.  This function
     * is intended only for debugging use.
     */
    public void display() {
	parent.display();
	System.out.println("[[ extended with the (ghost) "
			   + (staticContext ? "static" : "complete")
			   + " bindings of type "
	    + peervar.getExternalName() + " ]]");
	FieldDeclVec fdv = ((escjava.tc.TypeSig)peervar).jmlFields;
	for (int i=0; i<fdv.size(); ++i) {
		System.out.println("    " + fdv.elementAt(i).id);
	}
	
    }

    // Code to locate our ghost and model fields by name

    /**
     * Attempts to find a ghost or model field that belongs 
     * to us (including supertypes) with
     * name <code>n</code> that is not equal to <code>excluded</code>, which may
     * be <code>null</code>.
     *
     * @param n the name of the ghost field to get.
     * @param excluded a field declaration to avoid.
     * @return the ghost field declaration if successful, otherwise
     * <code>null</code>.
     */
    public FieldDecl getGhostField(String n, FieldDecl excluded) {
	FieldDeclVec fdv = peervar.getFields(false);
	for (int i=0; i<fdv.size(); ++i) {
	    FieldDecl f = fdv.elementAt(i);
	    if (!f.id.toString().equals(n))
		continue;

	    if (f != excluded)
		return f;
	}	

	return null;
    }

    static public boolean isStatic(FieldDecl d) {
	boolean isStatic = d.parent instanceof InterfaceDecl;
	if (Modifiers.isStatic(d.modifiers)) isStatic = true;
	if (Utils.findModifierPragma(d,
		TagConstants.INSTANCE) != null) isStatic = false;
	return isStatic;
    }

    // Misc. routines

    /**
     * Determines if a field is a ghost (not model or Java) field
     *
     * @return true if the FieldDecl <code>field</code> a ghost
     * field (and not a model or Java field).
     *
     */
    //@ requires field != null;
    public static boolean isGhostField(FieldDecl field) {
	TypeDecl d = field.getParent();
        // special fields like "length" can't be ghost fields
	if (d == null)
	    return false;

	return Utils.findModifierPragma(field.pmodifiers,
                               TagConstants.GHOST) != null;
/*
	TypeDeclElemVec elems = d.elems;
	for (int i = 0; i < elems.size(); i++) {
	    TypeDeclElem elem = elems.elementAt(i);
	    if (elem instanceof GhostDeclPragma) {
		FieldDecl ghost = ((GhostDeclPragma)elem).decl;
		if (field == ghost)
		    return true;
	    }
	}

	return false;
*/
    }

    /**
     * Override to make ghost fields visible if {@link
     * escjava.tc.FlowInsensitiveChecks#inAnnotation} is <code>true</code>.
     */
    protected boolean hasField(Identifier id) {
	if (peervar.hasField(id))
	    return true;

	if (!FlowInsensitiveChecks.inAnnotation)
	    return false;

	return (getGhostField(id.toString(), null) != null);
    }

    public FieldDeclVec getFields(boolean allFields) {
	FieldDeclVec fdv = FieldDeclVec.make();
	fdv.append(super.getFields(allFields));
	if (!(peervar instanceof escjava.tc.TypeSig)) return fdv;
	escjava.tc.TypeSig ts = (escjava.tc.TypeSig)peervar;
	fdv.append(ts.jmlFields);
	if (!allFields) return fdv;
	fdv.append(ts.jmlHiddenFields);
	fdv.append(ts.jmlDupFields);
        return fdv;
    }

} // end of class GhostEnv

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
