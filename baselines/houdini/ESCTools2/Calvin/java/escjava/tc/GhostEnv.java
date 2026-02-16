/* Copyright Hewlett-Packard, 2002 */

package escjava.tc;

import java.util.Enumeration;
import java.util.Hashtable;

import javafe.ast.*;
import escjava.ast.GhostDeclPragma;

import javafe.tc.*;


/**
 ** This class overrides EnvForTypeSig so that it "sees" ghost fields if
 ** escjava.tc.FlowInsensitiveChecks.inAnnotation is true.
 **/

public class GhostEnv extends EnvForTypeSig {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     ***************************************************/

    public GhostEnv(/*@ non_null @*/ Env parent,
		    /*@ non_null @*/ TypeSig peer,
		    boolean staticContext) {
	super(parent, peer, staticContext);
    }


    /***************************************************
     *                                                 *
     * Current/enclosing instances I:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Returns a new Env that acts the same as us, except that its
     ** current instance (if any) is not accessible. <p>
     **
     ** Note: this routine is somewhat inefficient and should be
     ** avoided unless an unknown environment needs to be coerced in
     ** this way. <p>
     **/
    public Env asStaticContext() {
	return new GhostEnv(parent, peer, true);
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Display information about an Env to System.out.  This function
     ** is intended only for debugging use.
     **/
    public void display() {
	parent.display();
	System.out.println("[[ extended with the (ghost) "
			   + (staticContext ? "static" : "complete")
			   + " bindings of type "
	    + peer.getExternalName() + " ]]");
    }


    /***************************************************
     *                                                 *
     * Code to locate our ghost fields by name:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Attempts to find a ghost field that belongs to us (including
     ** supertypes) with name n that is not equal to excluded.
     ** Returns the ghost field decl if successfull, otherwise null.
     ** Excluded may be null.
     **/
    public FieldDecl getGhostField(String n, FieldDecl excluded) {
	Enumeration e = collectGhostFields().elements();

	while (e.hasMoreElements()) {
	    FieldDecl f = (FieldDecl)e.nextElement();
	    if (!f.id.toString().equals(n))
		continue;

	    if (f!=excluded)
		return f;
	}	

	return null;
    }


    /***************************************************
     *                                                 *
     * Code to collect all our ghost fields:	       *
     *                                                 *
     ***************************************************/

    private Hashtable fields;	// used like a set

    /**
     ** Add all new ghost fields found in s (including its supertypes)
     ** to fields.  Also preps them (just resolves their types if needed).
     **/
    private void collectGhostFields(TypeSig s) {
	// Iterate over all TypeDeclElems in s:
	TypeDecl d = s.getTypeDecl();
	TypeDeclElemVec elems = d.elems;
	for (int i=0; i<elems.size(); i++) {
	    TypeDeclElem elem = elems.elementAt(i);
	    if (elem instanceof GhostDeclPragma) {
		FieldDecl ghost = ((GhostDeclPragma)elem).decl;
		if (!fields.containsKey(ghost)) {
		    FlowInsensitiveChecks.resolveType(s.getEnclosingEnv(), ghost.type);
		    //		    s.getEnclosingEnv().resolveType(ghost.type);
		    fields.put(ghost, ghost);
		}
	    }
	}

	// Now recursive to all supertypes:
	if (d instanceof ClassDecl) {
	    TypeName superClass = ((ClassDecl)d).superClass;
	    if (superClass!=null)
		collectGhostFields(TypeSig.getSig(superClass));
	} else if (d instanceof InterfaceDecl)
	    collectGhostFields(Types.javaLangObject());

	for (int i=0; i<d.superInterfaces.size(); i++)
	    collectGhostFields(TypeSig.getSig(
		d.superInterfaces.elementAt(i) ));
    }

    /**
     ** Return all our ghost fields (including our supertypes) as "set".
     **/
    private Hashtable collectGhostFields() {
	if (fields!=null)
	    return fields;

	fields = new Hashtable(5);
	collectGhostFields(peer);
	return fields;
    }


    /***************************************************
     *                                                 *
     * Misc. routines:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Is a given FieldDecl a ghost field? <p>
     **
     ** WARNING: The current implementation of this is slow.  If
     ** you need to call this routine a lot, rewrite it so it runs
     ** faster.
     **/
    public static boolean isGhostField(FieldDecl field) {
	TypeDecl d = field.getParent();
	if (d==null)
	    return false; // special fields like "length" can't be ghost fields


	TypeDeclElemVec elems = d.elems;
	for (int i=0; i<elems.size(); i++) {
	    TypeDeclElem elem = elems.elementAt(i);
	    if (elem instanceof GhostDeclPragma) {
		FieldDecl ghost = ((GhostDeclPragma)elem).decl;
		if (field==ghost)
		    return true;
	    }
	}

	return false;
    }


    /**
     ** Override to make ghost fields visible if
     ** escjava.tc.FlowInsensitiveChecks.inAnnotation is true.
     **/
    protected boolean hasField(Identifier id) {
	if (peer.hasField(id))
	    return true;

	if (!FlowInsensitiveChecks.inAnnotation)
	    return false;

	return (getGhostField(id.toString(), null) != null);
    }
}
