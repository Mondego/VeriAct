/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.ast;

import javafe.util.Assert;


/** Provides an extensible way to add 'fields' to the
<code>ASTNode</code>.  

Each <code>ASTDecoration</code> object essentially corresponds to an
additional field in the <code>ASTNode</code> class. The value of this
field for a particular <code>ASTNode</code> object is accessed and
mutated via the instance methods <code>get</code> and
<code>set</code>, respectively (see below). Each
<code>ASTDecoration</code> object also has an associated
<code>String</code>, which 'names' the corresponding field.

@see javafe.ast.ASTNode
*/

public class ASTDecoration {

    /***************************************************
     *                                                 *
     * Class variables:				       *
     *                                                 *
     **************************************************/

    //@ invariant allocated >= 0;
    //@ spec_public
    private static int allocated=0;

    //@ invariant 0<=my_slot && my_slot<allocated;
    //@ spec_public
    private int my_slot;


    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     **************************************************/

    /** The name of our decoration */
    //@ invariant name != null;
    //@ spec_public
    private String name;

    /** Our decoration's actual "static" type */
    //@ ghost public \TYPE decorationType;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * Creates a new <code>ASTDecoration</code> object with the given
     * name. <p>
     *
     * The caller should set the decorationType field of the result.<p>
     */
    //@ requires s != null;
    public ASTDecoration(String s) {
	name = s;
	my_slot = allocated;
	allocated++;
    }


    /***************************************************
     *                                                 *
     * Methods:					       *
     *                                                 *
     **************************************************/

   /**
    * Return the decoration value of an <code>ASTNode</code>, or <code> null
    * if the <code>ASTNode</code> has no decoration.
    */
    //@ requires n != null;
    //@ ensures \typeof(\result) <: decorationType;
    public /*@ non_null */ Object get(/*@ non_null */ ASTNode n) {
	Assert.notNull(n);
	Object[] v=n.getDecorations();
	if( v != null && my_slot < v.length )
	    return v[my_slot];
	else
	    return null;
    }	 //@ nowarn Post;


    /**
     * Set the decoration value of an <code>ASTNode</code>.
     */
    //@ requires n != null;
    //@ requires \typeof(val) <: decorationType;
    public void set(ASTNode n, Object val) {
	Object[] v = n.getDecorations();
	if (v==null) {
	    v = new Object[allocated];
	    n.setDecorations(v);
	} else if (my_slot >= v.length) {
	    Object[] _new = new Object[allocated];
	    System.arraycopy( v, 0, _new, 0, v.length );
	    v = _new;
	    n.setDecorations(v);
	}
	v[my_slot] = val;
    }


    /**
     * Return the name associated with <code>this</code>. 
     */
    public /*@non_null*/String toString() {
      return name;
    }

    /**
     * Return a string  containing the decoration's name, and the
     * decoration value for this <code>ASTNode</code>.
     */
    //@ requires n != null;
    public /*@non_null*/ String toString(/*@non_null*/ASTNode n) {
	Object val = get(n);
	return "[Decoration "+name+" "+ ( val==null? "" : val.toString() )
		+"]";
    }
}
