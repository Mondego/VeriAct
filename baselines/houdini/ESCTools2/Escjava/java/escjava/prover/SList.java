/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

import java.io.*;

/**
 * <code>SList</code>s represent possibly-empty lists of
 * <code>SExp</code>s.
 *
 * <h3> Creation </h3>
 *
 * <p> As a convenience, lists of fewer than 9 elements may be created
 * directly by calling <code>SList.make</code> on the desired
 * elements.  E.g., <code>SList.make(x, y)</code> creates the list
 * <code>(</code><i>x</i><code> </code><i>y</i><code>)</code> where
 * <i>x</i> and <i>y</i> are the contents of the <code>SExp</code>
 * variables <code>x</code> and <code>y</code>.  Longer lists may be
 * created either from arrays (see <code>fromArray</code>) or by
 * combining shorter lists via <code>append</code>. </p>
 *
 * <p> As a further convenience, <code>SList.make</code> calls
 * {@link SExp#fancyMake(Object)} on its arguments before placing them in
 * the resulting list.  This means that <code>Atom</code>s may be
 * specified by a <code>String</code> holding their name and that
 * S-expression integers may be specified by <code>Integer</code>s.
 * E.g., <code>SList.make("NEG", new Integer(1))</code> represents the
 * S-expression <code>(NEG, 1)</code>. </p>
 *
 * <p> <code>SList</code>s may not contain <code>null</code>
 * elements. </p>
 *
 *
 * <h3> Inspection </h3>
 *
 * <p> <code>SList</code>s can be tested to see if they are empty, can
 * have their length determined, and can have their <i>i</i>-index
 * element extracted.  (If an attempt is made to extract a
 * non-existent element, <code>SExpTypeError</code> will be
 * thrown.) </p>
 *
 *
 * <h3> Manipulation </h3>
 *
 * <p> S-expressions are currently immutable; accordingly all
 * manipulation methods act functionally.  S-expressions may be made
 * mutable later via the addition of mutation methods. </p>
 *
 * <p> <code>SList</code>s can be appended to each other.  Elements
 * can also be added to the front or end of a list. </p>
 *
 *
 * <h3> Implementation </h3>
 *
 * <p> <code>SList</code>s are currently implemented via pairs and a
 * nil object a la Lisp.  These are provided by the non-public classes
 * <code>SPair</code> and <code>SNil</code>.  This implementation is
 * subject to change at any time; clients should rely only on the
 * publically-provided interface of this class. </p>
 *
 * @see escjava.prover.SNil
 * @see escjava.prover.SPair
 *
 * @todo kiniry 14 March 2003 - Add a list model.
 */

public abstract class SList extends SExp
{
    /**
     * <code>SList</code> may not be extended outside this package.
     */
    /* package */ SList() {};

    /**
     * @return a S-expression representing the empty list.
     */
    //@ public normal_behavior
    //@   ensures \result.isEmpty();
    //@   ensures \result.length() == 0;
    public static /*@ pure non_null @*/ SList make() {
	return SNil.getNil();
    }

    /**
     * @return a S-expression representing a 1-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments.
     */
    //-@ pure
    public static /*@ non_null @*/ SList make(/*@ non_null @*/ Object a1) {
	return new SPair(SExp.fancyMake(a1), SNil.make());
    }

    /**
     * @return a S-expression representing a 2-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments.
     */
    //-@ pure
    public static /*@ non_null @*/ SList make(/*@ non_null @*/ Object a1, 
                                                   /*@ non_null @*/ Object a2) {
	return new SPair(SExp.fancyMake(a1), make(a2));
    }

    /**
     * @return a S-expression representing a 3-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments. <p>
     */
    public static /*@ pure non_null @*/ SList make(/*@ non_null @*/ Object a1,
                                                   /*@ non_null @*/ Object a2,
                                                   /*@ non_null @*/ Object a3) {
	return new SPair(SExp.fancyMake(a1), make(a2,a3));
    }

    /**
     * @return a S-expression representing a 4-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments. <p>
     */
    public static /*@ pure non_null @*/ SList make(/*@ non_null @*/ Object a1,
                                                   /*@ non_null @*/ Object a2,
                                                   /*@ non_null @*/ Object a3,
                                                   /*@ non_null @*/ Object a4) {
	return new SPair(SExp.fancyMake(a1), make(a2,a3,a4));
    }

    /**
     * @return a S-expression representing a 5-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments. <p>
     */
    public static /*@ pure non_null @*/ SList make(/*@ non_null @*/ Object a1,
                                                   /*@ non_null @*/ Object a2,
                                                   /*@ non_null @*/ Object a3,
                                                   /*@ non_null @*/ Object a4,
                                                   /*@ non_null @*/ Object a5) {
	return new SPair(SExp.fancyMake(a1), make(a2,a3,a4,a5));
    }

    /**
     * @return a S-expression representing a 6-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments. <p>
     */
    public static /*@ pure non_null @*/ SList make(/*@ non_null @*/ Object a1,
                                                   /*@ non_null @*/ Object a2,
                                                   /*@ non_null @*/ Object a3,
                                                   /*@ non_null @*/ Object a4,
                                                   /*@ non_null @*/ Object a5,
                                                   /*@ non_null @*/ Object a6) {
	return new SPair(SExp.fancyMake(a1), make(a2,a3,a4,a5,a6));
    }

    /**
     * @return a S-expression representing a 7-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments. <p>
     */
    public static /*@ pure non_null @*/ SList make(/*@ non_null @*/ Object a1,
                                                   /*@ non_null @*/ Object a2,
                                                   /*@ non_null @*/ Object a3,
                                                   /*@ non_null @*/ Object a4,
                                                   /*@ non_null @*/ Object a5,
                                                   /*@ non_null @*/ Object a6,
                                                   /*@ non_null @*/ Object a7) {
	return new SPair(SExp.fancyMake(a1), make(a2,a3,a4,a5,a6,a7));
    }

    /**
     * @return a S-expression representing a 8-element list whose
     * contents are the result of applying {@link SExp#fancyMake(Object)}
     * to our arguments.
     */
    public static /*@ pure non_null @*/ SList make(/*@ non_null @*/ Object a1,
                                                   /*@ non_null @*/ Object a2,
                                                   /*@ non_null @*/ Object a3,
                                                   /*@ non_null @*/ Object a4,
                                                   /*@ non_null @*/ Object a5,
                                                   /*@ non_null @*/ Object a6,
                                                   /*@ non_null @*/ Object a7,
                                                   /*@ non_null @*/ Object a8) {
	return new SPair(SExp.fancyMake(a1), make(a2,a3,a4,a5,a6,a7,a8));
    }


    /**
     * @return a S-expression representing a list whose
     * contents are the contents of a given array.
     */
    //@ public normal_behavior
    //@   requires \nonnullelements(a);
    //@   ensures \result != null;
    public static /*@ pure non_null @*/ SList fromArray(/*@ non_null @*/ SExp[] a) {
	SList l = make();

	for (int i = a.length-1; i >= 0; i--)
	    l = new SPair(a[i], l);

	return l;
    }

    //@ also
    //@ public normal_behavior
    //@   ensures \result;
    public /*@ pure @*/ boolean isList() {
	return true;
    }

    //@ also
    //@ public normal_behavior
    //@   ensures \result == this;
    public /*@ pure non_null @*/ SList getList() {
	return this;
    }

    /**
     * @return a flag indicating if we are we an empty list.
     */
    public abstract /*@ pure @*/ boolean isEmpty();

    /**
     * @return our length.
     */
    //@ public normal_behavior
    //@   ensures \result >= 0;
    public abstract /*@ pure @*/ int length();


    /**
     * Convert the current list to an fresh array.
     * @return an array
     */
    public SExp[] toArray()
    {
    	SExp[] res = new SExp[length()];
    	SList remaining = this;
    	int i = 0;

    	while (remaining instanceof SPair) {
    	    SPair p = (SPair)remaining;
    	    res[i++] = p.head;
    	    remaining = p.tail;
    	}
    	
    	return res;
    }

    /**
     * If we represent a non-empty list, return it as a
     * <code>SPair</code>; otherwise, throw <code>SExpTypeError</code>.
     */
    //@ exceptional_behavior
    //@   signals (SExpTypeError) true;
    /*package*/ /*@ non_null pure @*/ SPair getPair() throws SExpTypeError {
	throw new SExpTypeError();
    }

    /**
     * @return our element at index <code>i</code> (indices start at
     * 0).  If we are not long enough to have an element at that
     * index, then <code>SExpTypeError</code> is thrown.
     */
    //@ public normal_behavior
    //@   requires i >= 0;
    //@   ensures \result != null;
    //@ also
    //@ public exceptional_behavior
    //@   signals (SExpTypeError) true;
    public /*@ pure @*/ /*@ non_null */ SExp at(int i) throws SExpTypeError {
	SPair ptr = getPair();

	for (; i > 0; i--)
	    ptr = ptr.tail.getPair();
	
	return ptr.head;
    }

    /**
     * Modify the list in place by set the <code>i</code>th element to
     * <code>s</code>.
     */
    //@ public normal_behavior
    //@   requires 0 <= i;
    //@   requires i < this.length();
    //@   modifies \everything;
    //@   ensures at(i) == s;
    public void setAt(int i, /*@ non_null @*/ SExp s) throws SExpTypeError {
	SPair ptr = getPair();

	for (; i > 0; i--)
	    ptr = ptr.tail.getPair();
	
	ptr.head = s;
    }
    
    /**
     * @return the functional result of appending another list to
     * us.
     */
    //@ public normal_behavior
    //@   requires x != null;
    //@   modifies \everything;
    //@   ensures \result != null;
    //@ also
    //@ public normal_behavior
    //@   requires (this instanceof SNil);
    //@   modifies \everything;
    //@   ensures \result == x;
    public /*@ non_null @*/ SList append(/*@ non_null @*/ SList x) {
	if (this instanceof SNil)
	    return x;
	else {
	    SPair us = (SPair)this;		//@ nowarn Cast;

	    return new SPair(us.head, us.tail.append(x));
	}
    }

    /** 
     * Destructive list reversal
     */
    //@ public normal_behavior
    //@   requires l != null;
    //@   modifies \everything;
    //@   ensures \result != null;
    public static /*@ non_null */ SList reverseD(/*@ non_null */ SList l){
	SList res = SNil.getNil();
	while (! (l instanceof SNil)){
	    SList temp = res; 
            res = l; l = ((SPair)l).tail; ((SPair)res).tail = temp;
        }
	return res;
    }

    /**
     * @return the functional result of adding a given element at the
     * front of us.
     *
     * Precondition: <code>x</code> is non-null.<p>
     */
    //@ public normal_behavior
    //@   requires x != null;
    //@   modifies \everything;
    //@   ensures \result != null;
    //@   ensures \result.at(0) == x;
    public /*@ non_null */ SList addFront(/*@ non_null */ SExp x) {
	return new SPair(x, this);
    }

    /**
     * @return the functional result of adding a given element at the
     * end of us.
     *
     * <p> This function is likely to be slower than <code>addFront</code>.
     */
    //@ public normal_behavior
    //@   requires x != null;
    //@   modifies \everything;
    //@   ensures \result != null;
    //@   ensures \result.at(\result.length()) == x;
    public /*@ non_null */ SList addEnd(/*@ non_null */ SExp x) {
	return append(make(x));	
    }

    /**
     * Print a textual representation of us on a given
     * <code>PrintStream</code>.
     *
     * <p> Note: This routine will take a <code>PrintWriter</code>
     * instead when we switch to a more recent version of JDK. </p>
     */
    public /*@ pure @*/ void print(/*@ non_null @*/ PrintStream out) {
	// Is this the first item to be printed?
	boolean first = true;

	// The remaining list to be printed:
	SList remaining = this;

	out.print("(");
	while (remaining instanceof SPair) {
	    if (!first)
		out.print(" ");

	    SPair p = (SPair)remaining;
	    p.head.print(out);
	    first = false;
	    remaining = p.tail;
	}
	out.print(")");
    }

    /**
     * Pretty-print a textual representation of us on a given
     * <code>PrintStream</code>.
     *
     * <p> Note: This routine will take a <code>PrintWriter</code>
     * instead when we switch to a more recent version of JDK. </p>
     */
    public /*@ pure @*/ void prettyPrint(/*@ non_null @*/ PrintStream out) {
	try {
	    if (!specialPrint(out)) {
		if (this instanceof SNil)
		    out.print("()");
		else {
		    // by default, print (op arg1 ... argn) as 
		    // op(arg1 ... argn)
		    SPair p = (SPair) this;
		    SList remaining = p.tail;
		    boolean first = true;
		    p.head.prettyPrint(out);
		    out.print("(");
		    while (remaining instanceof SPair) {
			if (!first)
			    out.print(" ");				
			p = (SPair) remaining;
			p.head.prettyPrint(out);
			first = false;
			remaining = p.tail;
		    }
		    out.print(")");
		}
	    }
	}
	catch (SExpTypeError e) {
	    javafe.util.Assert.fail("Error:  Out of bounds access to SList");
	}
    }

    /**
     * Specially print a textual representation of us on a given
     * <code>PrintStream</code>.  This is a subroutine used in
     * pretty-printing.
     *
     * <p> Note: This routine will take a <code>PrintWriter</code>
     * instead when we switch to a more recent version of JDK. </p>
     */
    private boolean specialPrint(/*@ non_null @*/ PrintStream out) throws SExpTypeError {
	int len = length();
	if (len < 2 || len > 3)
	    return false;
	String first = at(0).toString();
	String op;
	if (len == 2) {
	    if (first.equals("NOT") || first.equals("boolNot") 
		|| first.equals("integralNot"))
		op = "!";
	    else if (first.equals("floatingNeg") ||
		     first.equals("integralNeg"))
		op = "-";
	    else if (first.equals("_array")) {
		at(1).prettyPrint(out);
		out.print("[]");
		return true;
	    }
	    else
		return false;
	    out.print(op);
	    at(1).prettyPrint(out);
	    return true;
	}
	else {
	    if (first.equals("EQ")) {
		at(1).prettyPrint(out);
		out.print(" == ");
		at(2).prettyPrint(out);
		return true;
	    }
	    else if (first.equals("NEQ")) {
		at(1).prettyPrint(out);
		out.print(" != ");
		at(2).prettyPrint(out);
		return true;
	    }
	    else if (first.equals("<:") || first.equals("typeLE")) {
		at(1).prettyPrint(out);
		out.print(" <: ");
		at(2).prettyPrint(out);
		return true;
	    }
	    else if (first.equals("is")) {
		out.print("typeof(");
		at(1).prettyPrint(out);
		out.print(") <: ");
		at(2).prettyPrint(out);
		return true;
	    }
	    else if (first.equals("select")) {
		SExp arg1 = at(1);
		SExp arg2 = at(2);
	    
		// if this select is an array access, print it appropriately
		// NOTE:  This may screw up the printing of variables that
		// contain the string "elems" in their name
		if (arg1.isList()) {
		    SList arg1L = (SList) arg1;
		    if (arg1L.length() == 3 && 
			arg1L.at(0).toString().equals("select") &&
			arg1L.at(1).toString().startsWith("elems")) {
			arg1L.at(2).prettyPrint(out);
			out.print("[");
			arg2.prettyPrint(out);
			out.print("]");
			return true;
		    }
		}
		//parenthesize identifiers containing special characters
		if (arg2.isAtom() && 
		    Atom.printableVersion(arg2.toString()).startsWith("|"))
		    {
			out.print("(");
			arg2.prettyPrint(out);
			out.print(")");
		    }
		else
		    arg2.prettyPrint(out);
		out.print(".");
		if (arg1.isAtom() && 
		    Atom.printableVersion(arg1.toString()).startsWith("|"))
		    {
			out.print("(");
			arg1.prettyPrint(out);
			out.print(")");
		    }
		else
		    arg1.prettyPrint(out);
		return true;
	    }
	    // print all of the infix operators correctly
	    else if (first.equals("anyEQ") || first.equals("boolEq") ||
		     first.equals("floatingEQ") || 
		     first.equals("integralEQ") || first.equals("refEQ")
		     || first.equals("typeEQ"))
		op = " == ";	      				     
	    else if (first.equals("anyNE") || first.equals("boolNE") ||
		     first.equals("floatingNE") || 
		     first.equals("integralNE") || first.equals("refNE")
		     || first.equals("typeNE"))
		op = " != ";
	    else if (first.equals("floatingAdd") || 
		     first.equals("integralAdd") || 
		     first.equals("stringCat") || first.equals("+"))
		op = " + ";
	    else if (first.equals("floatingSub") || 
		     first.equals("integralSub") || first.equals("-"))
		op = " - ";
						     
	    else if (first.equals("floatingMul") || 
		     first.equals("integralMul") || first.equals("*"))
		op = " * ";
    	    else if (first.equals("floatingDiv") || 
		     first.equals("integralDiv") || first.equals("/"))
		op = " / ";
    	    else if (first.equals("floatingMod") || 
		     first.equals("integralMod"))
		op = " % ";
	    else if (first.equals("allocLT") || first.equals("floatingLT")
		     || first.equals("integralLT") || first.equals("lockLT")
		     || first.equals("<"))
		op = " < ";
	    else if (first.equals("allocLE") || first.equals("floatingLE")
		     || first.equals("integralLE") || first.equals("lockLE")
		     || first.equals("<="))
		op = " <= ";	    
    	    else if (first.equals("floatingGT") || 
		     first.equals("integralGT") || first.equals(">"))
		op = " > ";
    	    else if (first.equals("floatingGE") || 
		     first.equals("integralGE") || first.equals(">="))
		op = " >= ";
	    else if (first.equals("boolAnd") || first.equals("integralAnd"))
		op = " && ";
	    else if (first.equals("boolOr") || first.equals("integralOr"))
		op = " || ";
	    else if (first.equals("boolImplies"))
		op = " ==> ";
	    else
		return false;
	    
	    out.print("(");
	    at(1).prettyPrint(out);
	    out.print(op);
	    at(2).prettyPrint(out);
	    out.print(")");
	    return true;
	}
    }

    /**
     * A simple test routine
     */
    public static void main(/*@ non_null @*/ String[] args) throws SExpTypeError {
	make().print(System.out);
	System.out.println();
	make("a").print(System.out);
	System.out.println();
	make("a","b").print(System.out);
	System.out.println();
	make("a","b","c").print(System.out);
	System.out.println();
	make("a","b","c","d").print(System.out);
	System.out.println();
	make("a","b","c","d","e").print(System.out);
	System.out.println();
	make("a","b","c","d","e","f").print(System.out);
	System.out.println();
	make("a","b","c","d","e","f","g").print(System.out);
	System.out.println();
	make("a","b","c","d","e","f","g","h").print(System.out);
	System.out.println();
	System.out.println();

	make("a", make("b", "c"), make("d", "e", make(), "f"))
		.print(System.out);
	System.out.println();

	SExp[] elements = new SExp[10];
	for (int i=0; i<10; i++)
	    elements[i] = SExp.make(i);
	fromArray(elements).print(System.out);
	System.out.println();
	System.out.println();

	make("a", "b").append(make()).append(make("c", "d", "e"))
		.print(System.out);
	System.out.println();

	make("a", "b").addFront(SExp.fancyMake("front")).addEnd(SExp.make(42))
		.print(System.out);
	System.out.println();
    }
}
