/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;


import java.io.*;


/**
 ** S-Expression datatype for use in interfacing to the ESC Theorem
 ** prover, Simplify. <p>
 **
 **
 ** Considered as a language, the grammer of s-expressions is as
 ** follows:<p>
 **
 **   <code>SExp</code> ::= <code>Atom</code> | <code>int</code>
 **	| <code>SList</code><br>
 **   <code>SList</code> ::= ( <code>SExp</code>* )<p>
 **
 ** That is, s-expressions consist of either an atom, an integer, or a
 ** possibly-empty list of s-expressions.  <code>Atom</code>s are
 ** interned <code>String</code>s used to represent symbols.<p>
 **
 **
 ** Methods are available to test whether an <code>SExp</code> is an
 ** <code>Atom</code>, an integer, or a list.  If the "type" of an
 ** <code>SExp</code> is known, it can be converted into what it
 ** represents.  If an attempt is made to convert an <code>SExp</code>
 ** to an incorrect "type", the checked exception
 ** <code>SExpTypeError<code> will be thrown.<p>
 **
 ** <code>SExp</code>s can be printed onto a <code>PrintStream</code>.
 **
 **
 ** @see SExpTypeError
 ** @see Atom
 ** @see SList
 **/

public abstract class SExp {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     ***************************************************/

    // <code>SExp</code> may not be extended outside this package:
    /*package*/ SExp() {}


    /**
     ** Return an S-expression representing a given integer. <p>
     **
     ** This is faster than using <code>fancyMake(new
     ** Integer(i))</code>.<p>
     **/
    //@ ensures \result!=null
    public static SExp make(int i) {
	return SInt.fromInt(i);
    }

    /**
     ** Return an S-expression representing an integer (passed wrapped in an
     ** <code>Integer</code>), an atom (specified via a
     ** <code>String</code>), or an existing S-expression (this case
     ** leaves the argument unchanged). <p>
     **
     ** Preconditon: <code>o</code>'s type is a subtype of
     **		     <code>String</code>, <code>Integer</code>, or
     **		     <code>SExp</code>; <code>o!=null</code>.<p> 
     **/
    //@ requires o!=null
    //@ ensures \result!=null
    public static SExp fancyMake(Object o) {
	javafe.util.Assert.precondition(o!=null);
	if (o instanceof SExp)
	    return (SExp)o;
	else if (o instanceof String)
	    return Atom.fromString((String)o);
	else if (o instanceof Integer)
	    return SInt.fromInt(((Integer)o).intValue());
	else {
	    javafe.util.Assert.fail(
		"Non-String/Integer/SExp passed to SExp.make");
	    return null;		// make compiler happy...
	}
    }

    /***************************************************
     *                                                 *
     * Accessors:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Do we represent an atom?
     **/
    public boolean isAtom() {
	return false;
    }

    /**
     ** Do we represent an integer?
     **/
    public boolean isInteger() {
	return false;
    }

    /**
     ** Do we represent a list?
     **/
    public boolean isList() {
	return false;
    }


    /**
     ** If we represent an atom, return it as an <code>Atom</code>; otherwise,
     ** throw <code>SExpTypeError</code>.
     **/
    //@ ensures \result!=null
    public Atom getAtom() throws SExpTypeError {
	throw new SExpTypeError();
    }

    /**
     ** If we represent an integer, return it as an <code>int</code>;
     ** otherwise, throw <code>SExpTypeError</code>.
     **/
    public int getInteger() throws SExpTypeError {
	throw new SExpTypeError();
    }

    /**
     ** If we represent a list, return it as an <code>SList</code>;
     ** otherwise, throw <code>SExpTypeError</code>.
     **/
    //@ ensures \result!=null
    public SList getList() throws SExpTypeError {
	throw new SExpTypeError();
    }
    
    /***************************************************
     *                                                 *
     * Printing:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Print a textual representation of us on a given
     ** <code>PrintStream</code>. <p>
     **
     ** Note: This routine will take a <code>PrintWriter</code> instead
     ** when we switch to a more recent version of JDK.<p>
     **/
    //@ requires out!=null
    public abstract void print(PrintStream out);


    /**
     ** Print a textual representation of us on
     ** <code>System.out</code>. <p>
     **
     ** (This is a convenience function.)<p>
     **/
    public void print() {
	print(System.out);
    }


    /**
     ** Pretty print a textual representation of us on the given
     ** <code>PrintStream</code>. <p>
     **
     ** Note: This routine will take a <code>PrintWriter</code> instead
     ** when we switch to a more recent version of JDK.<p>
     **
     **/
    public abstract void prettyPrint(PrintStream out);


    /**
     ** Return a textual representation of us as a String.
     **/
    public String toString() {
	ByteArrayOutputStream baos = new ByteArrayOutputStream();
	PrintStream ps = new PrintStream(baos);
	print(ps);
	String result = baos.toString();
	ps.close();
	return result;
    }


    /***************************************************
     *                                                 *
     * Test methods:				       *
     *                                                 *
     ***************************************************/

    /**
     ** A simple test routine
     **/
    public static void main(String[] args) throws SExpTypeError {
	display(make(14));

	display(fancyMake("hello"));
	display(fancyMake(new Integer(42)));

	display(fancyMake(SList.make()));
	display(fancyMake(SList.make("NOT", make(0))));
	display(fancyMake(SList.make("a", "b", "c", "d", "e")));


        System.out.println(
            SList.make().equals(
            SList.make()));

        System.out.println(
            SList.make("a", "b", "c", "d", "e").equals(
            SList.make("a", "b", "c", "d", "3")));

        System.out.println(
            SList.make("a", "b", "c", "d", "e").equals(
            SList.make("a", "b", "c", "d", "f")));

        System.out.println(
            SList.make("NOT", SExp.make(3)).equals(
            SList.make("NOT", SExp.make(2))));


        System.out.println(
            SList.make("NOT", SExp.make(3)).equals(
            SList.make("NOT", SExp.make(3))));


	// Intentionally invoke a SExpTypeError...
	System.out.println();
	System.out.println("Type error case:");
	make(14).getAtom();
    }


    /**
     ** Display a <code>SExp</code> verbosely, using all its accessor
     ** methods. <p>
     **
     ** This method is intended mainly for test use.<p>
     **/
    //@ requires x!=null
    public static void display(SExp x) throws SExpTypeError {
	if (x.isAtom()) {
	    System.out.print("[Atom]  "+x.getAtom().toString());
	}
	if (x.isInteger()) {
	    System.out.print("[int]  "+x.getInteger());
	}
	if (x.isList()) {
	    SList l = x.getList();
	    int n = l.length();
	    System.out.print("[" + (l.isEmpty() ? "empty " : "") +
				n + "-element SList]  ");
	    for (int i=0; i<n; i++) {
		if (i!=0)
		    System.out.print(" ");
		System.out.print("@" + i + "=");
		l.at(i).print(System.out);
	    }
	}

	System.out.print("  =>  ");
	x.print(System.out);
	System.out.println();
    }

    

}
