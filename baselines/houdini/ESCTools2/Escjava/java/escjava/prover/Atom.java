/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;


import java.util.Hashtable;
import java.io.*;


/**
 ** <code>Atom</code>s are S-expressions representing symbols. <p>
 **
 ** Interned <code>String</code>s are used to represent symbols.
 ** Accordingly, two <code>Atom</code>s are equal (via <code>==</code>)
 ** iff they represent the same symbol.<p>
 **/

final public class Atom extends SExp {
  
  /***************************************************
   *                                                 *
   * Class fields:                                   *
   *                                                 *
   ***************************************************/
  
  /**
   ** Our map from interned <code>String</code>s to already created
   ** non-null <code>Atom</code>s.
   **/
  //@ invariant map != null;
  //+@ invariant map.keyType == \type(String);
  //+@ invariant map.elementType == \type(Atom);
  //@ spec_public
  static private Hashtable map = new Hashtable(200);
  
  
  /***************************************************
   *                                                 *
   * Instance fields:                                *
   *                                                 *
   ***************************************************/
  
  /**
   ** The symbol we represent; always already interned.
   **/
  //@ invariant value != null;
  //@ spec_public
  private String value;
  
  
  /***************************************************
   *                                                 *
   * Creation:                                       *
   *                                                 *
   ***************************************************/
  
  /**
   ** Create an <code>Atom</code> representing a given
   ** symbol.  Clients are allowed to create
   ** <code>Atom</code>s only by calling <code>fromString</code> so we can
   ** intern.<p>
   **
   ** Precondition: <code>symbol</code> must already have been
   ** interned.<p>
   **/
  //@ requires symbol != null;
  private Atom(String symbol) {
    value = symbol;
    
    //+@ set map.keyType = \type(String);
    //+@ set map.elementType = \type(Atom);
    
    map.put(value, this);		// Save us in the interning table
  }
  
  /**
   ** Create a <code>Atom</code> representing a given symbol. <p>
   **
   ** This function always returns the same <code>Atom</code> when
   ** called on equal <code>String</code>s.<p>
   **/
  //@ requires symbol != null;
  //@ ensures \result != null;
  public static Atom fromString(String symbol) {
    String key = symbol.intern();
    Atom existing = (Atom)map.get(key);
    if (existing != null)
      return existing;
    return new Atom(key);
  }
  
  
  /***************************************************
   *                                                 *
   * Simple Accessors:                               *
   *                                                 *
   ***************************************************/
  
  /**
   ** Do we represent an atom?
   **/
  public boolean isAtom() {
    return true;
  }
  
  
  /**
   ** If we represent an atom, return it as an <code>Atom</code>; otherwise,
   ** throw <code>SExpTypeError</code>.
   **/
  public Atom getAtom() {
    return this;
  }
  
  
  /**
   ** Return the interned <code>String</code> for the symbol we
   ** represent.
   **/
  public /*@non_null*/String toString() {
    return value;
  }
  
  
  /**
   ** Return true if this atom and object o are the same atom.
   **/
  public boolean equals(Object o) {
    return this == o;
  }
  
  /***************************************************
   *                                                 *
   * Printing:                                       *
   *                                                 *
   ***************************************************/
  
  /**
   ** The list of special symbols that don't need to be quoted when by
   ** themselves.
   **/
  //@ invariant special != null;
  public final static String special = "!#$%&*+-./:<=>?@[]^_{}";
  
  
  /**
   ** Returns the printable version (e.g., with escape codes added as
   ** needed) of an S-expression symbol's name.
   **/
  //@ requires symbol != null;
  public static String printableVersion(String symbol) {
    if (symbol.equals(""))
      return "||";
    
    // Determine if symbol fits <alpha><alphanumeric>*:
    boolean identifier = true;
    for (int i=0; i<symbol.length(); i++) {
      char c = symbol.charAt(i);
      if ((c>='a' && c<='z') || (c>='A' && c<='Z'))
        continue;
      // Simplify doesn't like ids starting with _
      if ((c>='0' && c<='9') || (c=='_') && i>0)
        continue;
      identifier = false; break;
    }
    
    if (identifier)
      return symbol;
    
    
    // Determine if symbol fits <special>+:
    boolean operator = true;
    for (int i=0; i<symbol.length(); i++) {
      char c = symbol.charAt(i);
      if (special.indexOf(c)==-1) {
        operator = false;
        break;
      }
    }
    
    if (operator)
      return symbol;
    
    // don't escape a symbol that is already escaped
    if (symbol.startsWith("|") && symbol.endsWith("|")) {
      return symbol;
    }
    
    // Deal with escape codes later... HACK!!!!
    return "|" + symbol + "|";
  }
  
  
  /**
   ** Print a textual representation of us on a given
   ** <code>PrintStream</code>. <p>
   **
   ** Note: This routine will take a <code>PrintWriter</code> instead
   ** when we switch to a more recent version of JDK.<p>
   **/
  public void print(/*@non_null*/PrintStream out) {
    out.print(printableVersion(value));
  }
  
  
  /**
   ** Pretty print a textual representation of us on a given
   ** <code>PrintStream</code>. <p>
   **
   ** Note: This routine will take a <code>PrintWriter</code> instead
   ** when we switch to a more recent version of JDK.<p>
   **/
  public void prettyPrint(/*@non_null*/PrintStream out) {
    out.print(value);
  }
  
  
  /***************************************************
   *                                                 *
   * Test methods:                                   *
   *                                                 *
   ***************************************************/
  
  /**
   ** A simple test routine
   **/
  public static void main(String[] args) {
    Atom a1 = fromString("a");
    Atom b  = fromString("b");
    Atom a2 = fromString("a"+"");
    
    System.out.println("same: " + (a1==a2));
    System.out.println("different: " + (a1==b));
    System.out.println();
    
    fromString("").print(System.out); System.out.println();
    System.out.println();
    
    // identifiers:
    fromString("abcde").print(System.out); System.out.println();
    fromString("a253").print(System.out); System.out.println();
    fromString("a2b3c4de").print(System.out); System.out.println();
    fromString("a124b234").print(System.out); System.out.println();
    System.out.println();
    
    // non-identifiers:
    fromString("1abcd").print(System.out); System.out.println();
    fromString("ab+d").print(System.out); System.out.println();
    fromString("-a2b3c4de").print(System.out); System.out.println();
    System.out.println();
    System.out.println();
    
    
    // operators:
    fromString(special).print(System.out); System.out.println();
    fromString("=>").print(System.out); System.out.println();
    fromString("<==>").print(System.out); System.out.println();
    fromString("+").print(System.out); System.out.println();
    System.out.println();
    
    // non-operators:
    fromString("~").print(System.out); System.out.println();
    fromString("=~>").print(System.out); System.out.println();
    fromString("a+").print(System.out); System.out.println();
    fromString("~==+").print(System.out); System.out.println();
    System.out.println();
    
    // Test escape codes later...
  }
}
