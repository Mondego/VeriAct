/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.ast;

//@ model import javafe.tc.TagConstants;
import javafe.parser.ParserTagConstants;
import javafe.util.Assert;

/**
 * An <code>Identifier</code> is a symbol, that is, a sequence of
 * characters.  <code>Identifier</code>s are interned: for any given
 * sequence of characters, we create exactly one instance of
 * <code>Identifier</code> to represent it (we say that the sequence
 * of characters is <I>associated with</I> this
 * <code>Identifier</code> instance).  This allows us to use object
 * equality (that is, <code>==</code>) to check symbol equality.
 *
 * <p> This class is thread-safe: multiple threads can enter its any
 * of its methods (static or non-static) simultaneously. </p>
 */

public final class Identifier
{
  //// Private, static variables

  /** Size of intern table. */
  private static final int TABLE_SIZE = 128;

  /** Table containing every instance of <code>Identifier</code>
     created.  If a symbol <TT>s</TT> has been interned, its
     associated <code>Identifier</code> is found by hashing it to
     integer <TT>h</TT> such that <TT>0 &lt;= h &lt;= TABLE_SIZE</TT>,
     looking up <TT>v = chains[h]</TT>, which is a an array of arrays
     of <code>Identifier</code>s, then searching <TT>v</TT> for the
     <code>Identifier</code> associated with <TT>s</TT>.  If no such
     element exists, then <TT>s</TT> hasn't been interned yet.

     <P> This table is only extended, old parts are never updated.
     Thus, reading the table can occur without any locks being held.
     Extension of the table is protected by the mutex associated with
     the table itself (that is, the object pointed to by
     <code>chains</code>. */

  /*@ invariant (\forall int i; 0<=i && i<chains.length
	==> chains[i]==null || chains[i].length>0); */
  //@ spec_public
  private static final Identifier chains[][] = new Identifier[TABLE_SIZE][];


  /** Initial size of chains inside the table. */
  private static final int INITIAL_CHAIN_SIZE = 4;


  // Private, instance variables

  /** Sequence of characters represented by this Identifier (never
    <code>null</code>). */
  //@ invariant chars != null;

    //private char[] chars; //@ in objectState;
    public char[] chars; //@ in objectState;

  /** Memoization of <code>String.valueOf(chars, 0,
    chars.length)</code>; may be <code>null</code>.  This variable may
    be written exactly once. */
  private String equiv; //@ in privateState;


  //// "Friend", variables

  /** Constant used for hashing.  The hash function used to hash a
    <i>n</i>-character identifier <code>idn</code> is to sum
    <code>HC</code>^<i>(n-(i+1))</i><code>*idn[</code><i>i</i><code>]</code>.
    Note that this is the same hash function used by Java 1.1 for
    hashing <code>String</code> objects.  */
  static final int HC = 31;


    /**
     * This field defaults to <code>TagConstants.IDENT</code>, but
     * is set to other values by the scanner to indicate keywords.
     */
    /*@ invariant tokenType != TagConstants.BOOLEANLIT &&
                  tokenType != TagConstants.INTLIT &&
                  tokenType != TagConstants.LONGLIT &&
                  tokenType != TagConstants.FLOATLIT &&
                  tokenType != TagConstants.DOUBLELIT &&
                  tokenType != TagConstants.STRINGLIT &&
                  tokenType != TagConstants.CHARLIT &&
                  tokenType != TagConstants.LEXICALPRAGMA &&
                  tokenType != TagConstants.MODIFIERPRAGMA &&
                  tokenType != TagConstants.STMTPRAGMA &&
                  tokenType != TagConstants.TYPEDECLELEMPRAGMA &&
                  tokenType != TagConstants.TYPEMODIFIERPRAGMA; */
    int tokenType = ParserTagConstants.IDENT;


  //@ requires text != null;
  //@ requires 0<=textlen && textlen<=text.length;
  private Identifier(char[] text, int textlen, int hashcode) {
    this.chars = new char[textlen];
    System.arraycopy(text, 0, this.chars, 0, textlen);
  }


  /** Returns the <code>Identifier</code> associated with
    <TT>s</TT>. */
  public static /*@ non_null */ Identifier intern(/*@ non_null */ String s) {
    // Expensive way of doing things, but at least we don't have
    // to rewrite the above code...

    int len = s.length();
    char[] chars = new char[len];
    s.getChars(0, len, chars, 0);
    return intern(chars, len, hash(chars, len));
  }


  /** Intern a sequence of characters with a pre-computed hashcode.

    <BR> Requires: <code>hashcode = Identifier.hash(text, textlen)</code> 

    <P> Ensures: returns the <code>Identifier</code> instance
    associated with the symbol consisting of the first
    <code>textlen</code> characters of <code>text</code>. */

  //@ requires 0<=textlen && textlen<=text.length;
  static /*@non_null*/Identifier intern(/*@non_null*/char[] text, int textlen, int hashcode) {

    int h = Math.abs(hashcode) % TABLE_SIZE;
    Identifier[] chain = chains[h];

    // See if it's already in the table
    int index = 0;
    if (chain == null) {
      chain = chains[h] = new Identifier[INITIAL_CHAIN_SIZE];
      return (chain[0] = new Identifier(text, textlen, hashcode));
    } else
    searchloop:
      for( ; index < chain.length; index++) {
	Identifier k = chain[index];
	if (k == null) break;
	char[] chars = k.chars;
	if (chars.length == textlen) {
	  for(int j = 0; j < textlen; j++)
	    if (text[j] != chars[j]) continue searchloop;
	  return k;
	}
      }

    // Add it to the table
    if (index == chain.length) { // Expand this chain
      chain = new Identifier[2*index];
      System.arraycopy(chains[h], 0, chain, 0, index);
      chains[h] = chain;
    }
    return (chain[index] = new Identifier(text, textlen, hashcode));
  }

 //@ requires s != null;
  static int hash(String s) {
    int len = s.length();
    int hashcode = 0;
    for(int i = 0; i < len; i++)
      hashcode = HC*hashcode + s.charAt(i);
    return hashcode;
  }

  //@ requires text != null;
  //@ requires 0<=textlen && textlen<=text.length;
  static int hash(char[] text, int textlen) {
    int hashcode = 0;
    for(int i = 0; i < textlen; i++)
      hashcode = HC*hashcode + text[i];
    return hashcode;
  }


  /** Return a string containing the symbol associated with
    <code>this</code>. */
  public /*@non_null*/String toString() {
    if (equiv == null) equiv = String.valueOf(chars);
    return equiv;
  }

  public int hashCode() {
    return hash(chars, chars.length);
  }
  
  public boolean equals(/*@ nullable @*/ Object o) {
    return this == o;
  }

  /** Return true if all invariants are satisfied. */
  public static void check() {
    if (chains.length != TABLE_SIZE) Assert.fail("Bad size");
    for(int hashcode = 0; hashcode < TABLE_SIZE; hashcode++) {
      Identifier[] chain = chains[hashcode];
      if (chain != null) {
	int i;
	// First in chain must be good
	for(i = 0; i < chain.length && chain[i] != null; i++) {
	  Identifier idn = chain[i];
	  Identifier idn2 = slowFind(String.valueOf(idn.chars));
	  if (idn != idn2) Assert.fail("Missing entry");
	  // if (idn.hashcode != hash(idn.chars, idn.chars.length))
	  //   Assert.fail("Bad hashcode");
	  if (hashcode !=
	      Math.abs(hash(idn.chars, idn.chars.length)) % TABLE_SIZE)
	    Assert.fail("Bad position in table");
	  if (idn.equiv != null)
	    if (! idn.equiv.equals(String.valueOf(idn.chars)))
	      Assert.fail("Bad equiv field");
	}
	// Rest in chain must be null;
	for( ; i < chain.length; i++)
	  if (chain[i] != null) Assert.fail("Bad chain");
      }
    }
  }


  /** Used by <code>check</code>; checks for duplicates. */
  //@ requires s != null;
  private static Identifier slowFind(String s) {
    Identifier result = null;
    for(int h = 0; h < TABLE_SIZE; h++) {
      Identifier[] chain = chains[h];
      if (chain != null)
	for(int i = 0; i < chain.length && chain[i] != null; i++) {
	  Identifier idn = chain[i];
	  if (s.equals(String.valueOf(idn.chars)))
	    if (result == null) result = idn;
	    else Assert.fail("Duplicate entry (" + s + ')');
	}
    }
    if (result == null) Assert.fail("Strange error");
    return result;
  }

    //@ requires argv != null;
    /*@ requires (\forall int i; (0<=i && i<argv.length)
		==> argv[i] != null); */
  public static void main(String[] argv) {
    String stem = "gensym";
    int stemlen = stem.length();

    Identifier[] table = new Identifier[2*TABLE_SIZE*INITIAL_CHAIN_SIZE];


    // Build table
    int stemhash = HC*HC*HC*HC*hash(stem);
    char[] buffer = new char[stemlen + 4];
    stem.getChars(0, stemlen, buffer, 0);
    for(int i = 0; i < table.length; i++) {
      String num = Integer.toHexString(0x1000 + i);
      if (num.length() != 4)
	Assert.fail("Ooops");
      num.getChars(0, 4, buffer, stemlen);
      table[i] = intern(buffer, stemlen+4, stemhash + hash(num));
    }

    // Check table
    for(int i = 0; i < table.length; i++) {
      String key = stem + Integer.toHexString(0x1000 + i);
      Identifier idn = intern(key);
      if (table[i] != idn) Assert.fail("== failed " + i);
      if (! key.equals(idn.toString()))
	Assert.fail("toString failed");
    }

    // Check invariants of table
    check();
  }
}
