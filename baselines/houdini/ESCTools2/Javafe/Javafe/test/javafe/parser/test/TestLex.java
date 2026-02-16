/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser.test;

import java.io.IOException;
import java.util.Random;
import javafe.ast.*;
import javafe.parser.*;
import javafe.tc.TagConstants;
import javafe.util.*;

/**
 * Tokenizes standard input and prints 
 * {@link javafe.ast.PrettyPrint#toString(int)} 
 * of the resulting stream, one token per output line.
 */

public class TestLex implements PragmaParser
{
  /**
   * Fail with a specific error message.
   *
   * @param msg the error message to print to {@link System.err}. 
   */
  //@ ensures false;
  public static void fatal(String msg) {
    System.err.println("Fatal error: " + msg);
    System.exit(99);
  }
  
  /**
   * Tokenizes standard input and prints 
   * {@link javafe.ast.PrettyPrint#toString(int)} 
   * of the resulting stream, one token per output line.
   *
   * If the command-line argument "javakeywords" is used, then the
   * Java keywords are added to the lexer as recognisable tokens.  If
   * the command-line argument "lookahead" is used, then a random
   * lookahead between 0 and 9 is chosen to test the lexer; if it is
   * not passed, then no lookahead is used.  If the command-line
   * argument "parsepragmas" is used then two test operators, "==>"
   * and "<:", are added to the lexer for testing using
   * {@link javafe.parser.Lex#addPunctuation(java.lang.String, int)}.
   *
   * @param argv the command-line arguments.
   * @bon_constraints The command-line arguments may only be
   * "javakeywords", "lookahed", or "parsepragmas".
   */
  //@ requires \nonnullelements(argv);
  public static void main(String[] argv) {
    boolean javakeywords = false;
    boolean lookahead = false;
    boolean parsepragmas = false;
    for(int i = 0; i < argv.length; i++) {
      if (argv[i].equals("javakeywords")) javakeywords = true;
      else if (argv[i].equals("lookahead")) lookahead = true;
      else if (argv[i].equals("parsepragmas")) parsepragmas = true;
      else fatal("Bad argument.");
    }
    
    Lex ll = new Lex(parsepragmas ? new TestLex() : null, false);
    ll.addJavaPunctuation();
    if (javakeywords) ll.addJavaKeywords();
    ll.restart(new FileCorrelatedReader(System.in, "stdin"));
    
    Random r = new Random(System.currentTimeMillis());
    
    int lac = (lookahead ? Math.abs(r.nextInt()) % 9 + 1 : 0);
    int la = ll.lookahead(lac);
    while (ll.ttype != TagConstants.EOF) {
      Assert.notFalse(lac != 0 || ll.ttype == la,	//@ nowarn Pre;
                      "Bad lookahead, is "
                      + PrettyPrint.inst.toString(ll.ttype)
                      + ", expected " + PrettyPrint.inst.toString(la));
      System.out.println(ll.ztoString());
      ll.getNextToken();
      if (lac == 0) {
        lac = (lookahead ? Math.abs(r.nextInt()) % 9 + 1 : 0);
        la = ll.lookahead(lac);
      } else lac--;
      ll.zzz();
    }
    LexicalPragmaVec lpv = ll.getLexicalPragmas();
    for(int i = 0; i < lpv.size(); i++)
      System.out.println(PrettyPrint.inst.toString(TagConstants.LEXICALPRAGMA) 
                         + " ("
                         + lpv.elementAt(i) + ")");
    System.out.println(ll.ztoString());
    
    Identifier.check();
    TagConstants.zzzz();
    ll.zzz();
    
    // Make sure that <code>Character.isXXX(-1)</code> returns false
    // as expected by by the lexer code.
    Assert.notFalse(! Character.isDigit((char)-1));	//@ nowarn Pre;
    Assert.notFalse(! Character.isWhitespace((char)-1));	//@ nowarn Pre;
    Assert.notFalse(! Character.isJavaIdentifierStart((char)-1));  //@ nowarn Pre;
    Assert.notFalse(! Character.isJavaIdentifierPart((char)-1)); //@ nowarn Pre;
    
    System.exit(ErrorSet.errors);
  }
  
  // =================================================================  
  // Below this point are instance variables and methods that
  // implement the javafe.parser.PragmaParser interface.
  
  /*@ spec_public @*/ private /*@ non_null */ Lex l;
  
  private static final int IMPLIES = 48*1024;
  private static final int SUBTYPE = IMPLIES + 1;
  private static final Identifier LEXICAL = Identifier.intern("lexical");
  private static final Identifier MODIFIER = Identifier.intern("modifier");
  private static final Identifier STATEMENT = Identifier.intern("statement");
  private static final Identifier TDE = Identifier.intern("typedeclelem");

  /**
   * Create a test lexer and add a couple of test pragma operators to
   * the lexer.
   */
  public TestLex() {
    l = new Lex(null, true);
    l.addPunctuation("==>", IMPLIES);
    l.addPunctuation("<:", SUBTYPE);
  }

  public boolean checkTag(int tag) {
    return tag == '@';
  }

  //@ also
  //@   modifies l.m_in;
  //@   ensures  l.m_in != null;
  public void restart(CorrelatedReader in, boolean eolComment) {
    l.close();
    try {
      int c = in.read();
      Assert.notFalse(c == '@');	//@ nowarn Pre;
    } catch (IOException e) { Assert.fail("I/O exception"); }   //@ nowarn Pre;
    l.restart(in);
  }

  //@ also
  //@   requires l.m_in != null;
  public boolean getNextPragma(ParserToken destination) {
    if (l.ttype == TagConstants.EOF) return false;
    int loc = l.startingLoc;
    error: {
      if (l.ttype != TagConstants.IDENT) break error;
      Identifier keyword = l.identifierVal;
      l.copyInto(destination); // Get location information
      l.getNextToken();
      if (keyword == LEXICAL) {
        if (l.ttype != TagConstants.INTLIT) break error;
        destination.ttype = TagConstants.LEXICALPRAGMA;
        destination.auxVal = new LexTestLexPrag(l.auxVal.toString());
        l.getNextToken();
      } else if (keyword == MODIFIER) {
        if (l.ttype != TagConstants.IDENT) break error;
        destination.ttype = TagConstants.MODIFIERPRAGMA;
        destination.auxVal = new LexTestModPrag(l.identifierVal.toString());
        l.getNextToken();
      } else if (keyword == STATEMENT) {
        if (l.ttype != TagConstants.IDENT) break error;
        destination.ttype = TagConstants.STMTPRAGMA;
        destination.auxVal = new LexTestStmtPrag(l.identifierVal.toString());
        l.getNextToken();
      } else if (keyword == TDE) {
        if (l.ttype != TagConstants.IDENT) break error;
        destination.ttype = TagConstants.TYPEDECLELEMPRAGMA;
        destination.auxVal = new LexTestTDEPrag(l.identifierVal.toString());
        l.getNextToken();
        
      }
      else Assert.fail("Bad Pragma at line "
                       +Location.toString( l.startingLoc ));
      if (l.ttype == TagConstants.SEMICOLON) l.getNextToken();
      return true;
    }
    ErrorSet.error(loc, "Bad pragma");
    l.close();
    return false;
  }

  public void close() {
    l.close();
  }
  
  public javafe.ast.FieldDecl isPragmaDecl(javafe.parser.ParserToken l) {
    return null;
  }
  
  //@ invariant fakeLoc != Location.NULL;
  public static int fakeLoc = Location.createFakeLoc("<fake location>");
}

class LexTestLexPrag extends LexicalPragma
{
  /*@ non_null @*/ String s;
  
  LexTestLexPrag(/*@ non_null @*/ String s) {
    super();   //@ nowarn Pre; // can't put set before super...
    this.s = s;
  }
  
  public int getTag() { return TagConstants.LEXICALPRAGMA; }
  public String toString() { return s; }
  public void accept(javafe.ast.Visitor v) { }
  public Object accept(javafe.ast.VisitorArgResult v, Object o) { return v; }
  public int getStartLoc() { return TestLex.fakeLoc; }
  public int childCount() { return 0; }
  public Object childAt(int i) { return null; }
}

class LexTestModPrag extends ModifierPragma
{
  /*@ non_null @*/ String s;
  
  LexTestModPrag(/*@ non_null @*/ String s) {
    super();   //@ nowarn Pre; // can't put set before super...
    this.s = s;
  }
  
  public int getTag() { return TagConstants.MODIFIERPRAGMA; }
  public String toString() { return s; }
  public void accept(javafe.ast.Visitor v) { }
  public Object accept(javafe.ast.VisitorArgResult v, Object o) { return v; }
  public int getStartLoc() { return TestLex.fakeLoc; }
  public int childCount() { return 0; }
  public Object childAt(int i) { return null; }
}

class LexTestStmtPrag extends StmtPragma
{
  /*@ non_null @*/ String s;
  
  LexTestStmtPrag(/*@ non_null @*/ String s) {
    super();   //@ nowarn Pre; // can't put set before super...
    this.s = s;
  }
  
  public int getTag() { return TagConstants.STMTPRAGMA; }
  public String toString() { return s; }
  public void accept(javafe.ast.Visitor v) { }
  public Object accept(javafe.ast.VisitorArgResult v, Object o) { return v; }
  public int getStartLoc() { return TestLex.fakeLoc; }
  public int childCount() { return 0; }
  public Object childAt(int i) { return null; }
}

class LexTestTDEPrag extends TypeDeclElemPragma
{
  /*@ non_null @*/ String s;
  
  LexTestTDEPrag(/*@ non_null @*/ String s) {
    super();   //@ nowarn Pre; // can't put set before super...
    this.s = s;
  }
  
  public int getTag() { return TagConstants.TYPEDECLELEMPRAGMA; }
  public String toString() { return s; }
  public void accept(javafe.ast.Visitor v) { }
  public Object accept(javafe.ast.VisitorArgResult v, Object o) { return v; }
  public int getStartLoc() { return TestLex.fakeLoc; }
  public int childCount() { return 0; }
  public Object childAt(int i) { return null; }
}
