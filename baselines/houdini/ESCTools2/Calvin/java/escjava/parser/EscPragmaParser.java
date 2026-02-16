/* Copyright Hewlett-Packard, 2002 */

package escjava.parser;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import javafe.parser.Lex;
import javafe.parser.PragmaParser;
import javafe.parser.Token;
import javafe.parser.Parse;
import javafe.util.Location;
import javafe.util.CorrelatedReader;
import javafe.util.ErrorSet;
import javafe.util.Assert;

import escjava.Main;

import java.io.IOException;


/**

Grammar:

<pre>
Pragma ::= LexicalPragma | ModifierPragma | TypeDeclElemPragma | StmtPragma

LexicalPragma ::= "nowarn" [ Idn [',' Idn]* ] [';']


StmtPragma ::= SimpleStmtPragma [';']
 | ExprStmtPragma SpecExpr [';']
 | 'set' PrimaryExpr '=' SpecExpr [';']

SimpleStmtPragma ::= 'unreachable'

ExprStmtPragma ::= 'assume' | 'assert' | 'loop_invariant' | 'loop_predicate'


TypeDeclElemPragma ::=
   ExprDeclPragma SpecExpr [';']
 | 'ghost' Modifiers Type VariableDeclarator [';']
 | 'still_deferred' Idn [',' Idn]* [';']

ExprDeclPragma ::= 'axiom' | 'invariant' | 'global_invariant' | 'oti' | 'ti'


ModifierPragma ::=
   SimpleModifierPragma 
 | ExprModifierPragma SpecExpr [';']
 | VarExprModifierPragma '(' Type Idn ')' SpecExpr [';']
 | 'monitored_by' SpecExpr [',' SpecExpr]* [';']
 | ModifiesModifierPragma SpecDesignator [',' SpecDesignator]* [';']
 | ActionModifierPragma '(' SpecDesignator [',' SpecDesignator]* ')' '{' SpecExpr '}' [';']

SimpleModifierPragma ::=
   'uninitialized' | 'monitored' | 'non_null' | 'spec_public'
 | 'writable_deferred' | 'helper'

ExprModifierPragma ::= 'readable_if' | 'writable_if' | 'requires'
                     | 'also_requires' (if Main.allowAlsoRequires)
                     | 'ensures' | 'also_ensures'

VarExprModifierPragma ::= 'exsures' | 'also_exsures"

ModifiesModifierPragma ::= 'modifies' | 'also_modifies'

PerformsModifierPragma ::= 'performs' ...

</pre>

The grammar of SpecExpr is:

<pre>
SpecExpr:
   Name
 | \witness
 | \main
 | \tid
 | \result
 | \lockset
 | this
 | Literal
 | SpecExpr '.' Idn
 | SpecExpr '[' SpecExpr ']'
 | UnOp SpecExpr
 | '(' Type ')' SpecExpr
 | SpecExpr BinOp SpecExpr
 | SpecExpr 'instanceof' Type
 | Function '(' [ SpecExpr [ ',' SpecExpr ]* ] ')'
 | '\type' '(' Type ')'
 | SpecExpr '?' SpecExpr ':' SpecExpr
 | '(' {'\forall'|'\exists'} Type Idn [',' Idn]* ';' SpecExpr ')'
 | '(' {'\lblpos'|'\lblneg'} Idn SpecExpr ')'
 | '(' SpecExpr ')'
 | Name '.' this                         [1.1]
 | Name ([])* '.' class                  [1.1]

Function ::= '\fresh' | '\nonnullelements' | '\elemtype' | '\max' | '\old'
           | '\typeof'

UnOp ::= '+' | '-' | '!' | '~'

BinOp ::= '+' | '-' | '*' | '/' | '%' | '<<' | '>>' | '>>>'
        | '<' | '<=' | '==' | '!=' | '>=' | '>'
        | '&' | '|' | '^' | '&&' | '||'
</pre>

Also, the grammar for Type is extended (recursively) with the new base
type 'TYPE'.

*/

public class EscPragmaParser extends Parse implements PragmaParser {
  
    private /*@ non_null @*/ EscPragmaLex scanner;

    /** The value -2 means there is nothing else to process.
      * The value -1 means there is something to process and the next thing to
      * read begins a new pragma (or ends the pragma-containing comment).
      * The other values indicate that the pragma parser is in the middle of
      * parsing some pragma, and is expected to continue this parsing next time
      * it gets control.
      **/

    private int inProcessTag;
    /*@ invariant inProcessTag==-2 || inProcessTag==-1 ||
                  inProcessTag==TagConstants.STILL_DEFERRED ||
                  inProcessTag==TagConstants.MONITORED_BY ||
                  inProcessTag==TagConstants.MODIFIES ||
                  inProcessTag==TagConstants.ALSO_MODIFIES ||
		  inProcessTag==TagConstants.LOOP_PREDICATE; */
    private int inProcessLoc;
    private CorrelatedReader pendingJavadocComment;
    

    /**
     ** Maximum # of levels of nesting of annotation comments allowed.
     ** 0 == no nesting of annotation comments allowed.
     ** 
     ** If you change this, change the error message in
     ** EscPragmaParser(int) below as well.
     **/
    static final int maxAnnotationNestingLevel = 1;


    public EscPragmaParser() {
	this(0);
    }

    public EscPragmaParser(int level) {
	PragmaParser pp;
	if (level<maxAnnotationNestingLevel)
	    pp = new EscPragmaParser(level+1);
	else
	    pp = new ErrorPragmaParser("Annotation comments may be nested "
				       + "at most 1 time");

	scanner = new EscPragmaLex(pp, true);
	scanner.addPunctuation("==>", TagConstants.IMPLIES);
	scanner.addPunctuation("<:", TagConstants.SUBTYPE);
	scanner.addPunctuation("->", TagConstants.MAPSTO);
	addOperator(TagConstants.IMPLIES, 75, false);
	addOperator(TagConstants.SUBTYPE, 140, true);
	scanner.addKeyword("\\tid", TagConstants.TID);
	scanner.addKeyword("\\main", TagConstants.MAIN);
	scanner.addKeyword("\\witness", TagConstants.WITNESS);
	scanner.addKeyword("\\result", TagConstants.RES);
	scanner.addKeyword("\\lockset", TagConstants.LS);
	scanner.addKeyword("\\TYPE", TagConstants.TYPETYPE);
	scanner.addKeyword("\\choice", TagConstants.CHOICE);
	scanner.addKeyword("action", TagConstants.ACTION);
	inProcessTag = -2;
    }

  public boolean checkTag(int tag) {
      return tag == '@' || tag == '*';
  }


  /**
   ** Restart a pragma parser on a new input stream.  If
   ** <code>this</code> is already opened on another
   ** <code>CorrelatedReader</code>, closes the old reader.
   **/
  public void restart(CorrelatedReader in, boolean eolComment) {
    try {
      int c = in.read();
      // System.out.println("restart: c = '"+(char)c+"'");

      switch (c) {
	case '@':
	  /* Normal escjava annotation: */

	  eatWizardComment(in);
	  in = new JmlCorrelatedReader(in,
				       eolComment ?
				       JmlCorrelatedReader.EOL_COMMENT :
				       JmlCorrelatedReader.C_COMMENT);
	  /*
	   * At this point, in has been trimmed/replaced as needed to
	   * represent the annotation part of the comment (if any --
	   * may be empty)
	   */
	  scanner.restart(in);
	  inProcessTag = -1;
	  break;

	case '*':
	  if (eolComment) {
	    // This is not a Javadoc comment, so ignore
	    inProcessTag = -2;
	  } else {
	    // Javadoc comment -- look for <esc> ... </esc> inside

	    if (pendingJavadocComment != null) {
	      pendingJavadocComment.close();
	    }
	    pendingJavadocComment = in;
	    processJavadocComment();
	  }
	  break;

	default:
	  Assert.fail("Bad starting character on comment:"+c+" "+(char)c);
      }
    } catch (IOException e) {
      ErrorSet.fatal(in.getLocation(), e.toString());
    }
  }

  private boolean processJavadocComment() throws IOException {
    if (pendingJavadocComment == null) {
      return false;
    }

    final String startEnclosedPragma = "<esc>";
    final String endEnclosedPragma = "</esc>";

    int startLoc = scanFor(pendingJavadocComment, startEnclosedPragma);
    if (startLoc == Location.NULL) {
      pendingJavadocComment = null;
      inProcessTag = -2;
      return false;
    }

    // Mark the current character, the first character inside the enclosed
    // pragma:
    pendingJavadocComment.mark();

    if (scanFor(pendingJavadocComment, endEnclosedPragma) != Location.NULL) {
      // Restrict "pendingJavadocComment" to just before endEnclosedPragma
      CorrelatedReader nu =
	pendingJavadocComment.createReaderFromMark(endEnclosedPragma.length());
      nu = new JmlCorrelatedReader(nu, JmlCorrelatedReader.JAVADOC_COMMENT);
      scanner.restart(nu);
      inProcessTag = -1;
      return true;
    } else {
      ErrorSet.error(startLoc,
		     "Pragma in javadoc comment is not correctly " +
		     "terminated (missing " + endEnclosedPragma + ")");
      pendingJavadocComment = null;
      inProcessTag = -2;
      return false;
    }
  } //@ nowarn Exception // IndexOutOfBoundsException

    /**
     ** Eat any wizard inserted comment at the start of an
     ** escjava annotation.<p>
     **
     ** May side-effect the mark of <code>in</code>.
     **
     ** Eats "([^)]*)", if present, from <code>in</code>.
     **/

    private void eatWizardComment(/*@ non_null */ CorrelatedReader in) 
                                    throws IOException {
      in.mark();
      int cc = in.read();
      if (cc != '(') {
	in.reset();
	return;
      }
	  
      // Handle (...) comment:
      // Skip up to and including the next close paren:
      do {
	cc = in.read();
	if (cc == -1 || cc == '\n') {
	  // At end-of-comment or end-of-line but still no close paren:
	  ErrorSet.error(in.getLocation(),
			 "Badly formed wizard comment (missing `)')");
	  return;
	}
      } while (cc != ')');
    }



    /** Scans input stream for a string matching match. Only works if
      * the first character is not repeated in the string. <p>
      *
      * If present, the location of the match is returned.  If not
      * present, Location.NULL is returned.<p>
      *
      * Requires that <code>in</code> is not closed.
      **/

    private int scanFor(/*@ non_null */ CorrelatedReader in,
			/*@ non_null */ String match)
	throws IOException
    {

	int start = match.charAt(0);
	int c = in.read();
	
      mainLoop:
	for(;;) {
	    while (c != -1 && c != start )
		c = in.read();

	    if( c == -1 ) return Location.NULL;
	    int startLoc = in.getLocation();
	    Assert.notFalse(startLoc != Location.NULL);

	    // Have a match for the first character in the string
	    
	    for( int i=1; i<match.length(); i++ ) {
		c = in.read();
		if( c != match.charAt(i) )
		    continue mainLoop;
	    }

	    // Have a match
	    return startLoc;
	}
    }
	    

    public void close() {
	scanner.close();
	if (pendingJavadocComment != null) {
	  pendingJavadocComment.close();
	  pendingJavadocComment = null;
	}
    }

  public boolean getNextPragma(Token dst) {
    try {
      if (inProcessTag == -2) {
	return false;
      }

      // See if we need to continue a previous pragma, for example
      // "monitored_by", which can take multiple SpecExprs
      if (inProcessTag != -1) { continuePragma(dst); return true; }

      if (scanner.ttype == TagConstants.EOF) { 
	  LexicalPragma PP = scanner.popLexicalPragma();
	  if (PP!=null) {
	      dst.ttype = TagConstants.LEXICALPRAGMA;
	      dst.auxVal = PP;
	      return true;
	  }

	  if (pendingJavadocComment != null) {
	    scanner.close();
	    if (processJavadocComment()) {
	      return true;
	    }
	  }
	  close();
	  return false;
      }
      //@ assume scanner.m_in != null;  // TBW: is this right??  --KRML

      // Start a new pragma
      int loc = scanner.startingLoc;
      if (scanner.ttype != TagConstants.IDENT)
	ErrorSet.fatal(loc, "Pragma must start with an identifier");
      Identifier kw = scanner.identifierVal;
      scanner.getNextToken();

      int tag = TagConstants.fromIdentifier(kw);
      boolean semiNotOptional = false;

      switch (tag) {
      case TagConstants.NOWARN:
	dst.ttype = TagConstants.LEXICALPRAGMA;
	seqIdentifier.push();
	if (scanner.ttype == TagConstants.IDENT)
	  for (;;) {
	    seqIdentifier.addElement(parseIdentifier(scanner));
	    if (scanner.ttype != TagConstants.COMMA) break;
	    scanner.getNextToken(); // Discard COMMA
	  }
	IdentifierVec checks = IdentifierVec.popFromStackVector(seqIdentifier);
	dst.auxVal = NowarnPragma.make(checks, loc);
	if (scanner.ttype == TagConstants.SEMICOLON) scanner.getNextToken();
	if (scanner.ttype != TagConstants.EOF)
	  ErrorSet.fatal(loc, "Syntax error in nowarn pragma");
	break;

      case TagConstants.STILL_DEFERRED:
      case TagConstants.MONITORED_BY:
      case TagConstants.MODIFIES:
      case TagConstants.ALSO_MODIFIES:
	inProcessTag = tag;
	inProcessLoc = loc;
	continuePragma(dst);
	return true;

      case TagConstants.UNREACHABLE:
	dst.ttype = TagConstants.STMTPRAGMA;
	dst.auxVal = SimpleStmtPragma.make(tag, loc);
	if (scanner.ttype == TagConstants.SEMICOLON) scanner.getNextToken();
	break;
	
      case TagConstants.ASSERT:
      case TagConstants.ASSUME:
      case TagConstants.LOOP_INVARIANT:
	dst.ttype = TagConstants.STMTPRAGMA;
	dst.auxVal = ExprStmtPragma.make(tag, parseExpression(scanner), loc);
	semiNotOptional = true;
	break;

      case TagConstants.LOOP_PREDICATE:
	inProcessTag = tag;
	inProcessLoc = loc;
	continuePragma(dst);
	semiNotOptional = true;
	return true;	  

      case TagConstants.SET:
	dst.ttype = TagConstants.STMTPRAGMA;
	Expr target = parsePrimaryExpression(scanner);
	int locOp = scanner.startingLoc;
	if (target instanceof TidExpr) {
	    ErrorSet.fatal(locOp, "\\tid is a constant.  It is illegal to assign to it.");
	}
	if (target instanceof MainExpr) {
	    ErrorSet.fatal(locOp, "\\main is a constant.  It is illegal to assign to it.");
	}
	expect(scanner, TagConstants.ASSIGN);
	Expr value = parseExpression(scanner);
	dst.auxVal = SetStmtPragma.make(target, locOp, value, loc);
	semiNotOptional = true;
	break;

      case TagConstants.AXIOM:
      case TagConstants.INVARIANT:
      case TagConstants.GLOBAL_INVARIANT:
      case TagConstants.OTI:
      case TagConstants.TI:
	dst.ttype = TagConstants.TYPEDECLELEMPRAGMA;
	dst.auxVal = ExprDeclPragma.make(tag, parseExpression(scanner), loc);
	semiNotOptional = true;
	break;

      case TagConstants.GHOST:
	  {
	      dst.ttype = TagConstants.TYPEDECLELEMPRAGMA;

	      int modifiers = parseModifiers(scanner);
	      ModifierPragmaVec modifierPragmas = this.modifierPragmas;

	      int locType = scanner.startingLoc;
	      Type type = parseType(scanner);
	      
	      // make modifierPragmas non-null, so can retroactively extend
	      if( modifierPragmas == null )
		  modifierPragmas = ModifierPragmaVec.make();
	      
	      int locId     = scanner.startingLoc;
	      Identifier id = parseIdentifier(scanner);
	      Type vartype = parseBracketPairs(scanner, type);
	      
	      if (vartype instanceof MapType) {
		  // only unwritable_by_env_if pragma is allowed on maps
		  for (int i = 0; i < modifierPragmas.size(); i++) {
		      ModifierPragma prag = modifierPragmas.elementAt(i);
		      int pragTag = prag.getTag();
		      if (pragTag != TagConstants.UNWRITABLE_BY_ENV_IF) {
			  ErrorSet.fatal(prag.getStartLoc(), TagConstants.toString(pragTag) +
					 " is not allowed pragma for a map variable");
		      }
		  }
	      }
	      
	      VarInit init = null;
	      int locAssignOp = Location.NULL;
	      if( scanner.ttype == TagConstants.ASSIGN ) {
		  locAssignOp = scanner.startingLoc;
		  scanner.getNextToken();
		  init = parseVariableInitializer(scanner, false);
	      }
	      FieldDecl decl
		  = FieldDecl.make(modifiers, modifierPragmas, 
				   id, vartype, locId, init, locAssignOp );
	      
	      if (scanner.ttype == TagConstants.MODIFIERPRAGMA
		  || scanner.ttype == TagConstants.SEMICOLON ) {
		  // if modifier pragma, retroactively add to modifierPragmas
		  parseMoreModifierPragmas( scanner, modifierPragmas );
	      }
	      
	      if (scanner.ttype == TagConstants.COMMA)
		  fail(scanner.startingLoc,
		       "Only one ghost field may be declared per ghost annotation.");
	      
	      dst.auxVal = GhostDeclPragma.make(decl, loc);
	      semiNotOptional = true;
	      break;
	  }

      case TagConstants.SKOLEM_CONSTANT:
	  {
	      dst.ttype = TagConstants.STMTPRAGMA;
	      
	      int locType = scanner.startingLoc;
	      Type type = parseType(scanner);
	      
	      LocalVarDeclVec v = LocalVarDeclVec.make();
	      int nextTtype;
	      
	    loop:
	      for(;;) {
		  int locId     = scanner.startingLoc;
		  Identifier id = parseIdentifier(scanner);
		  Type vartype = parseBracketPairs(scanner, type);
		  
		  LocalVarDecl decl
		      = LocalVarDecl.make(Modifiers.NONE, null, id, 
					  vartype, locId, null, Location.NULL);
		  v.addElement(decl);
		  
		  switch( scanner.ttype ) {
		    case TagConstants.COMMA:
		      scanner.getNextToken();
		      continue loop;

		    default:
		      fail( scanner.startingLoc, 
			    "Expected comma or semicolon in skolem_constant decl");

		    case TagConstants.SEMICOLON:
		      break loop;

		  }
	      }

	      dst.auxVal = SkolemConstantPragma.make(v, locType, scanner.startingLoc);
	      semiNotOptional = true;
	      break;
	  }

      case TagConstants.UNINITIALIZED:
      case TagConstants.MONITORED:
      case TagConstants.NON_NULL:
      case TagConstants.SPEC_PUBLIC:
      case TagConstants.WRITABLE_DEFERRED:
      case TagConstants.HELPER:
      case TagConstants.THREAD_LOCAL:
      case TagConstants.THREAD_LOCAL_RESULT:
	dst.ttype = TagConstants.MODIFIERPRAGMA;
	dst.auxVal = SimpleModifierPragma.make(tag, loc);
	break;

      case TagConstants.PERFORMS: {
	dst.ttype = TagConstants.MODIFIERPRAGMA;
	PerformsStmt ps = parsePerformsChoice(scanner, loc);
	LabelRelation lr = new LabelRelation(ps);
	//	System.out.println("------\n" + lr);
	PerformsModifierPragma pmp = PerformsModifierPragma.make(loc, ps, lr);
	dst.auxVal = pmp;
	semiNotOptional = true; 

	break;
      }

      case TagConstants.UNWRITABLE_BY_ENV_IF: {
	IdentifierVec ids = parseParameterList(scanner);
	Expr expr = parseExpression(scanner);
	dst.ttype = TagConstants.MODIFIERPRAGMA;	
	dst.auxVal = ParamExprModifierPragma.make(tag, ids, expr, loc);
	semiNotOptional = true;
	break;
      }

      case TagConstants.DEFINED_IF:
      case TagConstants.WRITABLE_IF:
      case TagConstants.REQUIRES:
      case TagConstants.ALSO_REQUIRES:
      case TagConstants.ENSURES:
      case TagConstants.ALSO_ENSURES:
	if (tag == TagConstants.ALSO_REQUIRES && !Main.allowAlsoRequires) {
	  ErrorSet.fatal(loc, TagConstants.toString(tag) +
			 " is not allowed pragma");
	}
	dst.ttype = TagConstants.MODIFIERPRAGMA;
	dst.auxVal
	  = ExprModifierPragma.make(tag, parseExpression(scanner), loc);
	semiNotOptional = true;
	break;

      case TagConstants.EXSURES:
      case TagConstants.ALSO_EXSURES: {
	dst.ttype = TagConstants.MODIFIERPRAGMA;
	expect(scanner, TagConstants.LPAREN);
	FormalParaDecl arg = parseExsuresFormalParaDecl(scanner);
	expect(scanner, TagConstants.RPAREN);
	Expr expr = parseExpression(scanner);
	dst.auxVal
	  = VarExprModifierPragma.make(tag, arg, expr, loc);
	semiNotOptional = true;
	break;
      }
	
      case TagConstants.ELEMS_UNWRITABLE_BY_ENV_IF: {
	IdentifierVec ids = parseParameterList(scanner);
	Expr expr = parseExpression(scanner);
	dst.ttype = TagConstants.MODIFIERPRAGMA;	
	dst.auxVal = ParamExprModifierPragma.make(tag, ids, expr, loc);
	semiNotOptional = true;
	break;
      }

      default:
	ErrorSet.fatal(loc, "Unrecognized pragma");
      }

      if (semiNotOptional)
	if (scanner.ttype == TagConstants.SEMICOLON) scanner.getNextToken();
	else if (scanner.ttype != TagConstants.EOF)
	  ErrorSet.fatal(scanner.startingLoc, 
			 ("Semicolon required when a " + kw.toString()
			  + " pragma is followed by another pragma."));

      return true;
    } catch (IOException e) { return false; }
  }


    private ExprVec parsePerformsList(/*@ non_null */ EscPragmaLex l) {
	ExprVec result = ExprVec.make();

	if (l.lookahead(0) != TagConstants.RPAREN) {	    
	    Expr first = parseExpression(l);
	    result.addElement(first);
	    
	    while(l.lookahead(0) == TagConstants.COMMA) {
		expect(l, TagConstants.COMMA);
		result.addElement(parseExpression(l));
	    }
	}

	return result;
    }


    private IdentifierVec parseParameterList(/*@ non_null */ EscPragmaLex l) {
	IdentifierVec result = IdentifierVec.make();

	while(l.lookahead(0) == TagConstants.LSQBRACKET) {
	    expect(l, TagConstants.LSQBRACKET);
	    Identifier idn = parseIdentifier(l);
	    if (result.contains(idn)) {
		ErrorSet.fatal(l.startingLoc, 
			       "Parameters must be distinct");
	    }
	    result.addElement(idn);
	    expect(l, TagConstants.RSQBRACKET);
	}

	return result;
    }

    /*@ requires inProcessTag==TagConstants.STILL_DEFERRED ||
                 inProcessTag==TagConstants.MONITORED_BY ||
                 inProcessTag==TagConstants.MODIFIES ||
                 inProcessTag==TagConstants.ALSO_MODIFIES ||
		 inProcessTag==TagConstants.LOOP_PREDICATE */
    //@ requires dst!=null
    //@ requires scanner.startingLoc != Location.NULL;
    //@ requires scanner.m_in != null;
    private void continuePragma(Token dst) throws IOException {
    if (inProcessTag == TagConstants.STILL_DEFERRED) {
      int locId = scanner.startingLoc;
      Identifier idn = parseIdentifier(scanner);
      dst.ttype = TagConstants.TYPEDECLELEMPRAGMA;
      dst.auxVal = StillDeferredDeclPragma.make(idn, inProcessLoc, locId);
    } else if (inProcessTag == TagConstants.MONITORED_BY
	       || inProcessTag == TagConstants.MODIFIES
	       || inProcessTag == TagConstants.ALSO_MODIFIES) {
      dst.startingLoc = inProcessLoc;
      Expr e = parseExpression(scanner);
      dst.auxVal = ExprModifierPragma.make(inProcessTag, e, inProcessLoc);
      dst.ttype = TagConstants.MODIFIERPRAGMA;
    } else if (inProcessTag == TagConstants.LOOP_PREDICATE) {
      dst.startingLoc = inProcessLoc;
      Expr e = parseExpression(scanner);
      dst.auxVal = ExprStmtPragma.make(inProcessTag, e, inProcessLoc);
      dst.ttype = TagConstants.STMTPRAGMA;
    } else Assert.precondition(false);

    switch (scanner.ttype) {
    case TagConstants.SEMICOLON:
      scanner.getNextToken();
    case TagConstants.EOF:
      inProcessTag = -1;
      break;

    case TagConstants.COMMA:
      scanner.getNextToken();
      break;

    default:
      ErrorSet.fatal(scanner.startingLoc,
		     "Unexpected token '" +TagConstants.toString(scanner.ttype)
		     + "', expected ',', ';' or end-of-file");
    }
  }

  protected Expr parsePrimaryExpression(Lex l) {

    /* Lookahead for:
     *  \tid
     *  \main
     *  \result
     *  \lockset
     *  Function '('
     * '\type' '('
     * '(' {'\forall'|'\exists'} Type
     * '(' {'\lblpos'|'\lblneg'} Idn
     */

    Expr primary = null /*@ uninitialized */;

    // First parse PrimaryCore into variable primary
    switch(l.ttype) {

    case TagConstants.TID:
      {
	  primary = TidExpr.make( l.startingLoc );
	  l.getNextToken();
	  break;
      }

    case TagConstants.MAIN:
      {
	  primary = MainExpr.make( l.startingLoc );
	  l.getNextToken();
	  break;
      }

    case TagConstants.WITNESS:
      {
	  primary = WitnessExpr.make( l.startingLoc );
	  l.getNextToken();
	  break;
      }
	
    case TagConstants.RES: 
      { 
	primary = ResExpr.make( l.startingLoc );
	l.getNextToken();
	break;
      }

    case TagConstants.LS: 
      { 
	primary = LockSetExpr.make( l.startingLoc );
	l.getNextToken();
	break;
      }

    case TagConstants.IDENT: 
      { 
	  // First comes a Name...
	  int loc = l.startingLoc;
	  Name n = parseName(l);

	  // May be followed by ( ArgumentListopt ) :
	  if (l.ttype == TagConstants.LPAREN) {
	      int locOpenParen = l.startingLoc;
	      l.getNextToken();

	      Identifier kw = n.identifierAt(0);
	      int tag = TagConstants.fromIdentifier(kw);

	      if (n.size()!=1)
		  fail(loc, "Annotations may not contain method calls");
	      switch (tag) {
	      case TagConstants.TYPE:
		{
		  Type subexpr = parseType(l);
		  primary = TypeExpr.make(loc, l.startingLoc, subexpr);
		  expect(l, TagConstants.RPAREN);
		  break;
		}
	    
	      case TagConstants.DTTFSA:
		{
		  Type t = parseType(l);
		  TypeExpr te = TypeExpr.make(loc, l.startingLoc, t);
		  expect(l, TagConstants.COMMA);
		  ExprVec args = parseExpressionList(l, TagConstants.RPAREN);
		  args.insertElementAt(te, 0);
		  primary = NaryExpr.make(loc, l.startingLoc, tag, args);
		  break;
		}
	    
	      case TagConstants.FRESH: 
	      case TagConstants.ELEMSNONNULL:
	      case TagConstants.ELEMTYPE:
	      case TagConstants.MAX: 
	      case TagConstants.PRE:
	      case TagConstants.TYPEOF:
		{
		  ExprVec args = parseExpressionList(l, TagConstants.RPAREN);
		  primary = NaryExpr.make(loc, l.startingLoc, tag, args);
		  break;
		}
	    
	      default:
		  ErrorSet.fatal(loc, "Unknown ESC function symbol " + kw);
	      }
	      break;
	  }

	  // Look for 'TypeName . this'
	  if (l.lookahead(0) == TagConstants.FIELD &&
	      l.lookahead(1) == TagConstants.THIS) {
	      expect( l, TagConstants.FIELD );
	      int locThis = l.startingLoc;
	      expect( l, TagConstants.THIS );
	      primary = ThisExpr.make(TypeName.make(n), locThis);
	      break;
	  }

	  // Or ([])* . class:
	  // (need to look ahead fully because of "<type>[] x;" declarations)
	  int i = 0;
	  while ( l.lookahead(i) == TagConstants.LSQBRACKET &&
		  l.lookahead(i+1) == TagConstants.RSQBRACKET )
	      i += 2;
	  if (l.lookahead(i) == TagConstants.FIELD &&
	      l.lookahead(i+1) == TagConstants.CLASS) {
	      Type t = TypeName.make(n);
	      t = parseBracketPairs(l, t);
	      primary = parseClassLiteralSuffix(l, t);
	      break;
	  }

	  // Ok, it must have just been a naked Name:
	  primary = AmbiguousVariableAccess.make(n);
	  break;
      }
      
    case TagConstants.LPAREN: 
      { 
	// LPAREN Expression RPAREN
	int locOpenParen = l.startingLoc;
	l.getNextToken();
	boolean regularParenExpr = true;
	
	// First see if we have a (LBLxxx ...) or (quantifier ...)
	if (l.ttype == TagConstants.IDENT) {
	  Identifier kw = l.identifierVal;
	  int tag = TagConstants.fromIdentifier(kw);
	  if ((tag == TagConstants.LBLPOS || tag == TagConstants.LBLNEG)
	      && l.lookahead(1) == TagConstants.IDENT) {
	    regularParenExpr = false;
	    boolean pos = (tag == TagConstants.LBLPOS);
	    l.getNextToken(); // Discard LBLxxx
	    Identifier label = l.identifierVal;
	    l.getNextToken();
	    Expr e = parseExpression(l);
	    primary = LabelExpr.make(locOpenParen, l.startingLoc,
				     pos, label, e);
	    expect(l, TagConstants.RPAREN);
	  } else if (tag == TagConstants.FORALL
		     || tag == TagConstants.EXISTS) 
	    {
	      int lookahead = l.lookahead(1);
	      if (lookahead==TagConstants.IDENT ||
			isPrimitiveKeywordTag(lookahead)) {
		regularParenExpr = false;
		l.getNextToken(); // Discard quantifier
		Type type = parseType(l);
		primary = parseQuantifierRemainder(l, tag, type, locOpenParen);
	      }
	    }
	}
	if (regularParenExpr) {
	  Expr e = parseExpression(l);
	  int locCloseParen = l.startingLoc;
	  expect(l, TagConstants.RPAREN);
	  primary = ParenExpr.make(e, locOpenParen, locCloseParen);
	}
	break;
      }

    default:
      primary = super.parsePrimaryExpression( l );
			     
    }

    // Ok, parsed a PrimaryCore expression into primary. 
    // Now handle PrimarySuffix*

    return parsePrimarySuffix( l, primary );
  }

  protected Expr parsePrimarySuffix(Lex l, Expr primary) {

    for(;;) {
      primary = super.parsePrimarySuffix( l, primary );

      if( l.ttype == TagConstants.LSQBRACKET 
	 && l.lookahead(1) == TagConstants.STAR 
	 &&  l.lookahead(2) == TagConstants.RSQBRACKET) 
      {
	int locOpen = l.startingLoc;
	l.getNextToken();
	l.getNextToken();
	int locClose = l.startingLoc;
	l.getNextToken();
	primary = WildRefExpr.make( primary, locOpen, locClose );
	// and go around again
      } else {
	// no PrimarySuffix left, all done
	return primary;
      }
    }
  }

  //@ requires l.m_in != null;
  //@ requires type.syntax;
  //@ ensures \result != null;
  private QuantifiedExpr parseQuantifierRemainder(/*@ non_null */ Lex l,
						  int tag,
						  /*@ non_null */ Type type,
						  int loc) {
    int idLoc = l.startingLoc;
    Identifier idn = parseIdentifier(l);
    LocalVarDecl v = LocalVarDecl.make(0, null, idn, type, idLoc,
				       null, Location.NULL);

    GenericVarDeclVec vs = GenericVarDeclVec.make();
    vs.addElement(v);
    
    int endLoc = 0 /*@ uninitialized */;
    Expr rest = null /*@ uninitialized */;
    if (l.ttype == TagConstants.COMMA) {
      l.getNextToken();
      QuantifiedExpr r = parseQuantifierRemainder(l, tag, type, Location.NULL);
      rest = r;
      endLoc = r.eloc;
    } else if (l.ttype == TagConstants.SEMICOLON) {
      l.getNextToken();
      rest = parseExpression(l);
      endLoc = l.startingLoc;
      expect(l, TagConstants.RPAREN);
    } else
      ErrorSet.fatal(l.startingLoc, "Syntax error in quantified expression.");
    return QuantifiedExpr.make(loc, endLoc, tag, vs, rest, null);
  }


  /**
   ** Is a tag a PrimitiveType keyword?  Overriden to add type TYPE.
   **/
  public boolean isPrimitiveKeywordTag(int tag) {
    switch( tag ) {
    case TagConstants.TYPETYPE:
	return true;

    default:
	return super.isPrimitiveKeywordTag(tag);
    }
  }

  /** Parses a PrimitiveType.  Overriden to add type TYPE.
      Returns null on failure.
      <PRE>   
      PrimitiveType: one of
         boolean byte short int long char float double void TYPE
      </PRE>
   */

  public PrimitiveType parsePrimitiveType(Lex l) {
    int tag;

    switch( l.ttype ) {
    case TagConstants.TYPETYPE: tag = TagConstants.TYPECODE; break;

    default: return super.parsePrimitiveType(l);
    }
    // get here => tag is defined
    
    int loc = l.startingLoc;
    l.getNextToken();
    return PrimitiveType.make( tag, loc );
  }

    public boolean isStartOfUnaryExpressionNotPlusMinus(int tag) {
	// All previous cases apply...
	if (super.isStartOfUnaryExpressionNotPlusMinus(tag))
	    return true;

	// New Identifier-like keywords:
	if (tag==TagConstants.RES || tag==TagConstants.LS)
	    return true;

	return false;
    }

    //@ requires l.m_in != null;
    public FormalParaDecl parseExsuresFormalParaDecl(/*@ non_null */ EscPragmaLex l) {
      int modifiers = parseModifiers(l);
      ModifierPragmaVec modifierPragmas = this.modifierPragmas;
      Type paratype = parseExsuresType(l);
      Identifier idn;
      int locId = l.startingLoc;

      if (l.ttype == TagConstants.IDENT) {
	idn = l.identifierVal;
	l.getNextToken();
      } else {
	idn = TagConstants.ExsuresIdnName;
      }
      
      // allow more modifier pragmas
      modifierPragmas = parseMoreModifierPragmas( l, modifierPragmas );

      return FormalParaDecl.make(modifiers, modifierPragmas, 
				 idn, paratype, locId);
  }

    //@ requires l.m_in != null;
    //@ ensures \result != null;
    //@ ensures \result.syntax;
    public Type parseExsuresType(/*@ non_null */ EscPragmaLex l) {
	Type type = parseExsuresPrimitiveTypeOrTypeName(l);
	return parseBracketPairs(l, type);
    }


  //@ requires l!=null && l.m_in!=null
  //@ ensures \result!=null
  //@ ensures \result.syntax
    public Type parseExsuresPrimitiveTypeOrTypeName(EscPragmaLex l) {
	Type type = parseExsuresPrimitiveType(l);
	if (type != null)
	    return type;
	else
	    return parseExsuresTypeName(l);
    }

    //@ requires l!=null && l.m_in!=null
    //@ ensures \result!=null ==> \result.syntax
    public PrimitiveType parseExsuresPrimitiveType (EscPragmaLex l) {

	int tag;
	switch( l.ttype ) {
	case TagConstants.TYPETYPE:tag = TagConstants.TYPECODE; break;
	case TagConstants.BOOLEAN: tag = TagConstants.BOOLEANTYPE; break;
	case TagConstants.BYTE:    tag = TagConstants.BYTETYPE;    break;
	case TagConstants.SHORT:   tag = TagConstants.SHORTTYPE;   break;
	case TagConstants.INT:     tag = TagConstants.INTTYPE;     break;
	case TagConstants.LONG:    tag = TagConstants.LONGTYPE;    break;
	case TagConstants.CHAR:    tag = TagConstants.CHARTYPE;    break;
	case TagConstants.FLOAT:   tag = TagConstants.FLOATTYPE;   break;
	case TagConstants.DOUBLE:  tag = TagConstants.DOUBLETYPE;  break;
	case TagConstants.VOID:    tag = TagConstants.VOIDTYPE;    break;
	default: return null;	// Fail!
	}
	// get here => tag is defined
    
	int loc = l.startingLoc;
	l.getNextToken();
	return PrimitiveType.make( tag, loc );
    }

    //@ requires l!=null && l.m_in!=null
    //@ ensures \result!=null
    //@ ensures \result.syntax
    public TypeName parseExsuresTypeName(EscPragmaLex l) {
	return parseTypeName(l);	
    }

    public Type parseType(Lex l) {
	int loc = l.startingLoc;
	Type t1 = super.parseType(l);

	if (l.ttype == TagConstants.MAPSTO) {
	    l.getNextToken();
	    return MapType.make(t1, parseType(l), loc);
	} else {	   	
	    return t1;
	}
    }




   PerformsStmt parsePerformsChoice(EscPragmaLex scanner, int loc) {
	PerformsStmt p = null;
	
	p = parsePerformsSequence(scanner, loc);

	if (scanner.ttype == TagConstants.LSQBRACKET)  {
	    PerformsStmt q;
	    expect(scanner, TagConstants.LSQBRACKET);
	    expect(scanner, TagConstants.RSQBRACKET);
	    q = parsePerformsChoice(scanner, scanner.startingLoc);
	    Assert.notFalse(p != null);
	    p = PerformsChoice.make(loc, p, q);
	}
	
	return p;
   }


   PerformsStmt parsePerformsSequence(EscPragmaLex scanner, int loc) {
	PerformsStmt p = null;
	PerformsStmtVec pv = PerformsStmtVec.make();

	if (scanner.ttype == TagConstants.LPAREN) {
	    expect(scanner, TagConstants.LPAREN);
	    p = parsePerformsChoice(scanner, loc);
	    expect(scanner, TagConstants.RPAREN);
	} else {
	    expect(scanner, TagConstants.ACTION);
	    if (scanner.ttype != TagConstants.STRINGLIT) {
		ErrorSet.fatal(scanner.startingLoc, "expected string literal for action label.");
	    }
	    String label = (String)scanner.auxVal;
	    scanner.getNextToken();
	    expect(scanner, TagConstants.LPAREN);
	    ExprVec exprs = parsePerformsList(scanner);
	    expect(scanner, TagConstants.RPAREN);
	    expect(scanner, TagConstants.LBRACE);
	    Expr pred = parseExpression(scanner);
	    expect(scanner, TagConstants.RBRACE);
	    p = PerformsAction.make(loc, exprs, pred, label);
	    pv.addElement(p);

	    while (true) {
		int tok = scanner.lookahead(1);
		if (scanner.ttype != TagConstants.SEMICOLON ||
		    (tok != TagConstants.ACTION &&
		     tok != TagConstants.LPAREN)) {
		    break;
		}
		expect(scanner, TagConstants.SEMICOLON);
		p = parsePerformsSequence(scanner, scanner.startingLoc);
		pv.addElement(p);
	    }
	    p = PerformsSequence.make(loc, pv);
	}
	return p;
   }
    

}
