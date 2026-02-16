/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

import javafe.ast.*;
import javafe.util.StackVector;
import javafe.util.ErrorSet;

import javafe.util.CorrelatedReader;	// For test harness only
import javafe.util.Location;

import java.util.List;
import java.util.LinkedList;
import java.util.Iterator;

/**
 * Parses java source.
 * 
 * <p> Uses the static <code>make*()</code> methods of the classes of
 * the <code>javafe.ast</code> package to create AST nodes.
 * 
 * <p> The main entry point is the method
 * {@link #parseStream(CorrelatedReader, boolean)}.
 * 
 * <P>Each parsing method for a particular syntactic unit is
 * documented with appropriate grammar production rules for that
 * syntactic unit.  These grammar rules follow the conventions
 * described in "The Java Language Specification", with the addition
 * of the symbols '(', ')', '[', ']', '*', '+', and '|', which have
 * their usual meaning. When necessary, we denote the corresponding
 * concrete tokens using LPAREN, RPAREN, LSQBRACKET, RSQBRACKET, STAR,
 * PLUS and BITOR.
 * 
 * @see javafe.ast.ASTNode
 * @see javafe.parser.ParseStmt
 */

public class Parse extends ParseStmt
{
    public Parse() {
	//@ set seqTypeName.elementType = \type(TypeName);
	//@ set seqTypeName.owner = this;

	//@ set seqFormalParaDecl.elementType = \type(FormalParaDecl);
	//@ set seqFormalParaDecl.owner = this;

	//@ set seqImportDecl.elementType = \type(ImportDecl);
	//@ set seqImportDecl.owner = this;

	//@ set seqTypeDecl.elementType = \type(TypeDecl);
	//@ set seqTypeDecl.owner = this;
    }

    /**
     * Internal working storage for many Parse functions.
     */
    //@ invariant seqTypeName.elementType == \type(TypeName);
    //@ invariant seqTypeName.owner == this;
    protected final /*@ non_null @*/ StackVector seqTypeName
	= new StackVector();

    /**
     * Internal working storage for many Parse functions.
     */
    //@ invariant seqFormalParaDecl.elementType == \type(FormalParaDecl);
    //@ invariant seqFormalParaDecl.owner == this;
    protected final /*@ non_null @*/ StackVector seqFormalParaDecl
	= new StackVector();

    /**
     * Internal working storage for many Parse functions.
     */
    //@ invariant seqImportDecl.elementType == \type(ImportDecl);
    //@ invariant seqImportDecl.owner == this;
    protected final /*@ non_null @*/ StackVector seqImportDecl
	= new StackVector();

    /**
     * Internal working storage for many Parse functions.
     */
    //@ invariant seqTypeDecl.elementType == \type(TypeDecl);
    //@ invariant seqTypeDecl.owner == this;
    protected final /*@ non_null @*/ StackVector seqTypeDecl
	= new StackVector();


  /** Parse a <TT>CompilationUnit</TT> from an input stream.

    <P> Requires: prefix of <TT>in</TT> contains text from which the
    <TT>CompilationUnit</TT> non-terminal can be derrived.

    <P> Ensures: parses a <TT>CompilationUnit</TT> from the input
    stream, generating a syntax tree for it.  All errors are treated
    as fatal errors and are reporting through <code>ErrorSet</code>.

    @see javafe.util.ErrorSet
    */

  public CompilationUnit parseStream(/*@non_null*/CorrelatedReader in, boolean specOnly) {
    Lex _parseStreamLexer = parseStreamLexer;
	if (_parseStreamLexer == null) _parseStreamLexer = parseStreamLexer = new Lex(null, true);
    _parseStreamLexer.restart(in);
    return parseCompilationUnit(_parseStreamLexer, specOnly);
  }


  private Lex parseStreamLexer;

  /** Parse a <TT>CompilationUnit</TT>.
      <PRE>
      CompilationUnit:
         [Package name ;] ImportDeclaration* TypeDeclaration*
      </PRE>
      To handle pragmas, call this method directly 
      with an appropriate <TT>Lex</TT> object.
    */
  // specOnly means parse without keeping the bodies of methods/constructors/..

  //@ requires l.m_in != null;
  public /*@non_null*/CompilationUnit parseCompilationUnit(/*@non_null*/Lex l, boolean specOnly) {
    Name pkgName = null;
    int loc = l.startingLoc;


    /* Optional PackageDeclaration: package name ; */
    if( l.ttype == ParserTagConstants.PACKAGE ) {
      l.getNextToken();
      pkgName = parseName(l);
      expect( l, ParserTagConstants.SEMICOLON );
    } 

    /* Import Declarations */
    seqImportDecl.push();
    while( l.ttype == ParserTagConstants.IMPORT ) { 
      seqImportDecl.addElement( parseImportDeclaration( l ) );
    }
    ImportDeclVec imports = ImportDeclVec.popFromStackVector(seqImportDecl);

    /* Type Declarations */
    seqTypeDeclElem.push();
    seqTypeDecl.push();
    while( l.ttype != ParserTagConstants.EOF ) {
      if( l.ttype == ParserTagConstants.SEMICOLON )
        l.getNextToken();
      else {
	int locstart = l.startingLoc;
	int modifiers = parseModifiers(l);
	ModifierPragmaVec modifierPragmas = this.modifierPragmas;

	  if (l.ttype == ParserTagConstants.TYPEDECLELEMPRAGMA) {
	    TypeDeclElemPragma pragma = (TypeDeclElemPragma)l.auxVal;
	    pragma.decorate(modifierPragmas);
		// FIXME - what about modifiers ?
	    seqTypeDeclElem.addElement( pragma );
	    l.getNextToken();
	  } else {
	    TypeDecl td = parseTypeDeclaration(l, specOnly,modifiers,modifierPragmas,locstart);
	    if (td != null) seqTypeDecl.addElement( td );
	  }
      }
    }
    TypeDeclVec elems = TypeDeclVec.popFromStackVector( seqTypeDecl );
    TypeDeclElemVec extras = TypeDeclElemVec.popFromStackVector( seqTypeDeclElem);

    LexicalPragmaVec lexicalPragmas = l.getLexicalPragmas();

    return CompilationUnit.make(pkgName, lexicalPragmas, 
				imports, elems, loc, extras );
  }

  /** Parse an <TT>ImportDeclaration</TT>.
    <PRE>
    ImportDeclaration:
      import Name ; 
      import Name . STAR ;
    </PRE>
   */
        
  //@ requires l.m_in != null;
  protected /*@non_null*/ImportDecl parseImportDeclaration(/*@non_null*/Lex l) {
    int loc = l.startingLoc;
    l.getNextToken();                // swallow import keyword
    Name name = parseName(l);
    if( l.ttype == ParserTagConstants.SEMICOLON ) {
      l.getNextToken();              // swallow semicolon
      TypeName typename = TypeName.make( name );
      return SingleTypeImportDecl.make( loc, typename );
    } else {
      expect( l, ParserTagConstants.FIELD );
      expect( l, ParserTagConstants.STAR );
      expect( l, ParserTagConstants.SEMICOLON );
      return OnDemandImportDecl.make( loc, name );
    }
  }


  /**********************************************************************
    Parse a <TT>TypeDeclaration</TT> (ie a class or interface declaration).
    <PRE>
      TypeDeclaration:
        ClassDeclaration
        InterfaceDeclaration  
        ;

      ClassDeclaration:
        TypeDeclElemPragma* Modifiers class Identifier [extends TypeName] [implements TypeNameList]
          { TypeDeclElem* }
	TypeDeclElemPragma* Modifiers interface Identifier [extends TypeNameList]
          { TypeDeclElem* }
     </PRE>
   */

  //@ requires l.m_in != null;
  protected /*@non_null*/TypeDecl parseTypeDeclaration(/*@non_null*/Lex l, boolean specOnly) {
    int locstart = l.startingLoc;
    int modifiers = parseModifiers(l);
    ModifierPragmaVec modifierPragmas = this.modifierPragmas;
    return parseTypeDeclaration(l,specOnly,modifiers,modifierPragmas,
					locstart);
  }
  protected /*@non_null*/TypeDecl parseTypeDeclaration(/*@non_null*/Lex l, boolean specOnly,
			int modifiers, ModifierPragmaVec modifierPragmas,
			int loc) {
    TypeDecl result = /*@(non_null)*/ parseTypeDeclTail(l, specOnly, loc, modifiers, modifierPragmas);
    return result;
  }

  /**********************************************************************
    Parse a <TT>TypeDeclTail</TT> (ie a class or interface declaration
    starting at the keyword 'class' or 'interface').
    <PRE>
      TypeDeclTail:
        class Identifier [TypeModifierPragma]* [extends TypeName] [implements TypeNameList]
          { TypeDeclElem* }
        interface Identifier [TypeModifierPragma]* [extends TypeNameList] { TypeDeclElem* }
     </PRE> */

  protected
  TypeDecl parseTypeDeclTail(Lex l, boolean specOnly, int loc, 
			     int modifiers, ModifierPragmaVec modifierPragmas){
    int keyword = l.ttype;
   
    if( keyword != ParserTagConstants.CLASS && keyword != ParserTagConstants.INTERFACE ) {
      if (keyword == ParserTagConstants.EOF) return null;
      fail(l.startingLoc, "expected 'class' or 'interface' instead of "
		+ ParserTagConstants.toString(keyword));
    }

    l.getNextToken();              // swallow keyword

    int locId = l.startingLoc;
    Identifier id = parseIdentifier(l);
    TypeModifierPragmaVec tmodifiers = parseTypeModifierPragmas(l);
    
    /* Parse superclass, if any */
    TypeName superClass = null;
    if( keyword == ParserTagConstants.CLASS && l.ttype == ParserTagConstants.EXTENDS ) {
      l.getNextToken();            // swallow "extends" keyword
      superClass = parseTypeName(l);
    }

    /* Parse super interfaces, if any */
    TypeNameVec superInterfaces =
      parseTypeNames( l, (keyword == ParserTagConstants.CLASS ? 
			  ParserTagConstants.IMPLEMENTS : ParserTagConstants.EXTENDS ) );

    /* Now parse class body */
    int locOpenBrace = l.startingLoc;
    expect( l, ParserTagConstants.LBRACE );
    
    /* Build up Vec of TypeDeclElems in class or interface */
    seqTypeDeclElem.push();
    while( l.ttype != ParserTagConstants.RBRACE ) {
      parseTypeDeclElemIntoSeqTDE( l, keyword, id, specOnly );
    }
    TypeDeclElemVec elems =
	TypeDeclElemVec.popFromStackVector( seqTypeDeclElem );

    int locCloseBrace = l.startingLoc;
    expect( l, ParserTagConstants.RBRACE );

    TypeDecl result;
    if (keyword == ParserTagConstants.CLASS) {
      addDefaultConstructor(elems, locOpenBrace, specOnly);
      result = ClassDecl.make(modifiers, modifierPragmas, id, 
			      superInterfaces, tmodifiers, elems,
			      loc, locId, locOpenBrace, locCloseBrace,
			      superClass);
    } else {
      result = InterfaceDecl.make(modifiers, modifierPragmas, id,
				  superInterfaces, tmodifiers, elems,
				  loc, locId, 
				  locOpenBrace, locCloseBrace );
    }
    result.specOnly = specOnly;
    return result;
  }
    

  /** checks for           X TypeModifierPragma* (
      in the input stream. 
      Use this to match the beginning of a constructor or method declration
      versus the beginning of a field declaration.
  */
    //@ requires l != null && l.m_in != null;
    private boolean atStartOfConstructorOrMethod(Lex l) {
	int i = 1;
	while ((l.lookahead(i) == ParserTagConstants.TYPEMODIFIERPRAGMA)) {
	    i++;
	}
	return l.lookahead(i) == ParserTagConstants.LPAREN;
    }




  /** If no constructors are found in "elems", adds a default one to it.
    If a default constructor is created, the "loc" and "locId" fields of
    the default constructor will be set to "loc". */
  void addDefaultConstructor(TypeDeclElemVec elems, int loc, boolean specOnly) {
      // Return if a constructor is already present:
      for (int i=0; i<elems.size(); i++) {
	  if (elems.elementAt(i) instanceof ConstructorDecl)
	      return;
      }

      /*
       * No constructor found, add one:
       *  name() { super(); }
       *
       * Don't put super constructor invocation in --  it is added by
       * the type checker.
       */
      BlockStmt blk = specOnly ? null :BlockStmt.make(StmtVec.make(), loc, loc);
      TypeNameVec raises = TypeNameVec.make();
      FormalParaDeclVec formals = FormalParaDeclVec.make();
      ConstructorDecl cd
	  = ConstructorDecl.make(Modifiers.ACC_PUBLIC, null, null, 
				 formals, raises, blk, loc, loc, loc,
				 Location.NULL );
      cd.implicit = true;
      cd.specOnly = specOnly;
      elems.addElement(cd);
  }

  /** Parse a keyword, 
      followed by a comma-separated list of <TT>TypeName</TT>s.

      Used to parse throws clauses, and super-interface clauses.
      <PRE>
      TypeNames:
         keyword TypeNameList
         empty

      TypeNameList:
        TypeName ( , TypeName )*
      </PRE>
   */

  //@ requires l != null && l.m_in != null;
  //@ ensures \result != null;
  protected TypeNameVec parseTypeNames(Lex l, int keyword)
  {
    if( l.ttype != keyword ) 
      return TypeNameVec.make();

    /* Have type names */
    seqTypeName.push();
    do {
      // Skip keyword or ',' .
      l.getNextToken();
      seqTypeName.addElement( parseTypeName(l) );
    } while( l.ttype == ParserTagConstants.COMMA );
    return TypeNameVec.popFromStackVector( seqTypeName );
  }
  
/************************************************************************ 

Parse a <TT>TypeDeclElem</TT>, which is either a field, method, or
constructor declaration, a static block, or a TypeDecl [1.1].  

<p> A field declaration may define many fields.  Since returning
multiple declared entities is cumbersome, this method simply adds all
the declared entities onto the StackVector seqTypeDeclElem.  The
<TT>keyword</TT> argument is either CLASS or INTERFACE.  The argument
containerId is the name of the enclosing class, which is necessary for
checking constructor declarations.

<PRE>
TypeDeclElem: 
   FieldDeclaration 
   MethodDeclaration
   ConstructorDeclaration 
   InitBlock
   TypeDeclElemPragma
   TypeDeclaration              [1.1]
   ;

FieldDeclaration:
   Modifiers Type VariableDeclarator (, VariableDeclarator)* ;

MethodDeclaration:
   Modifiers Type Identifier FormalParameterList BracketPair*
      [throws TypeNameList]
      ( ; | Block )

ConstructorDeclaration:
   Modifiers Identifier FormalParameterList Block

InitBlock:
   Modifiers Block

VariableDeclarator:
   Identifier BRACKET_PAIR* [ = VariableInitializer ]

</PRE>

@see javafe.parser.ParserTagConstants

*/

  protected TypeDeclElem
  parseTypeDeclElemIntoSeqTDE(Lex l, int keyword, /*@non_null*/Identifier containerId,
				   boolean specOnly)			
  {
    int loc = l.startingLoc;
    int modifiers = parseModifiers(l);
    //alx: dw clone the array to use it later
    int[] localUniverseArray = null;
    if (useUniverses)
    	localUniverseArray = (int[])this.universeArray.clone();
    //alx-end

    ModifierPragmaVec modifierPragmas = this.modifierPragmas;
    TypeDeclElem result = null;

    if( l.ttype == ParserTagConstants.SEMICOLON 
       && modifiers == Modifiers.NONE
       && modifierPragmas == null ) {
      // Semicolons are not in JLS, 
      // but accepted by javac and in many java progs
      // so allow them and do nothing
      l.getNextToken();
      return null;
    } 
    else if( l.ttype == ParserTagConstants.CLASS 
	  || l.ttype == ParserTagConstants.INTERFACE) {
      /* Nested class/interface */
      TypeDecl nested = parseTypeDeclaration(l, specOnly);
      nested.modifiers = modifiers;
      nested.pmodifiers = modifierPragmas;
      seqTypeDeclElem.addElement(nested);
      return nested;
    } 
    else if( l.ttype == ParserTagConstants.LBRACE ) {
      /* Initialization block */
      if( keyword == ParserTagConstants.INTERFACE ) 
        fail(l.startingLoc, 
	     "Cannot have initializer blocks in an interface");
      if (specOnly)
	parseBlock(l, true);
      else {
	seqTypeDeclElem.addElement( result = InitBlock.make( modifiers, modifierPragmas,
						    parseBlock(l,false) ) );
      }
      return result;
    } 
    else if( atStartOfConstructorOrMethod(l) ) {
      // Constructor declaration 
      if( keyword == ParserTagConstants.INTERFACE ) 
        fail(l.startingLoc, "Cannot declare constructors in an interface");
      int locId = l.startingLoc;
      Identifier id = parseIdentifier(l);
      TypeModifierPragmaVec tmodifiers = parseTypeModifierPragmas(l);
      if (id != containerId) {
	  if (containerId.toString().startsWith("$anon_"))
	      fail(l.startingLoc,
		   "Anonymous classes may not have constructors");
	  else
	      fail(l.startingLoc, 
		   "Invalid name '"+id+"' for constructor;"
		   +" expected '"+containerId+"'");
      }

      FormalParaDeclVec args = parseFormalParameterList(l);
      int locThrowsKeyword;
      if (l.ttype == ParserTagConstants.THROWS) {
	locThrowsKeyword = l.startingLoc;
      } else {
	locThrowsKeyword = Location.NULL;
      }
      TypeNameVec raises = parseTypeNames( l, ParserTagConstants.THROWS );

      // allow more modifier pragmas
      modifierPragmas = parseMoreModifierPragmas( l, modifierPragmas );
      BlockStmt body = null;
      int locOpenBrace = Location.NULL;
      if ( l.ttype == ParserTagConstants.SEMICOLON ) {
          l.getNextToken();   // swallow semicolon
      } else {
	  locOpenBrace = l.startingLoc;
	  // specOnly means do not keep any bodies of methods/constructors/etc.
	  body = specOnly ? parseBlock(l, true)
				    : parseConstructorBody(l);

      }
      ConstructorDecl cd = ConstructorDecl.make( modifiers,
				    modifierPragmas,
				    tmodifiers,
				    args, 
				    raises, body,
				    locOpenBrace,
				    loc, locId, 
				    locThrowsKeyword );
      cd.specOnly = specOnly;
      seqTypeDeclElem.addElement( cd);
      return cd;
    } 
    else if( l.ttype == ParserTagConstants.TYPEDECLELEMPRAGMA ) {
      // TypeDeclElemPragma
      if( modifiers != Modifiers.NONE)
	ErrorSet.error(l.startingLoc, 
	     "Cannot have modifiers outside of the annotation on a TypeDeclElem pragma");
      TypeDeclElemPragma pragma = (TypeDeclElemPragma)l.auxVal;
      pragma.decorate(modifierPragmas);
      seqTypeDeclElem.addElement( pragma );
      l.getNextToken();
    }
    else if (l.ttype == ParserTagConstants.POSTMODIFIERPRAGMA) {
	ErrorSet.error(l.startingLoc,
	    "Ignoring a modifier pragma that presumably follows a field declaration but is not in the same annotation comment"); // FIXME - can this be fixed?
      l.getNextToken();
    }
    else {
      /* Is field or method declaration */
      int locType = l.startingLoc;
      Type type = parseType(l);
      if(  atStartOfConstructorOrMethod(l) ) {
        // Method declaration 
        
        int locId              = l.startingLoc;
        Identifier id          = parseIdentifier(l);
	TypeModifierPragmaVec tmodifiers = parseTypeModifierPragmas(l);
        FormalParaDeclVec args = parseFormalParameterList(l);
        type = parseBracketPairs( l, type );
	int locThrowsKeyword;
	if (l.ttype == ParserTagConstants.THROWS) {
	  locThrowsKeyword = l.startingLoc;
	} else {
	  locThrowsKeyword = Location.NULL;
	}
        TypeNameVec raises     = parseTypeNames(l, ParserTagConstants.THROWS );
	int locOpenBrace;
        BlockStmt body;

	// Allow some more modifier pragmas here
	modifierPragmas = parseMoreModifierPragmas( l, modifierPragmas );

        if ( l.ttype == ParserTagConstants.SEMICOLON ) {
          l.getNextToken();   // swallow semicolon
	  locOpenBrace = Location.NULL;
          body = null;
// FIXME - check that this is specOnly for no body ?
        } else {
          if( keyword == ParserTagConstants.INTERFACE ) 
            fail(l.startingLoc, 
		 "Cannot define a method body inside an interface");
	  locOpenBrace = l.startingLoc;
          body = parseBlock(l, specOnly);
        }
        MethodDecl md = MethodDecl.make(modifiers, modifierPragmas, tmodifiers,
					args, 
					raises, body, locOpenBrace,
                                        loc, locId, locThrowsKeyword,
					id, type, locType);
        md.specOnly = specOnly;
        //alx: dw save universe return type to method declaration node
        if (useUniverses)
        	setUniverse(md,localUniverseArray,type,locType);
	//alx-end

        seqTypeDeclElem.addElement( md );
        return md;

      } else {

        // Field declaration. 
        
        // Have modifiers and type. 
        // May need to loop over many VariableDeclarators 

	// make modifierPragmas non-null, so can retroactively extend
	if( modifierPragmas == null )
	  modifierPragmas = ModifierPragmaVec.make();

        List fds = new LinkedList(); // a list of all the fds produced here
        for(;;) {
          int locId     = l.startingLoc;
          Identifier id = parseIdentifier(l);
          Type vartype = parseBracketPairs(l, type);

          VarInit init = null;
	  int locAssignOp = Location.NULL;
          if( l.ttype == ParserTagConstants.ASSIGN ) {
	    locAssignOp = l.startingLoc;
            l.getNextToken();
            init = parseVariableInitializer(l, specOnly);
          }
          FieldDecl fielddecl
	    = FieldDecl.make(modifiers, modifierPragmas.copy(), 
			     id, vartype, locId, init, locAssignOp );
          //alx: dw set universes of field decls
          if (useUniverses)
          	setUniverse(fielddecl,localUniverseArray);
	  //alx-end

          seqTypeDeclElem.addElement( fielddecl );
	  fds.add(fielddecl);

          if(l.ttype == ParserTagConstants.MODIFIERPRAGMA
		    || l.ttype == ParserTagConstants.SEMICOLON ) 
	  {
	    // if modifier pragma, retroactively add to modifierPragmas
	    parseMoreModifierPragmas( l, fielddecl.pmodifiers );

	    // End of Declaration 

	    // JML added some clauses that can follow type declarations.
	    // This bit of hackery is to check if there are any such
	    // and associate them with the correct declaration.  All other
	    // modifiers precede the declaration with which they are 
	    // associated (or at least precede the terminating semicolon).
            l.getNextToken();
	    while (l.ttype == ParserTagConstants.POSTMODIFIERPRAGMA) {
		Identifier idd = ((javafe.ast.IdPragma)l.auxVal).id();
		Iterator i = fds.iterator();
		while (i.hasNext()) {
		    FieldDecl fd = (FieldDecl)i.next();
		    if (idd == null || idd == fd.id)
			fd.pmodifiers.addElement((ModifierPragma)l.auxVal);
		}
		l.getNextToken();
	    }

            return fielddecl;
          } else {
            expect( l, ParserTagConstants.COMMA );
            /* And go around loop again */
          }
        } /* loop thru field declarations */
      } /* is field declarations */
    } /* field or method declarations */
    return null;
  } /* parseTypeDeclElemIntoSeqTDE */

  /**********************************************************************
    Parse a <TT>FormalParameterList</TT>, which includes enclosing parens.
    Note: this definition differs from JLS, where it does not include the
    parens

    <PRE>
      FormalParameterList:
        LPAREN FormalParameter (, FormalParameter)* RPAREN

      FormalParameter:
        [Modifiers] Type Identifier BracketPair* ModifierPragma*     [1.1]
    <PRE>       
   */

  //@ requires l != null && l.m_in != null;
  //@ ensures \result != null;
  public FormalParaDeclVec parseFormalParameterList(Lex l) 
  {
    /* Should be on LPAREN */
    if( l.ttype != ParserTagConstants.LPAREN ) 
      fail(l.startingLoc, "Expected open paren");
    if( l.lookahead(1) == ParserTagConstants.RPAREN ) {
      // Empty parameter list
      expect( l, ParserTagConstants.LPAREN );
      expect( l, ParserTagConstants.RPAREN );
      return FormalParaDeclVec.make();
    } else {
      seqFormalParaDecl.push();
      while( l.ttype != ParserTagConstants.RPAREN ) {
        l.getNextToken();                // swallow open paren or comma
	int modifiers = parseModifiers(l);
	//alx: dw save array to use it later
	int[] localUniverseArray = null;
	if (useUniverses)
		localUniverseArray = (int[]) this.universeArray.clone();
	//alx-end

	ModifierPragmaVec modifierPragmas = this.modifierPragmas;
        Type type = parseType(l);
        int locId = l.startingLoc;
        Identifier id = parseIdentifier(l);
        type = parseBracketPairs(l, type);
	modifierPragmas = parseMoreModifierPragmas(l, modifierPragmas);
	//alx: dw save array to the var decl node and add the node to the 
        //        vector
	if (useUniverses) seqFormalParaDecl.addElement( 
		             setUniverse(FormalParaDecl.make(modifiers, 
                                                             modifierPragmas, 
                                                             id, 
                                                             type, 
                                                             locId ),
                             localUniverseArray) );
	else
	//alx-end
        seqFormalParaDecl.addElement( FormalParaDecl.make(modifiers,
							  modifierPragmas, 
							  id, type, locId ) );
        if( l.ttype != ParserTagConstants.RPAREN && l.ttype != ParserTagConstants.COMMA )
          fail(l.startingLoc, "Expected comma or close paren");
      }
      expect( l, ParserTagConstants.RPAREN );
      return FormalParaDeclVec.popFromStackVector( seqFormalParaDecl );
    }
  }
}
