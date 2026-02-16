/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

import javafe.ast.*;
//alx: 
import javafe.tc.FlowInsensitiveChecks;
import javafe.util.ErrorSet;
//alx-end
import javafe.util.Location;
import javafe.util.StackVector;
import javafe.util.Assert;
import javafe.Tool;

//@ model import javafe.tc.TagConstants;

/**
 * Parses java expressions.  Extended by {@link javafe.parser.ParseStmt}.
 * 
 * @see javafe.ast.ASTNode
 * @see javafe.parser.ParseType
 * @see javafe.parser.ParseStmt
 */

public abstract class ParseExpr extends ParseType
{
    // ----------------------------------------------------------------------
    // Operator precedence parser

    // We need a stack of < Expr, op, precedence, locOp >
    // We keep it in these four parallel arrays, and resize as necessary.

    private int defaultStackSize = 12;

    //@ private invariant stackPtr >= -1;
    //@ private invariant stackPtr < exprStack.length;
    private int stackPtr = -1;         // Always points to top element, if any

    //@ private invariant exprStack != null;
    //@ private invariant exprStack.length > 0;
    //@ private invariant \typeof(exprStack) == \type(Expr[]);
    private Expr exprStack[]      = new Expr[defaultStackSize];

    //@ private invariant opStack != null;
    //@ private invariant opStack.length == exprStack.length;
    private int opStack[]         = new int[defaultStackSize];

    //@ private invariant precedenceStack != null;
    //@ private invariant precedenceStack.length == exprStack.length;
    private int precedenceStack[] = new int[defaultStackSize];

    //@ private invariant locStack != null;
    //@ private invariant locStack.length == exprStack.length;
    private int locStack[]        = new int[defaultStackSize];

    // The operator precedence parser can be extended with additional
    // binary operators.
    // We keep the precedence and associtivity of each operator
    // in the following tables. The precedence is 0 for tokens that are
    // not operators. These tables are sized as necessary.

    //@ private invariant precedenceTable != null;
    private int[] precedenceTable    = new int[0];

    //@ private invariant isLeftAssocTable != null;
    //@ private invariant isLeftAssocTable.length == precedenceTable.length;
    private boolean[] isLeftAssocTable = new boolean[0];

    /** If no constructors are found in "elems", adds a default one to
     it.  If a default constructor is created, the "loc" and "locId"
     fields of the default constructor will be set to "loc".
     A declaration of this method is needed because class declarations
     can be in expressions.  However, the body lives in Parse.java */
    //@ requires loc != Location.NULL;
    //@ requires elems != null;
    abstract void addDefaultConstructor(TypeDeclElemVec elems, int loc, boolean specOnly);

    /** Parse an element of a type declaration into "seq".
     "keyword" should be the kind of type decl, one of CLASS or INTERFACE.
     "containerId" should be the name of the containing type decl.
     Lives in Parse.java; more doc can be found there.  */
    //@ requires l != null && l.m_in != null;
    abstract TypeDeclElem parseTypeDeclElemIntoSeqTDE(Lex l, int keyword,
                                              /*@ non_null @*/ Identifier
                                              containerId,
                                              boolean specOnly);

    /**
     * Add an infix, binary operator to the parser with a given
     * precedence and associativity.  The precedence level must be
     * greater than zero.  The following table gives the precedence
     * levels assigned to the built-in Java operators
     *
     * <center><table>
     * <tr><td><b>Operator</b></td><td><b>Precedenc</b></td><td><b>Assoc</b></td>
     * </tr>
     * <tr><td>TagConstants.STAR</td><td>170</td><td>left</td></tr>
     * <tr><td>TagConstants.DIV</td><td> 170</td><td>left</td></tr>
     * <tr><td>TagConstants.MOD</td><td> 170</td><td>left</td></tr>
     * <tr><td>TagConstants.ADD</td><td> 160</td><td>left</td></tr>
     * <tr><td>TagConstants.SUB</td><td> 160</td><td>left</td></tr>
     * <tr><td>TagConstants.LSHIFT</td><td>150</td><td>left</td></tr>
     * <tr><td>TagConstants.RSHIFT</td><td>150</td><td>left</td></tr>
     * <tr><td>TagConstants.URSHIFT</td><td>150</td><td>left</td></tr>
     * <tr><td>TagConstants.GE</td><td>140</td><td>left</td></tr>
     * <tr><td>TagConstants.GT</td><td>140</td><td>left</td></tr>
     * <tr><td>TagConstants.LT</td><td>140</td><td>left</td></tr>
     * <tr><td>TagConstants.LE</td><td>140</td><td>left</td></tr>
     * <tr><td>TagConstants.INSTANCEOF</td><td>140</td><td>left</td></tr>
     * <tr><td>TagConstants.EQ</td><td>130</td><td>left</td></tr>
     * <tr><td>TagConstants.NE</td><td>130</td><td>left</td></tr>
     * <tr><td>TagConstants.BITAND</td><td>120</td><td>left</td></tr>
     * <tr><td>TagConstants.BITXOR</td><td>110</td><td>left</td></tr>
     * <tr><td>TagConstants.BITOR</td><td>100</td><td>left</td></tr>
     * <tr><td>TagConstants.AND</td><td>90</td><td>left</td></tr>
     * <tr><td>TagConstants.OR</td><td>80</td><td>left</td></tr>
     * <tr><td>TagConstants.QUESTIONMARK</td><td>70</td><td>left</td></tr>
     * <tr><td>TagConstants.ASSIGN</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGMUL</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGDIV</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGREM</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGADD</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGSUB</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGLSHIFT</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGRSHIFT</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGURSHIFT</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGBITAND</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGBITOR</td><td>60</td><td>right</td></tr>
     * <tr><td>TagConstants.ASGBITXOR</td><td>60</td><td>right</td></tr>
     * </table></center>
     * 
     * <p> (The operators <code>?</code> and <code>instanceof</code> are
     * treated specially by the parser, but this special treatment
     * respects the precedence levels indicated above.)
     */
   
    //@ requires ttype >= 0;
    public void addOperator(int ttype,
                            int precedence, 
                            boolean isLeftAssoc) 
    {

        if( precedenceTable.length <= ttype )
        {
            // expand
            int[] nuPrecedenceTable = new int[ ttype+1 ];
            System.arraycopy(precedenceTable, 0, nuPrecedenceTable, 0, 
                             precedenceTable.length );
            precedenceTable = nuPrecedenceTable;
            boolean[] nuIsLeftAssocTable = new boolean[ ttype+1 ];
            System.arraycopy(isLeftAssocTable, 0, nuIsLeftAssocTable, 0, 
                             isLeftAssocTable.length );
            isLeftAssocTable = nuIsLeftAssocTable;
        }
        precedenceTable[ ttype ] = precedence;
        isLeftAssocTable[ ttype ] = isLeftAssoc;
    }


    /** 
     * Parse an <tt>Expression</tt>.
     *
     * <p> Does operator-precedence parsing of a large amount of the
     * <tt>Expression</tt> hierarchy, all the way down to
     * <tt>UnaryExpression</tt>.
     *
     * <pre>
     * Expression:
     * UnaryExpression
     * Expression BinaryOp Expression
     * Expression instanceof Type
     * Expression ? Expression : Expression
     * 
     * BinaryOp: one of
     * STAR / % PLUS - << >> >>> > >= < <= == != & BITOR ^ && || 
     * = *= /= %= += -= <<= >>= >>>= &= |= ^=
     * </pre>
     * 
     * <p> This grammar is ambiguous; the precedence parsing machinery
     * resolves the ambiguity appropriately, according to the grammar
     * in chapter 19 of "The Java Language Specification".
     */
      
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public Expr parseExpression(Lex l) {
        // Save old stack pointer, so re-enterable
        int baseStackPtr = stackPtr;
    
        getExpression:
        for(;;) {
      
            // Get a UnaryExpression
            Expr e = parseUnaryExpression(l);
      
            getOp: 
            for(;;) {
        
                // Get following op
                int op = l.ttype;
                int locOp = l.startingLoc;
        
                // Figure out if op really is an operator, 
                // and if so, what is its precedence
                int precedence;
                boolean isLeftAssoc;

                if( 0 <= op && op < precedenceTable.length )
                {
                    precedence = precedenceTable[ op ];
                    isLeftAssoc = isLeftAssocTable[ op ];
                }
                else
                {
                    precedence = 0;
                    isLeftAssoc = false; // dummy value
                }

                // while precedence of new token is lower than that of
                // top operator on stack, do a reduction 
                // Combine the top stack expression, the top stack operator,
                // and the expression e to produce a new expression e
        
                //@ loop_invariant stackPtr < exprStack.length;
                while( stackPtr > baseStackPtr 
                       &&
                       (( isLeftAssoc && precedenceStack[stackPtr] >= precedence )
                        || 
                        ( !isLeftAssoc && precedenceStack[stackPtr] > precedence ))) 
                {
                    Expr ee = exprStack[stackPtr];
                    int binop = opStack[stackPtr];
                    LiteralExpr res = null;
                    if (ee instanceof LiteralExpr &&
                            e instanceof LiteralExpr) {
                      res = binaryConstantFolding(
                              (LiteralExpr)ee, binop, (LiteralExpr)e);
                      // returns null if no folding could be done;
                      // returns the new literal if the op could be folded
                    } 
                    if (res != null) {
                      e = res;
                    } else {
                      e = BinaryExpr.make(binop, 
                                        ee, 
                                        e, 
                                        locStack[stackPtr] );
                    }
                    stackPtr --;
                }
        
                // Now deal with new operator
        
                if( precedence == 0 ) 
                {
                    // We dont have an operator, so end of Expression
                    if( stackPtr != baseStackPtr )  
                        fail(l.startingLoc, 
                             "Internal error in operator precedence parser");
                    return e;
                }
                else if( op == ParserTagConstants.INSTANCEOF ) {
                    // get ReferenceType, and reduce
                    l.getNextToken();
                    //alx: dw parse the universe modifiers
                    int[] localUniverseArray=null;
                    if (useUniverses) {
                    	parseUniverses(l);
                    	localUniverseArray = 
			    (int[]) this.universeArray.clone();
                    }
		    //alx-end
                    Type t = parseType( l );
                    e = InstanceOfExpr.make( e, t, locOp );

                    //alx: dw set the universe modifiers
                    if (useUniverses)
                    	setUniverse(e,localUniverseArray,t,locOp);
		    //alx-end
          
                    // Now go to check following op
                    continue getOp;
                } else if( op == ParserTagConstants.QUESTIONMARK ) {
                    l.getNextToken();
                    Expr thn = parseExpression( l );
                    int locColon = l.startingLoc;
                    expect( l, ParserTagConstants.COLON );
                    Expr els = parseExpression( l );
                    e = CondExpr.make( e, thn, els, locOp, locColon);
          
                    // Now go to check following op
                    // This will never be a proper op, 
                    // so we will just pop stack, reduce and return
                    continue getOp;
                } else {
                    l.getNextToken();
                    // Put Expression and operator on top of the stack;
                    stackPtr++;

                    // check if we need to resize arrays

                    if( stackPtr == exprStack.length ) {
            
                        Expr exprStack2[]      = new Expr[2*exprStack.length];
                        System.arraycopy( exprStack, 0, exprStack2, 0, stackPtr);
                        exprStack = exprStack2;

                        int opStack2[]      = new int[2*opStack.length];
                        System.arraycopy( opStack, 0, opStack2, 0, stackPtr);
                        opStack = opStack2;

                        int precedenceStack2[]      = new int[2*precedenceStack.length];
                        System.arraycopy( precedenceStack, 0, precedenceStack2, 0, 
                                          stackPtr);
                        precedenceStack = precedenceStack2;

                        int locStack2[]      = new int[2*locStack.length];
                        System.arraycopy( locStack, 0, locStack2, 0, stackPtr);
                        locStack = locStack2;

                    } // resized

                    exprStack[stackPtr] = e;
                    opStack[stackPtr] = op;
                    precedenceStack[stackPtr] = precedence;
                    locStack[stackPtr] = locOp;
                    continue getExpression;
                }
                // We never get down to here
            }
        }
    }
    
    protected LiteralExpr binaryConstantFolding(LiteralExpr left, int op, LiteralExpr right) {
      if (op == ParserTagConstants.ADD) {
        if (left.tag == ParserTagConstants.INTLIT &&
                right.tag == ParserTagConstants.INTLIT) {
          return LiteralExpr.make(ParserTagConstants.INTLIT, 
                  new Integer(((Integer)left.value).intValue() + ((Integer)right.value).intValue()), left.loc);
        }
        if (left.tag == ParserTagConstants.STRINGLIT &&
                right.tag == ParserTagConstants.STRINGLIT) {
          return LiteralExpr.make(ParserTagConstants.STRINGLIT, 
                  (String)(left.value) + (String)(right.value), left.loc);
        }
      }
      return null;
    }

    /**********************************************************************

     Parse a <TT>UnaryExpression</TT>.

     <PRE>
     UnaryExpression:
     PLUS UnaryExpression
     -  UnaryExpression
     ++ UnaryExpression
     -- UnaryExpression
     ~  UnaryExpression
     !  UnaryExpression
     PrimaryExpression
     CastExpression
     </PRE>

     A <TT>CastExpression</TT> (as opposed to a <TT>PrimaryExpression</TT>)
     is recognised by the lookahead sequences:

     <PRE>
     LPAREN PrimitiveType
     LPAREN Name (LSQBRACKET RSQBRACKET)* RPAREN X
     </PRE>

     where <TT>X</TT> is the first token of a <TT>UnaryExpressionNotPlusMinus</TT>,
     cf. isStartOfUnaryExpressionNotPlusMinus(-).

     */

    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public Expr parseUnaryExpression(Lex l) {
        Expr primary;

        switch( l.ttype ) {
            case ParserTagConstants.SUB: {
                int locOp = l.startingLoc;
                l.getNextToken();
                if (l.ttype == ParserTagConstants.MAX_INT_PLUS_ONE) {
                    l.getNextToken();
                    return LiteralExpr.make(ParserTagConstants.INTLIT,
                                            new Integer(Integer.MIN_VALUE), locOp);
                } else if (l.ttype == ParserTagConstants.MAX_LONG_PLUS_ONE) {
                    l.getNextToken();
                    return LiteralExpr.make(ParserTagConstants.LONGLIT,
                                            new Long(Long.MIN_VALUE), locOp);
                } else return UnaryExpr.make(OperatorTags.UNARYSUB,
                                             parseUnaryExpression(l), locOp);
            }
				 
            case ParserTagConstants.ADD:
            case ParserTagConstants.INC: case ParserTagConstants.DEC: case ParserTagConstants.NOT: case ParserTagConstants.BITNOT:
                {
                    int op = l.ttype;
                    int locOp = l.startingLoc;
                    l.getNextToken();
                    if (op == ParserTagConstants.ADD) op = OperatorTags.UNARYADD;
                    return UnaryExpr.make(op, parseUnaryExpression(l), locOp);
                }

            default:
                // Need CastExpression or PrimaryExpression
                // For CastExpression, need lookahead
                //    LPAREN PrimitiveType
                // or
                //    LPAREN Name (LSQBRACKET RSQBRACKET)* RPAREN X
                // where X is the first token of a UnaryExpressionNotPlusMinus,
                // ie X is one of ~ ! Literal Id this new super LPAREN
                //
                // TypeModifierPragmas may appear between LSQBRACKET RSQBRACKET
                //  or after the Name.
                if( l.ttype == ParserTagConstants.LPAREN ) {

                    int i = 1; // lookahead counter

                    // count leading left parens: handles case for ((A))a;
                    int parenCount = 1;
                    while (l.lookahead(i) == ParserTagConstants.LPAREN) {
                        parenCount++;
                        i++;
                    }

                    //alx: dw go over the universe modifiers
                    if (useUniverses)
                    {
                    	int universeTag = l.lookahead(i);
                    	while (universeTag == ParserTagConstants.PEER || 
			       universeTag == ParserTagConstants.REP || 
			       universeTag == ParserTagConstants.READONLY)
			    universeTag = l.lookahead(++i);
                    }
		    //alx-end

                    switch( l.lookahead(i) ) {
                        case ParserTagConstants.IDENT:
                            { 
                                // Look for Name (LSQBRACKET RSQBRACKET)* RPAREN X
                                i += 1;
                                while( l.lookahead(i) == ParserTagConstants.FIELD && 
                                       l.lookahead(i+1) == ParserTagConstants.IDENT ) 
                                    i+=2;

                                // skip over TypeModifierPragmas
                                while (l.lookahead(i) == ParserTagConstants.TYPEMODIFIERPRAGMA) i++;

                                // skip over ([ TypeModifierPragma ])*
                                while(true) {
                                    int temp = i;
                                    if (l.lookahead(i) != ParserTagConstants.LSQBRACKET) break;
                                    i++;
                                    while (l.lookahead(i) == ParserTagConstants.TYPEMODIFIERPRAGMA) i++;
                                    if (l.lookahead(i) != ParserTagConstants.RSQBRACKET) {
                                        i = temp; break;
                                    }
                                    i++;
                                }
                                if( l.lookahead(i) == ParserTagConstants.RPAREN ) {
                                    // Look for prefix of UnaryExpressionNotPlusMinus
                                    parenCount--;

                                    // skip over any other parens:  ((A))a;
                                    while (l.lookahead(i + 1) == ParserTagConstants.RPAREN && parenCount > 0) {
                                        parenCount--;
                                        i++;
                                    }

                                    if (parenCount != 0) {
                                        return parsePrimaryExpression(l);
                                    }
		
                                    if (isStartOfUnaryExpressionNotPlusMinus(l.lookahead(i + 1)))
                                        return parseCastExpression(l);
                                    else
                                        // Not followed by UnaryExpressionNotPlusMinus prefix,
                                        // so is PrimaryExpression
                                        return parsePrimaryExpression(l);
                                } else {
                                    // Not RPAREN, so is PrimaryExpression
                                    return parsePrimaryExpression(l);
                                }
                            }
	  
                        default:
                            if (isPrimitiveKeywordTag(l.lookahead(1)))
                                return parseCastExpression(l);

                            // Token after LPAREN not IDENT or PrimitiveType
                            return parsePrimaryExpression(l);
                    }
                } else {
                    // Expression does not start with LPAREN
                    return parsePrimaryExpression(l);
                }
        }
    } // End parseUnaryExpression



    /**
     * Determines whether the tag is the first token of a
     * <TT>UnaryExpressionNotPlusMinus</TT>.
     *
     * For the default Java grammar, this amounts to is tag one of:
     *
     *   ~ ! Literal Id this new super LPAREN
     *
     * However, it is expected that grammar extensions may extend
     * this list.
     */
    public boolean isStartOfUnaryExpressionNotPlusMinus(int tag) {
	// Look for prefix of UnaryExpressionNotPlusMinus
	switch (tag) {
            case ParserTagConstants.BITNOT: 
            case ParserTagConstants.NOT:

                // All literals:
            case ParserTagConstants.CHARLIT: case ParserTagConstants.INTLIT:
            case ParserTagConstants.LONGLIT: case ParserTagConstants.FLOATLIT:
            case ParserTagConstants.DOUBLELIT: case ParserTagConstants.STRINGLIT:
            case ParserTagConstants.TRUE: case ParserTagConstants.FALSE:
            case ParserTagConstants.NULL:

            case ParserTagConstants.IDENT:
            case ParserTagConstants.THIS:
            case ParserTagConstants.NEW:
            case ParserTagConstants.SUPER:
            case ParserTagConstants.LPAREN:

                return true;

            default:
                return false;
	}
    }

    /**********************************************************************

     Parse a <TT>CastExpression</TT>.

     <PRE>
     CastExpression:
     ( PrimitiveType Dimsopt ) UnaryExpression
     ( Name Dimsopt ) UnaryExpressionNotPlusMinus
     <PRE>

     The non-terminal <TT>UnaryExpressionNotPlusMinus</TT> describes a
     subset of <TT>UnaryExpression</TT> as described in chapter 19 of "The
     Java Language Specification"

     */


    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public Expr parseCastExpression(Lex l) {
        int locOpen = l.startingLoc;
        expect( l, ParserTagConstants.LPAREN );

        int parenCount = 1;
        // count leading parens to handle ((A))a
        while (l.ttype == ParserTagConstants.LPAREN) {
            parenCount++;
            expect( l, ParserTagConstants.LPAREN );
        }

        //alx: dw parse modifiers
        int[] localUniverseArray = null;
        if (useUniverses) {
        	parseUniverses(l);
        	localUniverseArray = (int[]) this.universeArray.clone();
        }
	//alx-end

        Type castType = parseType(l);
        int locClose = l.startingLoc;

        // match all leading parens
        while (parenCount > 0) {
            expect( l, ParserTagConstants.RPAREN );
            parenCount--;
        }
    
        Expr exprAfterCast = parseUnaryExpression(l);
        //alx: dw save modifiers in node
        CastExpr ce = CastExpr.make( exprAfterCast, castType, 
				     locOpen, locClose );
        if (useUniverses)
	    setUniverse(ce,localUniverseArray,castType,locOpen);
        return ce;
	//alx-end
    }

    /**********************************************************************

     Parse a <TT>PrimaryExpression</TT>.

     <PRE>
     PrimaryExpression:
     PrimaryCore PrimarySuffix*
 
     PrimaryCore:
     Literal
     Name
     Name ArgumentList
     this
     super . Identifier
     super . Identifier ArgumentList
     NewExpression
     LPAREN Expression RPAREN
     TypeName . this			[1.1]
     Type . class				[1.1]
     (This allows void . class because we treat void as a primitive type)

     PrimarySuffix:
     ++
     --
     LSQBRACKET Expression RSQBRACKET
     . Identifier
     . Identifier ArgumentList
     </PRE>

     */

    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    protected Expr parsePrimaryExpression(Lex l) {
        Expr primary;

        /* hack to handle ((A))a as a typecast. 
         if (l.lookahead(0) == TagConstants.LPAREN &&
         l.lookahead(1) == TagConstants.LPAREN &&
         l.lookahead(2) == TagConstants.IDENT &&
         l.lookahead(3) == TagConstants.RPAREN &&
         l.lookahead(4) == TagConstants.RPAREN &&
         l.lookahead(5) == TagConstants.IDENT) {
         return parseCastExpression2(l);
         } */

        // First parse PrimaryCore into variable primary
        switch( l.ttype ) {
      
            // --- First try literals: Need to fix literal interface to Lex
      
            case ParserTagConstants.CHARLIT:
            case ParserTagConstants.DOUBLELIT:
            case ParserTagConstants.FLOATLIT:
            case ParserTagConstants.INTLIT:
            case ParserTagConstants.LONGLIT:
            case ParserTagConstants.STRINGLIT:
                primary = LiteralExpr.make(l.ttype, l.auxVal, l.startingLoc);
                l.getNextToken();
                break;

            case ParserTagConstants.TRUE:
                primary = LiteralExpr.make( ParserTagConstants.BOOLEANLIT, Boolean.TRUE, l.startingLoc );
                l.getNextToken();
                break;

            case ParserTagConstants.FALSE:
                primary = LiteralExpr.make( ParserTagConstants.BOOLEANLIT, Boolean.FALSE, l.startingLoc );
                l.getNextToken();
                break;

            case ParserTagConstants.NULL:
                primary = LiteralExpr.make( ParserTagConstants.NULLLIT, null, l.startingLoc );
                l.getNextToken();
                break;

                // Get here => not a literal

                // Try Name, Name ( ArgumentListopt ), Name []..., Name . class,
                //    Name . this
                //
                // Note that TypeModifierPragmas may appear between
                // Name and (.
            case ParserTagConstants.ASSERT:
                // Only process if assert is *not* a keyword.
                if (Tool.options == null || Tool.options.assertIsKeyword) {
                    fail(l.startingLoc, "\"assert\" is a Java keyword when you use the" +
                         " -source 1.4 option; rename this identifier.");
                }
		// fall-through
            case ParserTagConstants.IDENT: 
                {
                    Name n = parseName(l);
                    TypeModifierPragmaVec tmodifiers = null;
                    /*
                     // Look for type modifiers on Name
                     if (l.ttype == TagConstants.TYPEMODIFIERPRAGMA)	{
                     tmodifiers = parseTypeModifierPragmas(l);
                     }
                     */
                    // May be followed by ( ArgumentListopt ) :
                    if (l.ttype == ParserTagConstants.LPAREN) {
                        int locOpenParen = l.startingLoc;
                        ExprVec args = parseArgumentList(l);
                        primary = AmbiguousMethodInvocation.make(n, tmodifiers,
                                                                 locOpenParen, args);
                        break;
                    }

                    // Look for 'TypeName . this'
                    if (l.lookahead(0) == ParserTagConstants.FIELD &&
                        l.lookahead(1) == ParserTagConstants.THIS) {
                        expect( l, ParserTagConstants.FIELD );
                        int locThis = l.startingLoc;
                        expect( l, ParserTagConstants.THIS );
                        primary = ThisExpr.make(TypeName.make(n), locThis);
                        break;
                    }

                    // Or ([])* . class:
                    // (need to look ahead fully because of "<type>[] x;" declarations)
                    int i = 0;
                    while ( l.lookahead(i) == ParserTagConstants.LSQBRACKET &&
                            l.lookahead(i+1) == ParserTagConstants.RSQBRACKET )
                        i += 2;
                    if (l.lookahead(i) == ParserTagConstants.FIELD &&
                        l.lookahead(i+1) == ParserTagConstants.CLASS) {
                        Type t = TypeName.make(n);
                        t = parseBracketPairs(l, t);
                        primary = parseClassLiteralSuffix(l, t);
                        break;
                    }

                    // Else, just an AmbiguousVariableAccess...
                    primary = AmbiguousVariableAccess.make( n );
                    break;
                }
      
            case ParserTagConstants.SUPER:
                {
                    int locSuper = l.startingLoc;
                    Name n = parseSuper(l);
                    int locDot = l.startingLoc;
                    expect( l, ParserTagConstants.FIELD );
                    int locId = l.startingLoc;
                    Identifier id = parseIdentifier(l);
                    ObjectDesignator od = SuperObjectDesignator.make( locDot, locSuper );

                    // super may be follows by type modifiers.
                    if( l.ttype == ParserTagConstants.LPAREN ||
                        l.ttype == ParserTagConstants.TYPEMODIFIERPRAGMA) {
	  
                        TypeModifierPragmaVec tmodifiers = null;
                        if (l.ttype == ParserTagConstants.TYPEMODIFIERPRAGMA)	{
                            tmodifiers = parseTypeModifierPragmas(l);
                        }
                        // is a super method invocation
                        // PrimaryCore ::= super . Identifier ( ArgumentListopt )
                        int locOpenParen = l.startingLoc;
                        ExprVec args = parseArgumentList(l);
                        primary = MethodInvocation.make(od, id, tmodifiers,
                                                        locId, locOpenParen, args);
                    } else {
                        // is super field access
                        // PrimaryCore ::= super
                        primary = FieldAccess.make( od, id, locId );
                    }
                    break;
                }
      
            case ParserTagConstants.THIS:
                primary = ThisExpr.make(null, l.startingLoc);
                l.getNextToken();
                break;
      
            case ParserTagConstants.NEW:
                primary = parseNewExpression(l);
                break;
      
            case ParserTagConstants.LPAREN:
                // LPAREN Expression RPAREN
                int locOpenParen = l.startingLoc;
                l.getNextToken();
                Expr e = parseExpression(l);
                int locCloseParen = l.startingLoc;
                expect( l, ParserTagConstants.RPAREN );
                primary = ParenExpr.make( e, locOpenParen, locCloseParen );
                break;
      
            default:
                if (isPrimitiveKeywordTag(l.ttype)) {
                    Type t = parseType(l);
                    primary = parseClassLiteralSuffix(l, t);
                } else {
                    fail(l.startingLoc,
                         "Unexpected token '" + PrettyPrint.inst.toString(l.ttype) +
                         "' in Primary expression");
                    primary = null;       // dummy initialization
                }
        }
    
        // Ok, parsed a PrimaryCore expression into primary. 
        // Now handle PrimarySuffix*

        return parsePrimarySuffix( l, primary );
    }


    /**
     * parses '. class', then produces a class literal expression using
     * Type t.
     */ 
    //@ requires l != null && t != null && l.m_in != null;
    //@ requires t.syntax;
    //@ ensures \result != null;
    protected Expr parseClassLiteralSuffix(Lex l, Type t) {
        int locDot = l.startingLoc;
        expect( l, ParserTagConstants.FIELD );
        expect( l, ParserTagConstants.CLASS );

        return ClassLiteral.make(t, locDot);
    }


    //@ requires l != null && primary != null && l.m_in != null;
    //@ ensures \result != null;
    protected Expr parsePrimarySuffix(Lex l, Expr primary) {

        for(;;) {
            switch( l.ttype ) {
                case ParserTagConstants.INC: case ParserTagConstants.DEC:
                    primary = UnaryExpr.make(l.ttype == ParserTagConstants.INC 
                                             ? ParserTagConstants.POSTFIXINC 
                                             : ParserTagConstants.POSTFIXDEC, 
                                             primary, l.startingLoc );
                    l.getNextToken();
                    break;
        
                case ParserTagConstants.LSQBRACKET: {
                    if (l.lookahead(1) == ParserTagConstants.RSQBRACKET
                        || l.lookahead(1) == ParserTagConstants.STAR) 
                        return primary;
                    int locOpenBracket = l.startingLoc;
                    l.getNextToken();
                    Expr ndx = parseExpression(l);
                    primary = ArrayRefExpr.make(primary, ndx, 
                                                locOpenBracket, l.startingLoc);
                    expect( l, ParserTagConstants.RSQBRACKET );
                    break;
                }
        
                case ParserTagConstants.FIELD: {
                    int locDot = l.startingLoc;
                    if( l.lookahead(1) == ParserTagConstants.SUPER )
                        return primary;
                    l.getNextToken();
                    if( l.ttype == ParserTagConstants.NEW ) {
                        int locNew = l.startingLoc;
                        Expr tmp = parseNewExpression(l);
                        if( tmp.getTag() != ParserTagConstants.NEWINSTANCEEXPR) {
                            fail(locNew, "Cannot qualify an array allocation.\n");
                        }
                        NewInstanceExpr result = (NewInstanceExpr)tmp;
                        result.enclosingInstance = primary;
                        result.locDot = locDot;
                        if( result.type.name.size() != 1 ) {
                            fail(result.type.getStartLoc(),
                                 "Must have simple type name in a qualified new expression.\n");
                        }
                        primary = result;
                    } else {
                        int locId = l.startingLoc;
                        ObjectDesignator od = ExprObjectDesignator.make( locDot, primary );
                        if( l.ttype == ParserTagConstants.CLASS )
                            fail(l.startingLoc, ".class must follow a type");
                        Identifier id = parseIdentifier(l);
                        // identifier may be followed by TypeModifierPragmas
                        if( l.ttype == ParserTagConstants.LPAREN ||
                            l.ttype == ParserTagConstants.TYPEMODIFIERPRAGMA) {
                            TypeModifierPragmaVec tmodifiers = null;
                            if (l.ttype == ParserTagConstants.TYPEMODIFIERPRAGMA)	{
                                tmodifiers = parseTypeModifierPragmas(l);
                            } 
                            // is method invocation
                            // PrimaryExpression . Identifier ( ArgumentListopt )
                            int locOpenParen = l.startingLoc;
                            ExprVec args = parseArgumentList(l);
                            primary= MethodInvocation.make(od, id, tmodifiers,
                                                           locId, locOpenParen, args);
                        } else {
                            // is field access
                            // PrimaryExpression  . Identifier 
                            primary = FieldAccess.make( od, id, locId );
                        }
                    }
                    break;
                }
        
                default:
                    // Have parsed Primary, and there is no valid PrimarySuffix
                    // So just return current primary
                    return primary;
            }
        }                       // End loop over PrimarySuffix
    }
  
    /** Parse a <TT>NewExpression</TT>.
     NewExpression subsumes ClassInstanceCreationExpression and
     ArrayCreationExpression.
        
     <PRE>
     NewExpression:
     new TypeName ArgumentList [ TypeDeclBody ]
     new PrimitiveTypeOrTypeName DimExpr+ BracketPairs*

     DimExpr:
     LSQBRACKET Expression RSQBRACKET
     </PRE>
     */
  
    //@ requires l != null && l.m_in != null;
    //@ requires l.ttype == TagConstants.NEW;
    //@ ensures \result != null;
    public Expr parseNewExpression(Lex l) {
        int locNew = l.startingLoc;
        l.getNextToken();
    
        //alx: dw parse modifiers in new-expression
        int[] localUniverseArray=null;
        if (useUniverses) {
        	parseUniverses(l);
        	localUniverseArray = (int[]) this.universeArray.clone();
        }
	//alx-end

        // Next is Name or PrimitiveType
        Type type = parsePrimitiveTypeOrTypeName(l);

        //alx: dw set the universe modifiers
	Expr e = parseNewExpressionTail(l,type,locNew);
        if (useUniverses)
	    if (e.getTag()==ParserTagConstants.NEWINSTANCEEXPR) {
		setUniverse(e,localUniverseArray,type,locNew);
	    }
	    else if (e.getTag()==ParserTagConstants.NEWARRAYEXPR) {
		setUniverse(e,localUniverseArray,
			    ArrayType.make(type,locNew),locNew);
	    }

        return e;
	//alx-end
    }
    public Expr parseNewExpressionTail(Lex l, Type type, int locNew) {
        switch( l.ttype ) {
      
            case ParserTagConstants.LSQBRACKET:
                int[] openBrackets = new int[4];
                int cOB = 0;
                seqExpr.push();
                Type typeNew = null; // FIXME - JML needs this initializatio nbut Java does not
                ExprVec dims;
                ArrayInit init = null; // FIXME - JML needs this, javac does not

                if (l.lookahead(1) != ParserTagConstants.RSQBRACKET) {
                    // Parsing 'new' NonArrayType DimExprs Dims_opt
                    do {
                        // Should be LSQBRACKET Expression RSQBRACKET
                        if (cOB == openBrackets.length) {
                            int[] newOB = new int[2*cOB];
                            System.arraycopy(openBrackets, 0, newOB, 0, cOB);
                            openBrackets = newOB;
                        }
                        openBrackets[cOB++] = l.startingLoc;
                        l.getNextToken();
                        seqExpr.addElement( parseExpression(l) );
                        expect( l, ParserTagConstants.RSQBRACKET );
                    } while(l.ttype == ParserTagConstants.LSQBRACKET 
                            && l.lookahead(1) != ParserTagConstants.RSQBRACKET );
                    typeNew = parseBracketPairs(l, type);
                    if (l.ttype == ParserTagConstants.LSQBRACKET) {
                        fail(locNew, "Cannot index into a new array expression.");
                    } else if (l.ttype == ParserTagConstants.LBRACE) {
                        fail(locNew, "Cannot provide both explicit dimensions and array initializer.");
                    }
                    init = null;
                } else {
                    // Parsing 'new' NonArrayType Dims '{' ... '}'
                    openBrackets[cOB++] = l.startingLoc;
                    expect(l, ParserTagConstants.LSQBRACKET);
                    expect(l, ParserTagConstants.RSQBRACKET);
                    typeNew = parseBracketPairs(l, type);
                    if (l.ttype != ParserTagConstants.LBRACE) {
                        fail(locNew, "Must provide either explicit dimensions or array initializer.");
                    }
                    init = parseArrayInitializer(l);
                    seqExpr.addElement(LiteralExpr.make(ParserTagConstants.INTLIT,
                                                        new Integer(init.elems.size()),
                                                        init.locOpenBrace));
                }
                dims = ExprVec.popFromStackVector(seqExpr);
                Assert.notFalse(dims.size() == cOB);
                return NewArrayExpr.make( typeNew, dims, init, locNew, openBrackets );
      
            case ParserTagConstants.LPAREN:
                // ClassInstanceCreationExpression
                // type cannot be a PrimitiveType, must be TypeName
                if( type instanceof TypeName ) {
                    int locOpenParen = l.startingLoc;
                    ExprVec args = parseArgumentList(l);
                    ClassDecl cd = null;
                    if( l.ttype == ParserTagConstants.LBRACE ) {
                        int loc = l.startingLoc;
                        /*
                         * Warning: Parse.parseTypeDeclElemIntoSeq depends on
                         * anonymous classes id's starting with "$anon_"; if you
                         * update this code, make sure you change that routine in
                         * sync!
                         */
                        Identifier id
                            = Identifier.intern("$anon_" + Location.toLineNumber(loc));
    
                        expect( l, ParserTagConstants.LBRACE );
    
                        /* Build up Vec of TypeDeclElems in class or interface */
                        seqTypeDeclElem.push();
                        while( l.ttype != ParserTagConstants.RBRACE ) 
                            parseTypeDeclElemIntoSeqTDE( l, ParserTagConstants.CLASS, id, false );
                        TypeDeclElemVec elems
                            = TypeDeclElemVec.popFromStackVector( seqTypeDeclElem );
                        // Note: Anonymous classes do not have default constructors!

                        int locCloseBrace = l.startingLoc;
                        expect( l, ParserTagConstants.RBRACE );

                        cd = ClassDecl.make(Modifiers.NONE, null, id,
                                            TypeNameVec.make(), null, elems,
                                            loc, loc, loc, locCloseBrace,
                                            null);
                    }
                    return NewInstanceExpr.make( null, Location.NULL,
                                                 (TypeName)type, args, cd,
                                                 locNew, locOpenParen );
                } 
                // fall thru
      
            default:
                fail(l.startingLoc, "Bad 'new' expression");
                return null;    // Dummy return 
        }
    }

    /** Parse an <TT>ArgumentList</TT>, which includes enclosing parens.
     <PRE>
     ArgumentList:
     LPAREN [ Expression (, Expression)* ] RPAREN
     </PRE>
     */
  
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public ExprVec parseArgumentList(Lex l) {
        expect( l, ParserTagConstants.LPAREN );
        return parseExpressionList(l, ParserTagConstants.RPAREN);
    }
    /*
     if( l.ttype == TagConstants.RPAREN ) {
     l.getNextToken();
     return ExprVec.make();
     } else {
     seqExpr.push();
     for(;;) {
     seqExpr.addElement( parseExpression(l) );
     if( l.ttype == TagConstants.RPAREN ) {
     l.getNextToken();
     return ExprVec.popFromStackVector( seqExpr );
     }
     expect( l, TagConstants.COMMA );
     }
     }
     }
     */
    /** Parse an <TT>ExpressionList</TT>. Consumes specified terminator.
     <PRE>
     ExpressionList:
     [ Expression (, Expression)* ]
     </PRE>
     */
  
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public ExprVec parseExpressionList(Lex l, int terminator) {
        if( l.ttype == terminator ) {
            l.getNextToken();
            return ExprVec.make();
        } else {
            seqExpr.push();
            for(;;) {
                seqExpr.addElement( parseExpression(l) );
                if( l.ttype == terminator ) {
                    l.getNextToken();
                    return ExprVec.popFromStackVector( seqExpr );
                }
                expect( l, ParserTagConstants.COMMA );
            }
        }
    }

    /** Parse <TT>super</TT>.
     */

    //@ requires l != null && l.m_in != null;
    public Name parseSuper(Lex l) {
        expect( l, ParserTagConstants.SUPER );
        return SimpleName.make(Identifier.intern("super"), l.startingLoc);
    }

    /** Parse VariableInitializer.
     <PRE>
     VariableInitializer:
     Expression
     ArrayInitializer

     ArrayInitializer:
     { [ VariableInitializer ( , VariableInitializer )*] ,opt }
     </PRE>
     */

    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public VarInit parseVariableInitializer(Lex l, boolean specOnly) {
        if( l.ttype == ParserTagConstants.LBRACE ) {
            return parseArrayInitializer(l);
        } else {
            // Just a regular expression
            return parseExpression(l);
        }
    }

    //@ requires l.ttype == TagConstants.LBRACE;
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public ArrayInit parseArrayInitializer(Lex l) {
        int locOpenBrace = l.startingLoc;
        expect( l, ParserTagConstants.LBRACE );
        seqVarInit.push();
      
        if( l.ttype == ParserTagConstants.COMMA ) {
            // Should be LBRACE COMMA RBRACE
            l.getNextToken();
        } else {
            while( l.ttype != ParserTagConstants.RBRACE ) {
          
                seqVarInit.addElement( parseVariableInitializer(l, false) );
          
                switch( l.ttype ) {
                    case ParserTagConstants.COMMA: 
                        l.getNextToken(); break;
                    case ParserTagConstants.RBRACE: 
                        break;
                    default:
                        fail(l.startingLoc, "Bad variable initializer");
                }
            }
        }
        int locCloseBrace = l.startingLoc;
        if( l.ttype != ParserTagConstants.RBRACE ) 
            fail(l.startingLoc, "Bad variable initializer");
        l.getNextToken();

        return ArrayInit.make( VarInitVec.popFromStackVector( seqVarInit ),
                               locOpenBrace, locCloseBrace );
    }

    // ----------------------------------------------------------------------

    public ParseExpr() {
        //@ set seqExpr.elementType = \type(Expr);
        //@ set seqExpr.owner = this;

        //@ set seqVarInit.elementType = \type(VarInit);
        //@ set seqVarInit.owner = this;

        //@ set seqTypeDeclElem.elementType = \type(TypeDeclElem);
        //@ set seqTypeDeclElem.owner = this;

        // initialize the operator precedence table

        addOperator( ParserTagConstants.STAR, 170, true );
        addOperator( ParserTagConstants.DIV,  170, true );
        addOperator( ParserTagConstants.MOD,  170, true );

        addOperator( ParserTagConstants.ADD,  160, true );
        addOperator( ParserTagConstants.SUB,  160, true );

        addOperator( ParserTagConstants.LSHIFT, 150, true );
        addOperator( ParserTagConstants.RSHIFT, 150, true );
        addOperator( ParserTagConstants.URSHIFT, 150, true );

        addOperator( ParserTagConstants.GE, 140, true );
        addOperator( ParserTagConstants.GT, 140, true );
        addOperator( ParserTagConstants.LT, 140, true );
        addOperator( ParserTagConstants.LE, 140, true );

        // handled specially:
        addOperator( ParserTagConstants.INSTANCEOF, 140, true );


        addOperator( ParserTagConstants.EQ, 130, true );
        addOperator( ParserTagConstants.NE, 130, true );

        addOperator( ParserTagConstants.BITAND, 120, true );

        addOperator( ParserTagConstants.BITXOR, 110, true );

        addOperator( ParserTagConstants.BITOR, 100, true );

        addOperator( ParserTagConstants.AND, 90, true );

        addOperator( ParserTagConstants.OR, 80, true );

        addOperator( ParserTagConstants.QUESTIONMARK, 70, true );

        addOperator( ParserTagConstants.ASSIGN, 60, false );
        addOperator( ParserTagConstants.ASGMUL, 60, false );
        addOperator( ParserTagConstants.ASGDIV, 60, false );
        addOperator( ParserTagConstants.ASGREM, 60, false );
        addOperator( ParserTagConstants.ASGADD, 60, false );
        addOperator( ParserTagConstants.ASGSUB, 60, false );
        addOperator( ParserTagConstants.ASGLSHIFT, 60, false );
        addOperator( ParserTagConstants.ASGRSHIFT, 60, false );
        addOperator( ParserTagConstants.ASGURSHIFT, 60, false );
        addOperator( ParserTagConstants.ASGBITAND, 60, false );
        addOperator( ParserTagConstants.ASGBITOR, 60, false );
        addOperator( ParserTagConstants.ASGBITXOR, 60, false );
    }


    /**
     * Internal working storage for parseNewExpression,
     * parseExpressionList, and ParseStmt.parseForStmt functions.
     */
    //@ invariant seqExpr.elementType == \type(Expr);
    //@ invariant seqExpr.owner == this;
    protected final /*@ non_null @*/ StackVector seqExpr
            = new StackVector();

    /**
     * Internal working storage for parseArrayInitializer function.
     */
    //@ invariant seqVarInit.elementType == \type(VarInit);
    //@ invariant seqVarInit.owner == this;
    protected final /*@ non_null @*/ StackVector seqVarInit
            = new StackVector();

    /**
     * Internal working storage for parseNewExpression function.
     */
    //@ invariant seqTypeDeclElem.elementType == \type(TypeDeclElem);
    //@ invariant seqTypeDeclElem.owner == this;
    protected final /*@ non_null @*/ StackVector seqTypeDeclElem
            = new StackVector();
}
