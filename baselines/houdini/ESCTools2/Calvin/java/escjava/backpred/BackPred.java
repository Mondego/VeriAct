/* Copyright Hewlett-Packard, 2002 */

package escjava.backpred;

import java.util.*;
import java.io.*;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;

import escjava.Main;
import escjava.ast.TagConstants;
import escjava.ast.TypeExpr;
import escjava.ast.MapType;
import escjava.prover.Atom;
import escjava.translate.*;


/** Generates the background predicate for a given type. 

 * <p>The background predicate <I>for</I> a particular class or
 * interface type T consists of the conjunction of the background
 * predicates contributions of each contributor class. See
 * <code>FindContributors</code> for the definition of the
 * contributors of a class.

 * A type must be typechecked in order to generate its background
 * predicate.  A type need only be prepped in order to generate its
 * background predicate contribution.

*/

public class BackPred {

  /** Returns the universal background predicate as a 
    * sequence of simplify commands. */

  public static void genUnivBackPred(/*@ non_null */ PrintStream proverStream) {
    if (escjava.Main.univBackPredFile == null) {
      proverStream.print(DefaultUnivBackPred.s);
      return;
    }
    String filename = escjava.Main.univBackPredFile;
    try {
      FileInputStream fis = new FileInputStream(filename);
      int c;
      while( (c=fis.read()) != -1 ) proverStream.print((char)c);
      fis.close();
    } catch( IOException e) {
      ErrorSet.fatal("IOException: "+e);
    }
  }


  /**
   ** Return the type-specific background predicate as a formula.
   **/
  public static void genTypeBackPred(/*@ non_null */ FindContributors scope,
				     /*@ non_null */ PrintStream proverStream) {
    // set up the background predicate buffer
    proverStream.print("(AND ");

    // set up the distinct types axiom buffer
    StringBuffer dta =
	new StringBuffer("(DISTINCT arrayType |"
	    + UniqName.type(Types.booleanType) + "| |"
	    + UniqName.type(Types.charType) + "| |"
	    + UniqName.type(Types.byteType) + "| |"
	    + UniqName.type(Types.shortType) + "| |"
	    + UniqName.type(Types.intType) + "| |"
	    + UniqName.type(Types.longType) + "| |"
	    + UniqName.type(Types.floatType) + "| |"
	    + UniqName.type(Types.doubleType) + "| |"
	    + UniqName.type(escjava.tc.Types.typecodeType) + "|");


    // Print them out, and add their contribution to the BP. 
    Info.out("[TypeSig contributors for "
		+Types.printName(scope.originType)+":");
    for( Enumeration enum = scope.typeSigs();
	 enum.hasMoreElements(); )
      {
	TypeSig sig2 = (TypeSig)enum.nextElement();
	Info.out("    "+Types.printName( sig2 ));
	addContribution( sig2.getTypeDecl(), proverStream );
	dta.append(" "+simplifyTypeName( sig2 ) );
      }
    Info.out("]");


    // Add the distinct types axiom
    bg( dta.toString()+")", proverStream );


    // Handle constant fields' contribution:
    for( Enumeration enum = scope.fields();
	 enum.hasMoreElements(); ) {
	FieldDecl fd = (FieldDecl)enum.nextElement();
	if (!Modifiers.isFinal(fd.modifiers) || fd.init==null)
	    continue;

	int loc = fd.init.getStartLoc();
	VariableAccess f = VariableAccess.make(fd.id, loc, fd);

	if (Modifiers.isStatic(fd.modifiers)) {
	  genFinalInitInfo(fd.init, null, null, f, fd.type, loc, proverStream);
	} else {
	  LocalVarDecl sDecl = UniqName.newBoundVariable('s');
	  VariableAccess s = TrAnExpr.makeVarAccess(sDecl, Location.NULL);
	  genFinalInitInfo(fd.init, sDecl, s, GC.select(f, s), fd.type, loc,
			   proverStream);
	}
    }

    proverStream.print(")");
  }

  //@ requires loc != Location.NULL;
  //@ requires sDecl != null ==> s != null;
  private static void genFinalInitInfo(/*@ non_null */ VarInit initExpr,
				       GenericVarDecl sDecl, Expr s,
				       /*@ non_null */ Expr x,
				       /*@ non_null */ Type type,
				       int loc,
				       /*@ non_null */ PrintStream proverStream) {
    /* The dynamic type of the field is subtype of the static type of the
     * initializing expression.
     */
    {
      Type exprType = TypeCheck.inst.getType(initExpr);
      Expr tExpr = TypeExpr.make(initExpr.getStartLoc(),
				 initExpr.getEndLoc(),
				 exprType);
      produce(sDecl, s, GC.nary(TagConstants.IS, x, tExpr), proverStream);
    }

    /* Generate primitive type constant info */
    if (type instanceof PrimitiveType) {
      if (initExpr instanceof Expr) {
	Expr constant = eval((Expr)initExpr, loc);
	if (constant != null) {
	  produce(sDecl, s, eq(x, constant, type), proverStream);
	}
      }
      return;
    }

    /* Peel off parentheses and casts. */
    int tag;
    while (true) {
      tag = initExpr.getTag();

      if (tag == TagConstants.PARENEXPR) {
	initExpr = ((ParenExpr)initExpr).expr;
      } else if (tag == TagConstants.CASTEXPR) {
	initExpr = ((CastExpr)initExpr).expr;
      } else if (tag == TagConstants.NEWARRAYEXPR) {
	NewArrayExpr nae = (NewArrayExpr)initExpr;
	if (nae.init != null) {
	  initExpr = nae.init;
	  tag = initExpr.getTag();
	}
	break;
      } else {
	break;
      }
    }

    /* Generate null related info */
    if (isStaticallyNonNull(initExpr)) {
      produce(sDecl, s, GC.nary(TagConstants.REFNE, x, GC.nulllit), 
	      proverStream);
    } else if (initExpr.getTag() == TagConstants.NULLLIT) {
      produce(sDecl, s, GC.nary(TagConstants.REFEQ, x, GC.nulllit),
	      proverStream);
    }

    /* Generate new and array related info */
    if (tag == TagConstants.ARRAYINIT) {
      ArrayInit aInit = (ArrayInit)initExpr;
      
      // typeof(x) == array(T)
      Expr typeofx = GC.nary(TagConstants.TYPEOF, x);
      Expr arrayT = TypeExpr.make(aInit.getStartLoc(), aInit.getEndLoc(),
				  TypeCheck.inst.getType(aInit));
      produce(sDecl, s, GC.nary(TagConstants.TYPEEQ, typeofx, arrayT),
	      proverStream);
			
      // arrayLength(x) == len
      int len = aInit.elems.size();
      produce(sDecl, s, GC.nary(TagConstants.INTEGRALEQ,
				GC.nary(TagConstants.ARRAYLENGTH, x),
				LiteralExpr.make(TagConstants.INTLIT,
						 new Integer(len), loc)),
	      proverStream);

    } else if (tag == TagConstants.NEWARRAYEXPR) {
      NewArrayExpr nae = (NewArrayExpr)initExpr;
      Assert.notFalse(nae.dims.size() > 0);  // arrayinit is handled above
      // typeof(x) == array(...(array(T)))
      Expr typeofx = GC.nary(TagConstants.TYPEOF, x);
      Expr arrayT = TypeExpr.make(nae.getStartLoc(), nae.getEndLoc(),
				  TypeCheck.inst.getType(nae));
      produce(sDecl, s, GC.nary(TagConstants.TYPEEQ, typeofx, arrayT),
	      proverStream);

      LiteralExpr constant = eval(nae.dims.elementAt(0), loc);
      if (constant != null) {
	Assert.notFalse(constant.getTag() == TagConstants.INTLIT);
	if (0 <= ((Integer)constant.value).intValue()) {
	  // arrayLength(x) == constant
	  produce(sDecl, s, GC.nary(TagConstants.INTEGRALEQ,
				    GC.nary(TagConstants.ARRAYLENGTH, x),
				    constant),
		  proverStream);
	}
      }

    } else if (tag == TagConstants.NEWINSTANCEEXPR) {
      NewInstanceExpr ni = (NewInstanceExpr)initExpr;
      // typeof(x) == T
      Expr typeofx = GC.nary(TagConstants.TYPEOF, x);
      Expr T = TypeExpr.make(ni.getStartLoc(), ni.getEndLoc(), ni.type);
      produce(sDecl, s, GC.nary(TagConstants.TYPEEQ, typeofx, T),
	      proverStream);
    }
  }

  //@ requires sDecl != null ==> s != null;
  private static void produce(GenericVarDecl sDecl, Expr s,
			      /*@ non_null */ Expr e,
			      /*@ non_null */ PrintStream proverStream) {
    if (sDecl != null) {
      Expr ant = GC.nary(TagConstants.REFNE, s, GC.nulllit);
      ExprVec nopats = ExprVec.make(1);
      nopats.addElement(ant);
      e = GC.forall(sDecl, GC.implies(ant, e), nopats);
    }
    proverStream.print("\n");
    VcToString.computeTypeSpecific(e, proverStream);
  }
  
  /** Add to b the contribution from a particular TypeDecl, which is a
    * formula. */

  static void addContribution(/*@ non_null */ TypeDecl d,
			      /*@ non_null */ PrintStream proverStream) {

    TypeSig sig = TypeCheck.inst.getSig(d);

    // === ESCJ 8: Section 1.1

    if( d instanceof ClassDecl ) {
      ClassDecl cd = (ClassDecl)d;

      if( cd.superClass != null ) {
	saySubClass( sig, cd.superClass, proverStream );
      }

      if( Modifiers.isFinal(cd.modifiers) )
	sayIsFinal( sig, proverStream );

    } else {
      saySubType( sig, Types.javaLangObject(), proverStream );
    }
				      
    for( int i=0; i<d.superInterfaces.size(); i++ )
      saySubType( sig, d.superInterfaces.elementAt(i), proverStream );
  }


  /** Record in the background predicate that x is a subclass of y. */

  private static void saySubClass( Type x, Type y,
				   /*@ non_null */ PrintStream proverStream ) {

    saySubType( x, y, proverStream );
    bg("(EQ "+simplifyTypeName(x)+
       " (asChild "+simplifyTypeName(x)+" "+simplifyTypeName(y)+"))",
       proverStream);
  }

  /** Record in the background predicate that x is a subtype of y. */

  private static void saySubType( Type x, Type y,
				  /*@ non_null */ PrintStream proverStream ) {

    bg( "(<: "+simplifyTypeName(x)+" "+simplifyTypeName(y)+")" , proverStream);

  }

  /** Record in the background predicate that x is final. */

  private static void sayIsFinal( Type x,
				  /*@ non_null */ PrintStream proverStream ) {
      String n = simplifyTypeName(x);
      bg( "(FORALL (t) (PATS (<: t "+n+")) (IFF (<: t "+n+") (EQ t "+n+")))",
	  proverStream);
  }

  public static String simplifyTypeName(/*@ non_null */ Type x) {
    if (x instanceof ArrayType) {
      ArrayType at = (ArrayType)x;
      return "(array " + simplifyTypeName(at.elemType) + ")";
    } else if (x instanceof MapType) {
      MapType mt = (MapType)x;
      return "(map " + simplifyTypeName(mt.elemType) + ")";	
    } else {
      return Atom.printableVersion(UniqName.type(x));
    }
  }

  /** Called with an S-expression that may contain a free variable "t".
    * Wraps "(FORALL (s) (IMPLIES (NEQ s null) " and "))" around this
    * expression and adds it to the background predicate.
    **/

  private static void bgi(/*@ non_null */ String s,
			  /*@ non_null */ PrintStream proverStream) {
      proverStream.print("\n(FORALL (s) (IMPLIES (NEQ s null) ");
      proverStream.print(s);
      proverStream.print("))");
  }

  /** Called with a simplify command. Adds it to the background
    * predicate. */

  private static void bg(/*@ non_null */ String s,
			 /*@ non_null */ PrintStream proverStream) {
      proverStream.print('\n');
      proverStream.print(s);
  }


  // ======================================================================

    /**
     ** Do we know statically that an expression always returns a
     ** non-null value?
     **/
    private static boolean isStaticallyNonNull(VarInit e) {
	int tag = e.getTag();

	// New expressions are always non-null:
	if (tag==TagConstants.NEWARRAYEXPR ||
	    tag==TagConstants.NEWINSTANCEEXPR)
	    return true;

	// Array initializers are always non-null:
	if (tag==TagConstants.ARRAYINIT)
	    return true;

	// String constants can be non-null:
	if (tag==TagConstants.STRINGLIT) {
	    LiteralExpr asLit = (LiteralExpr)e;
	    if (asLit.value!=null)
		return true;
	}

	if (tag == TagConstants.ADD || tag == TagConstants.ASGADD) {
	  BinaryExpr be = (BinaryExpr)e;
	  Type leftType = TypeCheck.inst.getType( be.left );
	  Type rightType = TypeCheck.inst.getType( be.right );
	  if (Types.isReferenceOrNullType(leftType) ||
	      Types.isReferenceOrNullType(rightType)) {
	    // The "+" or "+=" operator is String catenation, which always
	    // returns a non-null value.
	    return true;
	  }
	}

	return false;
    }


    /**
     ** Like ConstantExpr.eval, but returns a LiteralExpr instead of an
     ** Integer/Long/etc. <p>
     **
     ** Ignores String constants.  (Always returns null in that case.)
     **
     ** If returns a non-null LiteralExpr, sets its loc to
     ** <code>loc</code>.<p>
     **/
    //@ requires e!=null && loc!=Location.NULL
    private static LiteralExpr eval(Expr e, int loc) {
	Object val = ConstantExpr.eval(e);

	if (val instanceof Boolean)
	    return LiteralExpr.make(TagConstants.BOOLEANLIT, val, loc);

	// char, byte, short, int cases:
	if (val instanceof Integer)
	    return LiteralExpr.make(TagConstants.INTLIT, val, loc);

	if (val instanceof Long)
	    return LiteralExpr.make(TagConstants.LONGLIT, val, loc);

	if (val instanceof Float)
	    return LiteralExpr.make(TagConstants.FLOATLIT, val, loc);
	if (val instanceof Double)
	    return LiteralExpr.make(TagConstants.DOUBLELIT, val, loc);

	// Ignore String because don't have the right location

	return null;
    }


    /**
     ** Generate the appropriate GC equality e1 == e2 based on type t.
     **/
    //@ requires e1!=null && e2!=null && t!=null;
    //@ ensures \result != null;
    private static Expr eq(Expr e1, Expr e2, Type t) {
	if (!(t instanceof PrimitiveType))
	    return GC.nary(TagConstants.REFEQ, e1, e2);

	switch (t.getTag()) {
	    case TagConstants.NULLTYPE:
		return GC.nary(TagConstants.REFEQ, e1, e2);

	    case TagConstants.BOOLEANTYPE:
		return GC.nary(TagConstants.BOOLEQ, e1, e2);

	    case TagConstants.CHARTYPE:
	    case TagConstants.BYTETYPE:
	    case TagConstants.SHORTTYPE:
	    case TagConstants.INTTYPE:
	    case TagConstants.LONGTYPE:
		return GC.nary(TagConstants.INTEGRALEQ, e1, e2);

	    case TagConstants.FLOATTYPE:
	    case TagConstants.DOUBLETYPE:
		return GC.nary(TagConstants.FLOATINGEQ, e1, e2);

	    case TagConstants.TYPECODE:
		return GC.nary(TagConstants.TYPEEQ, e1, e2);

	    default:
		Assert.fail("Bad primitive type passed to BackPred.eq:"
			    + TagConstants.toString(t.getTag()));
		return null;	// keep compiler happy...
	}
    }
}
