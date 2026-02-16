/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

/** Responsible for converting GCExpr to formula Strings for input to
  * Simplify. The GCEXpr language is documented elsewhere, as is
  * Simplifys input language.
  */

import java.io.*;
import java.util.Vector;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Arrays;
import javafe.ast.*;
import javafe.tc.Types;
import javafe.util.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.backpred.BackPred;
import escjava.prover.Atom;

public class VcToString {

  /** Resets any type-specific information that is accumulated through
    * calls to <code>computeTypeSpecific</code>.
    **/

  public static void resetTypeSpecific() {
    integralPrintNames = new Hashtable();
    //@ set integralPrintNames.keyType = \type(Long);
    //@ set integralPrintNames.elementType = \type(String);
    // add thresholds
    integralPrintNames.put(minThreshold, String.valueOf(-MaxIntegral));
    integralPrintNames.put(maxThreshold, String.valueOf(MaxIntegral));
  }

  /** Prints <code>e</code> as a simple-expression string to <code>to</code>.
    * Any symbolic names created for integral literals in <code>e</code> are
    * added to a static place (variable <code>integralPrintNames</code>)
    * so that successive calls to <code>compute</code> can produce properties
    * about these names.
    **/

  public static void computeTypeSpecific(/*@ non_null */ Expr e,
					 /*@ non_null */ PrintStream to) {
    VcToString vts = new VcToString();
    vts.printFormula(to, e);
  }

  /** Prints <code>e</code> as a verification-condition string to <code>to</code>.
    * Any symbolic names of integral literals stored by the most recent call
    * to <code>computeTypeBackPred</code>, if any, will be considered when
    * producing properties about such symbolic literals.
    **/

  public static void compute(/*@ non_null */ Expr e,
			     /*@ non_null */ PrintStream to) {
    Hashtable oldNames = integralPrintNames;
    integralPrintNames = (Hashtable)oldNames.clone();
    
    VcToString vts = new VcToString();
    vts.printDefpreds(to, vts.getDefpreds(e));
    to.println("\n(EXPLIES ");
    vts.printFormula(to, e);
    to.println(" (AND ");
    vts.distinctSymbols(to);
    vts.stringLiteralAssumptions(to);
    vts.integralPrintNameOrder(to);
    to.println("))");

    integralPrintNames = oldNames;
  }

  public static void computePC(/*@ non_null */ Expr e,
			     /*@ non_null */ PrintStream to) {
    Hashtable oldNames = integralPrintNames;
    integralPrintNames = (Hashtable)oldNames.clone();
    
    VcToString vts = new VcToString();
    to.println("\n(AND ");
    vts.printFormula(to, e);
    vts.distinctSymbols(to);
    vts.stringLiteralAssumptions(to);
    vts.integralPrintNameOrder(to);
    to.println(")");

    integralPrintNames = oldNames;
  }

  // holds set of symbols used
  private /*@ non_null */ Set symbols = new Set();
  
  // string of initial assumptions
  private /*@ non_null */ Set stringLiterals = new Set();

  //@ invariant integralPrintNames.keyType == \type(Long);
  //@ invariant integralPrintNames.elementType == \type(String);
  private static /*@ non_null */ Hashtable integralPrintNames;

  private VcToString() {
  }

  private String vc2Term(Expr e, Hashtable subst) {
    Assert.notNull( e );
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos);
    printTerm( ps, subst, e );
    String s = baos.toString();
    ps.close();
    // System.out.println("vc2Term yields: "+s);
    return s;
  }

  private void printDefpreds(/*@ non_null */ PrintStream to, DefPredVec preds) {
      for( int i=0; i<preds.size(); i++) {
	  DefPred dp = preds.elementAt(i);
	  to.print("(DEFPRED ("+dp.predId);
	  for( int j=0; j<dp.args.size(); j++) {
	      to.print( " "+ dp.args.elementAt(j).id );
	  }
	  to.print(") ");
	  printFormula(to, dp.body);
	  to.print(")\n");
      }
  }

    private DefPredVec preds;

  private DefPredVec getDefpreds(Expr e) {
      preds = DefPredVec.make();
      getDefpredsHelper(e);
      return preds;
  }

  private void getDefpredsHelper(Expr e) {
      if( e instanceof DefPredLetExpr ) {
	  DefPredLetExpr dple = (DefPredLetExpr)e;
	  preds.append( dple.preds );
      }
      for(int i=0; i< e.childCount(); i++ ) {
	  Object c = e.childAt(i);
	  if( c instanceof Expr ) {
	      getDefpredsHelper( (Expr)c );
	  }
      }
  }

  private void distinctSymbols(/*@ non_null */ PrintStream to) {
    to.print("(DISTINCT");
    Enumeration e = symbols.elements();
    while( e.hasMoreElements() ) {
      String s = (String)e.nextElement();
      to.print(" ");
      to.print(s);
    }
    to.print(")");
  }

  private void stringLiteralAssumptions(/*@ non_null */ PrintStream to) {
    Enumeration e = stringLiterals.elements();
    while (e.hasMoreElements()) {
      String litname = (String)e.nextElement();

      to.print(" (NEQ ");
      to.print(litname);
      to.print(" null)");

      to.print(" (EQ (typeof ");
      to.print(litname);
      to.print(") |T_java.lang.String|)");

      // We could also assume "litname" to be allocated (but at this
      // time we don't have the name of the initial value of "alloc";
      // probably it is "alloc", but it would be nice not to have to
      // rely on that here).
    }
  }
  
  // ======================================================================

  private void printFormula(PrintStream out, Expr e) {

    // maps GenericVarDecls to Strings
    // some complex invariant here
	
    Hashtable subst = new Hashtable();
    printFormula( out, subst, e );
  }
      
  private void printFormula(PrintStream out, Hashtable subst, Expr e) {

    Assert.notNull( e );

    reallyPrintFormula( out, subst, e );
  }

  private void reallyPrintFormula(PrintStream out, Hashtable subst, Expr e) {

    // System.out.print("printFormula: ");
    // PrettyPrint.inst.print(System.out, e);
    // System.out.println("\nsubst="+subst);

    switch( e.getTag() ) {

    case TagConstants.DEFPREDLETEXPR:
	{
	  DefPredLetExpr dple = (DefPredLetExpr)e;
	  printFormula(out, subst, dple.body );
	  break;
	}
	  

    case TagConstants.SUBSTEXPR:
      {
	SubstExpr se = (SubstExpr)e;
	// perform current substitution on expression
	String expr = vc2Term(se.val, subst);
	// get old val, install new val
	Object old = subst.put( se.var, expr );
	//System.out.println("old="+old);

	// print
	printFormula(out, subst, se.target);

	// restore old state
	if( old == null )
	  subst.remove(se.var);
	else
	  subst.put(se.var, old);

	break;
      }

    case TagConstants.LABELEXPR:
      {
	LabelExpr le = (LabelExpr)e;
	out.print("(" + (le.positive? "LBLPOS":"LBLNEG")+ " |" +
		  le.label+"| ");
	printFormula( out, subst, le.expr);
	out.print(")");
	break;
      }

    case TagConstants.GUARDEXPR:
      {
        if (!escjava.Main.guardedVC) {
          Assert.fail("VcToString.reallyPrintFormula: unreachable");
        } else {
          GuardExpr ge = (GuardExpr)e;
          String var = escjava.Main.guardedVCPrefix + 
            UniqName.locToSuffix(ge.locPragmaDecl);
          out.print("(IMPLIES |" + var + "| ");
          printFormula( out, subst, ge.expr );
          out.print(")");
          escjava.Main.guardVars.add(var);
          break;
        }
      }

    case TagConstants.FORALL:
    case TagConstants.EXISTS:
      {
	QuantifiedExpr qe = (QuantifiedExpr)e;
	if( qe.quantifier==TagConstants.FORALL )
	  out.print("(FORALL (");
	else
	  out.print("(EXISTS (");

	String prefix = "";
	for(int i=0; i<qe.vars.size(); i++ ) {
	  GenericVarDecl decl = qe.vars.elementAt(i);
	  Assert.notFalse( !subst.containsKey(decl),
			   "Quantification over "
			   +"substituted variable");
	  out.print( prefix );
	  printVarDecl( out, decl );
	  prefix = " ";
	}
	out.print(") ");
	if (qe.nopats != null && qe.nopats.size() != 0) {
	  Assert.notFalse(!insideNoPats);
	  insideNoPats = true;
	  out.print("(NOPATS");
	  for (int i = 0; i < qe.nopats.size(); i++) {
	    out.print(" ");
	    Expr nopat = qe.nopats.elementAt(i);
	    printTerm(out, subst, nopat);
	  }
	  out.print(") ");
	  insideNoPats = false;
	}
	printFormula( out, subst, qe.expr);
	out.print(")");
	break;
      }

      //  Operators on formulas
    case TagConstants.BOOLIMPLIES:
    case TagConstants.BOOLAND:
    case TagConstants.BOOLOR:
    case TagConstants.BOOLNOT:
    case TagConstants.BOOLEQ:
      {
	NaryExpr ne = (NaryExpr)e;
	String op;

	switch( ne.getTag() ) {
	case TagConstants.BOOLIMPLIES:
	  op = "IMPLIES"; break;
	case TagConstants.BOOLAND:
	  op = "AND"; break;
	case TagConstants.BOOLOR:
	  op = "OR"; break;
	case TagConstants.BOOLNOT:
	  op = "NOT"; break;
	case TagConstants.BOOLEQ:
	  op = "IFF"; break;
	default:
	  Assert.fail("Fall thru");
	  op = null; // dummy assignment
	}

	out.print("("+op);
	for(int i=0; i< ne.exprs.size(); i++) {
	  out.print(" ");
	  printFormula( out, subst, ne.exprs.elementAt(i));
	}
	out.print(")");
	break;
      }

    case TagConstants.BOOLNE:
      {
	NaryExpr ne = (NaryExpr)e;
	out.print("(IFF ");
	printFormula( out, subst, ne.exprs.elementAt(0));
	out.print(" (NOT ");
	printFormula( out, subst, ne.exprs.elementAt(1));
	out.print("))");
	break;
      }

      // PredSyms
    case TagConstants.ALLOCLT:
    case TagConstants.ALLOCLE:
    case TagConstants.ANYEQ:
    case TagConstants.ANYNE:
    case TagConstants.INTEGRALEQ:
    case TagConstants.INTEGRALGE:
    case TagConstants.INTEGRALGT:
    case TagConstants.INTEGRALLE:
    case TagConstants.INTEGRALLT:
    case TagConstants.INTEGRALNE:
    case TagConstants.LOCKLE:
    case TagConstants.LOCKLT:
    case TagConstants.REFEQ:
    case TagConstants.REFNE:
    case TagConstants.TYPEEQ:
    case TagConstants.TYPENE:
    case TagConstants.TYPELE:
      {
	NaryExpr ne = (NaryExpr)e;
	String op;

	switch( ne.getTag() ) {
	case TagConstants.ALLOCLT:
	  op = "<"; break;
	case TagConstants.ALLOCLE:
	  op = "<="; break;
	case TagConstants.ANYEQ:
	  op = "EQ"; break;
	case TagConstants.ANYNE:
	  op = "NEQ"; break;
	case TagConstants.INTEGRALEQ:
	  op = "EQ"; break;
	case TagConstants.INTEGRALGE:
	  op = ">="; break;
	case TagConstants.INTEGRALGT:
	  op = ">"; break;
	case TagConstants.INTEGRALLE:
	  op = "<="; break;
	case TagConstants.INTEGRALLT:
	  op = "<"; break;
	case TagConstants.INTEGRALNE:
	  op = "NEQ"; break;
	case TagConstants.LOCKLE:
	  op = TagConstants.toVcString(TagConstants.LOCKLE);
	  break;
	case TagConstants.LOCKLT:
	  op = TagConstants.toVcString(TagConstants.LOCKLT);
	  break;
	case TagConstants.REFEQ:
	  op = "EQ"; break;
	case TagConstants.REFNE:
	  op = "NEQ"; break;
	case TagConstants.TYPEEQ:
	  op = "EQ"; break;
	case TagConstants.TYPENE:
	  op = "NEQ"; break;
	case TagConstants.TYPELE:
	  op = "<:"; break;
	default:
	  Assert.fail("Fall thru");
	  op = null; // dummy assignment
	}

	out.print("("+op);
	for(int i=0; i< ne.exprs.size(); i++) {
	  out.print(" ");
	  printTerm( out, subst, ne.exprs.elementAt(i));
	}
	out.print(")");
	break;
      }

    default:
      {
	if (e.getTag() == TagConstants.DTTFSA) {
	  NaryExpr ne = (NaryExpr)e;
	  TypeExpr te = (TypeExpr)ne.exprs.elementAt(0);
	  if (Types.isBooleanType(te.type)) {
	    // print this DTTFSA as a predicate
	    printDttfsa(out, subst, ne);
	    break;
	  }
	}
	// not a predicate, must be a term
	out.print("(EQ |@true| ");
	printTerm( out, subst, e);
	out.print(")");
	break;
      }
    }
  }

  // ======================================================================

  /** <code>insideNoPats</code> is really a parameter to
   ** <code>printTerm</code>.
   **/

  private boolean insideNoPats = false;

  private void printTerm(PrintStream out, Hashtable subst, Expr e) {

    // System.out.print("printTerm: ");
    // PrettyPrint.inst.print(System.out, e);
    // bSystem.out.println();

    int tag = e.getTag();
    switch( tag ) {

      case TagConstants.SUBSTEXPR:
	{
	  SubstExpr se = (SubstExpr)e;
	  // perform current substitution on expression
	  String expr = vc2Term(se.val, subst);
	  // get old val, install new val
	  Object old = subst.put( se.var, expr );
	  // print
	  printTerm(out, subst, se.target);

	  // restore old state
	  if( old == null )
	    subst.remove(se.var);
	  else
	    subst.put(se.var, old);

	  break;
	}

      case TagConstants.DEFPREDAPPLEXPR:
	{
	  DefPredApplExpr dpae = (DefPredApplExpr)e;
	  out.print("("+dpae.predId);
	  for(int i=0; i<dpae.args.size(); i++ ) {
	      out.print(" ");
	      printTerm(out, subst, dpae.args.elementAt(i));
	  }
	  out.print(")");
	  break;
	}

      case TagConstants.TYPEEXPR:
	{
	  TypeExpr te = (TypeExpr)e;
	  out.print(BackPred.simplifyTypeName(te.type));
	  break;
	}

	// Literals

      // This handles case 8 of ESCJ 23b:
      case TagConstants.STRINGLIT:
	{
	  LiteralExpr le = (LiteralExpr)e;
	  String s = "S_" + UniqName.locToSuffix(le.loc);
	  s = Atom.printableVersion(s);
	  stringLiterals.add(s);
	  out.print(s);
	  break;
	}

      case TagConstants.BOOLEANLIT:
	{
	  LiteralExpr le = (LiteralExpr)e;
	  if( ((Boolean)le.value).booleanValue() )
	    out.print("|@true|");
	  else
	    out.print("|bool$false|");
	  break;
	}

      case TagConstants.CHARLIT:
      case TagConstants.INTLIT:
	{
	  LiteralExpr le = (LiteralExpr)e;
	  out.print( integralPrintName(((Integer)le.value).intValue()) );
	  break;
	}

      case TagConstants.LONGLIT:
	{
	  LiteralExpr le = (LiteralExpr)e;
	  out.print( integralPrintName(((Long)le.value).longValue()) );
	  break;
	}

      case TagConstants.FLOATLIT:
	{
	  LiteralExpr le = (LiteralExpr)e;
	  out.print( "|F_" + ((Float)le.value).floatValue() + "|" );
	  break;
	}

      case TagConstants.DOUBLELIT: 
	{
	  LiteralExpr le = (LiteralExpr)e;
	  out.print( "|F_" + ((Double)le.value).doubleValue() + "|" );
	  break;
	}

      case TagConstants.NULLLIT:
	out.print("null");
	break;

      case TagConstants.SYMBOLLIT:
	{
	  // Handles case 5 of ESCJ 23b:
	  LiteralExpr le = (LiteralExpr)e;
	  String s = "|"+(String)le.value+"|";
	  symbols.add(s);
	  out.print(s);
	  break;
	}

      case TagConstants.VARIABLEACCESS: 
	{
	  VariableAccess va = (VariableAccess)e;
	  String to = (String)subst.get( va.decl );
	  if( to != null )
	    out.print(to);
	  else
	    printVarDecl( out, va.decl );
	  break;
	}

	// Nary functions
      case TagConstants.ALLOCLT:
      case TagConstants.ALLOCLE:
      case TagConstants.ARRAYLENGTH:
      case TagConstants.ARRAYFRESH:
      case TagConstants.ARRAYSHAPEMORE:
      case TagConstants.ARRAYSHAPEONE:
      case TagConstants.ASELEMS:
      case TagConstants.ASFIELD:
      case TagConstants.ASLOCKSET:
      case TagConstants.BOOLAND:
      case TagConstants.BOOLEQ:
      case TagConstants.BOOLIMPLIES:
      case TagConstants.BOOLNE:
      case TagConstants.BOOLNOT:
      case TagConstants.BOOLOR:
      case TagConstants.CLASSLITERALFUNC:
      case TagConstants.CONDITIONAL:
      case TagConstants.ELEMSNONNULL:
      case TagConstants.ELEMTYPE:
      case TagConstants.ECLOSEDTIME:
      case TagConstants.FCLOSEDTIME:

      case TagConstants.FLOATINGADD:
      case TagConstants.FLOATINGDIV:
      case TagConstants.FLOATINGEQ:
      case TagConstants.FLOATINGGE:
      case TagConstants.FLOATINGGT:
      case TagConstants.FLOATINGLE: 
      case TagConstants.FLOATINGLT: 
      case TagConstants.FLOATINGMOD:
      case TagConstants.FLOATINGMUL:
      case TagConstants.FLOATINGNE: 
      case TagConstants.FLOATINGNEG:
      case TagConstants.FLOATINGSUB:

      case TagConstants.INTEGRALADD:
      case TagConstants.INTEGRALAND:
      case TagConstants.INTEGRALDIV:
      case TagConstants.INTEGRALEQ:
      case TagConstants.INTEGRALGE:
      case TagConstants.INTEGRALGT:
      case TagConstants.INTEGRALLE:
      case TagConstants.INTEGRALLT:
      case TagConstants.INTEGRALMOD:
      case TagConstants.INTEGRALMUL:
      case TagConstants.INTEGRALNE:
      case TagConstants.INTEGRALNEG:
      case TagConstants.INTEGRALNOT:
      case TagConstants.INTEGRALOR:
      case TagConstants.INTSHIFTL:
      case TagConstants.LONGSHIFTL:
      case TagConstants.INTSHIFTR:
      case TagConstants.LONGSHIFTR:
      case TagConstants.INTSHIFTRU:
      case TagConstants.LONGSHIFTRU:
      case TagConstants.INTEGRALSUB:
      case TagConstants.INTEGRALXOR:
	    
      case TagConstants.IS:
      case TagConstants.ISALLOCATED:
      case TagConstants.ISNEWARRAY:
      case TagConstants.LOCKLE:
      case TagConstants.LOCKLT:
      case TagConstants.MAX:
      case TagConstants.REFEQ:
      case TagConstants.REFNE:
      case TagConstants.SELECT:
      case TagConstants.STORE:
      case TagConstants.STRINGCAT:
      case TagConstants.TYPEEQ:
      case TagConstants.TYPENE:
      case TagConstants.TYPELE:
      case TagConstants.TYPEOF:
      case TagConstants.VALLOCTIME:
	{
	  NaryExpr ne = (NaryExpr)e;
	  String op;
	  switch( tag ) {
	    case TagConstants.INTEGRALADD:
	      op = "+"; break;
	    case TagConstants.INTEGRALMUL:
	      op = "*"; break;
	    case TagConstants.INTEGRALNEG:
	      op = "- 0"; break;
	    case TagConstants.INTEGRALSUB:
	      op = "-"; break;
	    case TagConstants.TYPELE:
	      op = "<:"; break;
	    case TagConstants.INTEGRALNE:
	    case TagConstants.REFNE:
	    case TagConstants.TYPENE:
	    case TagConstants.ANYNE:
	      if (insideNoPats) {
		op = "NEQ";
		break;
	      }
	      // fall thru
	    default:
	      op = TagConstants.toVcString(tag);
	  }
	  out.print("("+op);
	  for(int i=0; i< ne.exprs.size(); i++) {
	    out.print(" ");
	    printTerm( out, subst, ne.exprs.elementAt(i));
	  }
	  out.print(")");
	  break;
	}

      case TagConstants.CAST:
	{
	  NaryExpr ne = (NaryExpr)e;
	  Assert.notFalse(ne.exprs.size() == 2);
	  Expr x = ne.exprs.elementAt(0);
	  TypeExpr t = (TypeExpr)ne.exprs.elementAt(1);

	  if (Types.isBooleanType(t.type)) {
	    // Peephole optimization to remove casts to boolean, since
	    // anything is a boolean in the underlying logic (it either
	    // equals |@true| or it doesn't).  The reason this peephole
	    // optimization is performed here, rather than in trExpr
	    // and trSpecExpr, is that then any expression cast to a
	    // boolean will be printed as a term, not as a predicate.
	    // This may sometimes be useful for boolean DTTFSA expression,
	    // which are printed as predicate whenever they occur in
	    // predicate positions.
	    printTerm(out, subst, x);
	  } else {
	    out.print("(");
	    out.print(TagConstants.toVcString(tag));
	    out.print(" ");
	    printTerm(out, subst, x);
	    out.print(" ");
	    printTerm(out, subst, t);
	    out.print(")");
	  }
	  break;
	}

      case TagConstants.DTTFSA:
	{
	  NaryExpr ne = (NaryExpr)e;
	  printDttfsa(out, subst, ne);
	  break;
	}

      default:
	Assert.fail("Bad tag in GCTerm: " +
		    "(tag="+tag+","+TagConstants.toVcString(tag)+") " + e);
    }
  }

  //@ requires ne.op == TagConstants.DTTFSA;
  private void printDttfsa(PrintStream out, Hashtable subst, NaryExpr ne) {
    LiteralExpr lit = (LiteralExpr)ne.exprs.elementAt(1);
    String op = (String)lit.value;
    if (ne.exprs.size() == 2) {
      out.print(op);
    } else if (op.equals("identity")) {
      Assert.notFalse(ne.exprs.size() == 3);
      Expr e2 = ne.exprs.elementAt(2);
      printTerm(out, subst, e2);
    } else {
      out.print("(");
      out.print(op);
      for(int i = 2; i < ne.exprs.size(); i++) {
	out.print(" ");
	Expr ei = ne.exprs.elementAt(i);
	printTerm(out, subst, ei);
      }
      out.print(")");
    }
  }

  // ======================================================================

  private void printVarDecl(PrintStream out, GenericVarDecl decl) {
    out.print(Atom.printableVersion(UniqName.variable(decl)));
  }

  // ======================================================================


    private static final long MaxIntegral = 1000000;
    private static final Long minThreshold = new Long(-MaxIntegral);
    private static final Long maxThreshold = new Long(MaxIntegral);

    /**
     ** Convert an integral # into its printname according to the rules
     ** of ESCJ 23b, part 9.
     **/

    private String integralPrintName(long n) {
      if (-MaxIntegral <= n && n <= MaxIntegral) {
	return String.valueOf(n);
      }

      Long l = new Long(n);
      String name = (String)integralPrintNames.get(l);
      if (name != null) {
	return name;
      }

      if (n == -n) {
	// Need to handle minlong specially...
	name = "neg" + String.valueOf(n).substring(1);
      } else if (n < 0) {
	name = "neg" + String.valueOf(-n);
      } else {
	name = "pos" + String.valueOf(n);
      }

      integralPrintNames.put(l, name);
      return name;
    }

  /** Generates the inequalities that compare the integral literals
    * that were replaced in <code>integralPrintName</code> by symbolic
    * names.
    **/
  
  private void integralPrintNameOrder(/*@ non_null */ PrintStream ps) {
    int n = integralPrintNames.size();
    Assert.notFalse(2 <= n);  // should contain the two thresholds
    if (n == 0) {
      return;
    }

    Long[] keys = new Long[n];
    Enumeration e = integralPrintNames.keys();
    for (int i = 0; e.hasMoreElements(); i++) {
      Long l = (Long)e.nextElement();
      keys[i] = l;
    }
    Arrays.sort(keys);

    String valueI = (String)integralPrintNames.get(keys[0]);
    /* loop invariant:  valueI == integralPrintNames.get(keys[i]); */
    for (int i = 0; i < n-1; i++) {
      String valueNext = (String)integralPrintNames.get(keys[i+1]);
      if (keys[i] == minThreshold) {
	Assert.notFalse(keys[i+1] == maxThreshold);
      } else {
	ps.print(" (< ");
	ps.print(valueI);
	ps.print(" ");
	ps.print(valueNext);
	ps.print(")");
      }
      valueI = valueNext;
    }
  }

  static {
    resetTypeSpecific();
  }
}
