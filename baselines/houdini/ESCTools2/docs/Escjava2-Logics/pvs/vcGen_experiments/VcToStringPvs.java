package escjava.translate;

/**
 * Responsible for converting GCExpr to formula Strings for input to Simplify.
 * The GCExpr language is documented elsewhere, as is Simplifys input language.
 */

import java.io.*;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Arrays;
import java.util.Vector;
import java.util.Iterator;
import javafe.ast.*;
import javafe.tc.Types;
import javafe.util.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.backpred.BackPred;
import escjava.prover.Atom;

public class VcToStringPvs {

    /* list of already declared variables in the pvs logic 
       like LS, null, Java.lang.Object etc... */
    static public Vector listOfDecl = new Vector();

    /* name of the variables that need to be declared
       Before adding a variable to this set, we check if the variable isn't
       already in this set or in listOfDecl */
    static public Vector listOfDeclAdd = new Vector();

    /* set containing the name of the new fonctions that need to be declared,
       that are introduced during the creation of gc, but aren't explicitly redeclared
       here normally (because Simplify seems to allow declarations 'on the fly' ??) 

       The second vector contains the number of parameters for functions
       declared in the first one.

       It means :
       listOfDeclFun = < f, g>
       listOfDeclFunNbParam = < 1, 2>

       will be translated to :
       f(a : S) : S
       g(a, b :S) : S
    */
    static public Vector listOfDeclFun = new Vector();
    static public Vector listOfDeclFunNbParam = new Vector();
    
  /**
   * Resets any type-specific information that is accumulated through calls to
   * <code>computeTypeSpecific</code>.
   */
  public static void resetTypeSpecific() {
    integralPrintNames = new Hashtable();
    //-@ set integralPrintNames.keyType = \type(Long);
    //-@ set integralPrintNames.elementType = \type(String);
    // add thresholds
    integralPrintNames.put(minThreshold, String.valueOf(-MaxIntegral));
    integralPrintNames.put(maxThreshold, String.valueOf(MaxIntegral));
  }
  
  /**
   * Prints <code>e</code> as a simple-expression string to <code>to</code>.
   * Any symbolic names created for integral literals in <code>e</code> are
   * added to a static place (variable <code>integralPrintNames</code>) so
   * that successive calls to <code>compute</code> can produce properties
   * about these names.
   */
  public static void computeTypeSpecific(/*@  non_null */Expr e,
      /*@  non_null */PrintStream to) {
    VcToString vts = new VcToString();
    vts.printFormula(to, e);
  }
  
  /**
   * Prints <code>e</code> as a verification-condition string to
   * <code>to</code>. Any symbolic names of integral literals stored by the
   * most recent call to <code>computeTypeBackPred</code>, if any, will be
   * considered when producing properties about such symbolic literals.
   */
  public static void compute(/*@  non_null */Expr e,
      /*@  non_null */PrintStream to) {

      listOfDecl.add("elems");
      listOfDecl.add("alloc");
      listOfDecl.add("java_null");
      listOfDecl.add("T_double");
      listOfDecl.add("T_float");
      listOfDecl.add("T_long");
      listOfDecl.add("T_int");
      listOfDecl.add("T_short");
      listOfDecl.add("T_byte");
      listOfDecl.add("T_char");
      listOfDecl.add("T_boolean");
      listOfDecl.add("T_java_lang_Cloneable");
      listOfDecl.add("LS");

      Hashtable oldNames = integralPrintNames;
      integralPrintNames = (Hashtable)oldNames.clone();
    
      // tant qu'à faire, tant qu'à always pretty print!! 
      //if (escjava.Main.options().prettyPrintVC)
	  to = new PrintStream(new escjava.prover.PPOutputStream(to));
    
      VcToStringPvs vts = new VcToStringPvs();
      vts.printDefpreds(to, vts.getDefpreds(e));
      //to.println("\n(EXPLIES ");
      to.println("\n(EXPLIES(");
      vts.printFormula(to, e);
      //to.println(" (AND ");
      to.println(",true))");

      // /* Remove the distinct clause at the end of the output
// 	 it's replaced by declaring pvs variables
//        */

      //vts.distinctSymbols(to);
      vts.stringLiteralAssumptions(to);

      to.print("Start of pvs declarations :\n ");

      vts.integralPrintNameOrder(to);

      //to.println("))");
    
      integralPrintNames = oldNames;

      /* This piece of code declares all variables before leaving */
      Iterator i = null;
      String tempString = null; // makes the compiler happy
      Integer tempInteger = null;

      to.println("ecReturn : S");
      to.println("ecThrow : S;");

      /* was in the simplify logic */
      to.println("distinctAxiom : AXIOM\nrefEQ(ecReturn, ecThrow) = bool_false\n");

      /* list of functions declarations

      It means :
      listOfDeclFun = < f, g>
      listOfDeclFunNbParam = < 1, 2>
      
      will be translated to :
      f(a : S) : S
      g(a, b :S) : S
      
      */

      if(listOfDeclFun.size() != listOfDeclFunNbParam.size())
	  System.out.println("Warning, inconsistency in declaration of new functions...");

      if(listOfDeclFun.size()!=0) { /* something to declare */
	  i = listOfDeclFun.iterator();
	  Iterator j = listOfDeclFunNbParam.iterator();	  

	  while(i.hasNext() && j.hasNext()){

	      try {
		  tempString = (String)i.next();
		  tempInteger = (Integer)j.next();
	      }
	      catch(Exception ex){
		  System.out.println("VcToStringPvs::add2Decl *** error *** "+ex);
	      }

	      /* name of the function, for example : g */
	      to.print(tempString+"(");
	      char c = 'a';

	      /* declaring (a1, a2 :S) for g for example 
		 note that if there is more than 24 parameters, it will stupidly fail
		 because c++ will come back to a, anyway it should not happen
	      */
	      for( int k = 0; k < tempInteger.intValue(); k++, c++) { 
		  to.print(c);
		  to.print(" : S");
	      
		  if(k < tempInteger.intValue() - 1) /* not the last */
		      to.print(" ,");
		  else /* finishing declaration, adding ) : S at the end */
		      to.print(") : S\n"); 
		      //to.print(") : S = "+c +"\n"); // experimental
	      }
	  }

	  to.print(";\n");
      }

      /* theorem */
      i = listOfDeclAdd.iterator();

      to.println("testTheorem : THEOREM\nFORALL(");
      while(i.hasNext()){
	    
	  try{ tempString = (String)i.next();}
	  catch(Exception ex){
	      System.out.println("VcToStringPvs::add2Decl *** error *** "+ex);
	  }
	
	  to.print(tempString);

	  if(i.hasNext())
	      to.print(" , ");

      }

      to.print(" : S ) :\n");

  }
  
  public static void computePC(/*@  non_null */Expr e,
      /*@  non_null */PrintStream to) {
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
  //@ spec_public
  protected/*@  non_null */Set symbols = new Set();
  
  // string of initial assumptions
  //@ spec_public
  protected/*@  non_null */Set stringLiterals = new Set();
  
  //-@ invariant integralPrintNames.keyType == \type(Long);
  //-@ invariant integralPrintNames.elementType == \type(String);
  //@ spec_public
  protected static/*@  non_null */Hashtable integralPrintNames;
  
    protected VcToStringPvs() {}
  
  protected String vc2Term(/*@ non_null */Expr e, Hashtable subst) {
    Assert.notNull(e);
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos);
    printTerm(ps, subst, e);
    String s = baos.toString();
    ps.close();
    // System.out.println("vc2Term yields: "+s);

    return s;
  }
  
  protected void printDefpreds(/*@  non_null */PrintStream to, DefPredVec preds) {
    for (int i = 0; i < preds.size(); i++) {
      DefPred dp = preds.elementAt(i);
      to.print("(DEFPRED (" + dp.predId);
      for (int j = 0; j < dp.args.size(); j++) {
        to.print(" " + dp.args.elementAt(j).id);
      }
      to.print(") ");
      printFormula(to, dp.body);
      to.print(")\n");
    }
  }
  
  protected DefPredVec preds;
  
  protected DefPredVec getDefpreds(/*@ non_null */Expr e) {
    preds = DefPredVec.make();
    getDefpredsHelper(e);
    return preds;
  }
  
  protected void getDefpredsHelper(/*@ non_null */Expr e) {
    if (e instanceof DefPredLetExpr) {
      DefPredLetExpr dple = (DefPredLetExpr)e;
      preds.append(dple.preds);
    }
    for (int i = 0; i < e.childCount(); i++) {
      Object c = e.childAt(i);
      if (c instanceof Expr) {
        getDefpredsHelper((Expr)c);
      }
    }
  }
  
    /*
      the call to this fonction has been removed
      as it's useless for the pvs logic
    */
  protected void distinctSymbols(/*@  non_null */PrintStream to) {

    to.print("(DISTINCT");
    Enumeration e = symbols.elements();

    while (e.hasMoreElements()) {
      String s = (String)e.nextElement();
      to.print(" ");

    }
    to.print(")");
  }
  
  protected void stringLiteralAssumptions(/*@  non_null */PrintStream to) {
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
  
  protected void printFormula(/*@ non_null */PrintStream out,
      /*@ non_null */Expr e) {
    // maps GenericVarDecls to Strings
    // some complex invariant here
    
    Hashtable subst = new Hashtable();
    printFormula(out, subst, e);
  }
  
  protected void printFormula(/*@ non_null */PrintStream out, Hashtable subst,
      /*@ non_null */Expr e) {
    Assert.notNull(e);

    reallyPrintFormula(out, subst, e);

  }


    protected void reallyPrintFormula(/*@ non_null */PrintStream out,
				      Hashtable subst,
				      /*@ non_null */Expr e) {

	//++
// 	System.out.println("VcToStringPvs::reallyPrintFormula");
	//++
    
	switch (e.getTag()) {
      
	case TagConstants.DEFPREDLETEXPR: {
	    DefPredLetExpr dple = (DefPredLetExpr)e;
	    printFormula(out, subst, dple.body);
	    break;
	}
      
	case TagConstants.SUBSTEXPR: {
	    SubstExpr se = (SubstExpr)e;
	    // perform current substitution on expression
	    String expr = vc2Term(se.val, subst);
	    // get old val, install new val
	    Object old = subst.put(se.var, expr);
	    //System.out.println("old="+old);
        
	    // print
	    printFormula(out, subst, se.target);
        
	    // restore old state
	    if (old == null) subst.remove(se.var);
	    else subst.put(se.var, old);
        
	    break;
	}
      
	case TagConstants.LABELEXPR: {
	    LabelExpr le = (LabelExpr)e;
// 	    out.print("(" + (le.positive ? "LBLPOS" : "LBLNEG") + " |" + le.label
// 		      + "| ");
// 	    printFormula(out, subst, le.expr);
// 	    out.print(")");
	    out.println("%LBL"+(le.positive ? "POS" : "NEG")+replaceBadChar(le.label.toString())+"% ");
	    printFormula(out, subst, le.expr);

	    break;
	}
      
	case TagConstants.GUARDEXPR: {
	    if (!escjava.Main.options().guardedVC) {
		Assert.fail("VcToString.reallyPrintFormula: unreachable");
	    } else {
		GuardExpr ge = (GuardExpr)e;
		String var = escjava.Main.options().guardedVCPrefix
		    + UniqName.locToSuffix(ge.locPragmaDecl);
		//out.print("(IMPLIES |" + var + "| ");
		out.print("(IMPLIES " + replaceBadChar(var) + " ");
		printFormula(out, subst, ge.expr);
		out.print(")");
		escjava.Main.options().guardVars.add(var);
		break;
	    }
	}
      
	case TagConstants.FORALL:
	case TagConstants.EXISTS: {
	    QuantifiedExpr qe = (QuantifiedExpr)e;
	    if (qe.quantifier == TagConstants.FORALL) out.print("(FORALL (");
	    else out.print("(EXISTS (");
        
	    // System.out.println("qe.vars.size() "+qe.vars.size());

	    String prefix = "";
	    for (int i = 0; i < qe.vars.size(); i++) {
		GenericVarDecl decl = qe.vars.elementAt(i);
		Assert.notFalse(!subst.containsKey(decl), "Quantification over "
				+ "substituted variable");
		out.print(prefix);

		// name of the variable
		printVarDecl(out, decl);

		// type
		out.print(" : S");

// 		// writing comma, fixme
		if(i < qe.vars.size()-1)
		    out.print(",");

		prefix = " ";
	    }
	    //    out.print(") ");

	    // add the : after FORALL in pvs
	    out.print(") : \n");

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
	    if (qe.pats != null && qe.pats.size() != 0) {
		Assert.notFalse(!insideNoPats);
		insideNoPats = true;
		if (qe.pats.size() == 1) out.print("(PATS");
		else out.print("(PATS (MPAT");
		for (int i = 0; i < qe.pats.size(); i++) {
		    out.print(" ");
		    Expr nopat = qe.pats.elementAt(i);
		    printTerm(out, subst, nopat);
		}
		if (qe.pats.size() == 1) out.print(") ");
		else out.print("))");
		insideNoPats = false;
	    }
	    printFormula(out, subst, qe.expr);
	    /* for pvs,
	       useless since we added conversion between S and bool */
	    out.print(" = bool_true ");

	    out.print(") ");

	    break;
	}
      
	    //  Operators on formulas
	case TagConstants.BOOLIMPLIES:
	case TagConstants.BOOLAND:
	case TagConstants.BOOLANDX:
	case TagConstants.BOOLOR:
	case TagConstants.BOOLNOT:
	case TagConstants.BOOLEQ: {
	    NaryExpr ne = (NaryExpr)e;
	    String op;
	    boolean normalCase = true;
	    String pvsOp = null;
        
	    switch (ne.getTag()) {
	    case TagConstants.BOOLIMPLIES:
		//op = "IMPLIES";
		op = "boolImplies";
		break;
	    case TagConstants.BOOLAND:
	    case TagConstants.BOOLANDX:
		//op = "AND";
		op = "boolAnd";
		normalCase = false;
		pvsOp = "AND";
		break;
	    case TagConstants.BOOLOR:
		//op = "OR";
		op = "boolOr";
		normalCase = false;
		pvsOp = "OR";
		break;
	    case TagConstants.BOOLNOT:
		//op = "NOT";
		op = "boolNot";
		break;
	    case TagConstants.BOOLEQ:
		//op = "IFF";
		op = "boolEq";
		break;
	    default:
		Assert.fail("Fall thru");
		op = null; // dummy assignment
	    }

	    /* (EQ a b c d) =>
	     * boolAnd(EQ(a ,b), EQ(b,c)) AND boolAnd( EQ(b,c), EQ(c,d)))
	     */

	    out.print(op+"(");

	    for (int i = 0; i < ne.exprs.size(); i++) {
		out.print("(");
		printFormula(out, subst, ne.exprs.elementAt(i));
		out.print(")");

		if((i < ne.exprs.size() - 1)){ /* not the last */
		    
		    if( isEven(i) )
			out.print(",");
		    else {
			// this is the pvs operator for bool
			if((i < ne.exprs.size() - 2)){
			    out.print(")"+pvsOp+" "+op+"( ");
			}
			else /* before last */
			    out.print(")"+pvsOp+" "+"( ");
		    }
		}
		
	    }
 	    out.print(")");

// 	    /*
// 	     * Variant of the code for current_unsorted_logic_variante.pvs
// 	     */

// 	    int nbBracket = 0;
// 	    int j = 0;

// 	    out.print(op+"(");

// 	    if(normalCase){
// 		for (int i = 0; i < ne.exprs.size(); i++) {
// 		    printFormula(out, subst, ne.exprs.elementAt(i));

// 		    if( i != ne.exprs.size() -1)
// 			out.print(",");
// 		}
// 	    }
// 	    else {

// 		if(ne.exprs.size() == 2){ // no need for a list
// 		    printFormula(out, subst, ne.exprs.elementAt(0));
// 		    out.print(",");
// 		    printFormula(out, subst, ne.exprs.elementAt(1));

// 		}
// 		else {
// 		    for (int i = 0; i < ne.exprs.size(); i++) {
// 			out.print("cons(");
// 			printFormula(out, subst, ne.exprs.elementAt(i));
// 			out.print(",");
		    
// 			if(i == (ne.exprs.size() -1)) {// last
// 			    out.print("null");
// 			    for( int j2 = 0; j2 <= i; j2++)
// 				out.print(")");
// 			}
// 		    }
// 		}
// 	    }	    

// 	    out.print(")");

	    break;
	}
      
	case TagConstants.BOOLNE: {
	    NaryExpr ne = (NaryExpr)e;
	    out.print("(IFF ");
	    printFormula(out, subst, ne.exprs.elementAt(0));
	    out.print(" (NOT ");
	    printFormula(out, subst, ne.exprs.elementAt(1));
	    out.print("))");
	    break;
	}
      
	case TagConstants.METHODCALL: {
	    NaryExpr ne = (NaryExpr)e;
	    
	    /*
	     * useless to add EQ( true, call_to_fonction_returning_bool)
	     * in pvs
	     */

	    //++
//	    System.out.println("VcToStringPvs::reallyPrintFormula::MethodCall");
	    //++

	    //out.print("(EQ |@true| ( |");
	    out.print(ne.methodName);
	    //out.print("| ");
	    int n = ne.exprs.size();
	    for (int i = 0; i < n; i++) {
		printTerm(out, subst, ne.exprs.elementAt(i));
		out.print(" ");
	    }
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
	case TagConstants.TYPELE: {
	    NaryExpr ne = (NaryExpr)e;
	    String op;
	    boolean possibleDeclarationOfStackTrace = false;
        
	    switch (ne.getTag()) {
	    case TagConstants.ALLOCLT:
		op = "lockLT";
		break;
	    case TagConstants.ALLOCLE:
		op = "lockLE";
		break;
	    case TagConstants.ANYEQ:
		//op = "EQ";
		op = "refEQ";
		possibleDeclarationOfStackTrace = true;
		// for elems, alloc
		break;
	    case TagConstants.ANYNE:
		//op = "NEQ";
		op = "refNE";
		break;
	    case TagConstants.INTEGRALEQ:
		//op = "EQ";
		op = "integralEQ";
		break;
	    case TagConstants.INTEGRALGE:
		op = "integralGE";
		break;
	    case TagConstants.INTEGRALGT:
		op = "integralGT";
		break;
	    case TagConstants.INTEGRALLE:
		op = "integralLE";
		break;
	    case TagConstants.INTEGRALLT:
		op = "integralLT";
		break;
	    case TagConstants.INTEGRALNE:
		//op = "NEQ";
		op = "integralNE";
		break;
	    case TagConstants.LOCKLE:
		op = TagConstants.toVcString(TagConstants.LOCKLE);
		break;
	    case TagConstants.LOCKLT:
		op = TagConstants.toVcString(TagConstants.LOCKLT);
		break;
	    case TagConstants.REFEQ:
		//op = "EQ";
		op = "refEQ";
		break;
	    case TagConstants.REFNE:
		//op = "NEQ";
		op = "refNE";
		break;
	    case TagConstants.TYPEEQ:
		//op = "EQ";
		//op = "typeEQ";
		op = "=";
		break;
	    case TagConstants.TYPENE:
		//op = "NEQ";
		//op = "typeNE";
		op = "/=";
		break;
	    case TagConstants.TYPELE:
		//op = "<:";
		op = "<=";
		break;
	    default:
		Assert.fail("Fall thru");
		op = null; // dummy assignment
	    }
        
	    //out.print("(" + op);
	    out.print(op + "(");
	    for (int i = 0; i < ne.exprs.size(); i++) {
		if( i>=1 ) out.print(", ");
		//out.print(" ");
		// makes it more 'readable' cough cough...
		printTerm(out, subst, ne.exprs.elementAt(i));
	    }
	    out.print(")");
	    break;
	}
      
	default: {
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

	    // the two next lines are useless in pvs

	    //out.print("(EQ |@true| ");
	    printTerm(out, subst, e);
	    //out.print(")");
	    break;
	}
	}
    }

    protected boolean isEven(int i){
	
	int j = i/2;
	
	return ((j * 2) == i);

    }

  // ======================================================================
  
  /**
   * <code>insideNoPats</code> is really a parameter to *
   * <code>printTerm</code>.
   */
  
  protected boolean insideNoPats = false;

    protected void printTerm(/*@ non_null */PrintStream out, Hashtable subst,
			     /*@ non_null */Expr e) {
    
	//++
// 	System.out.println("VcToStringPvs::printTerm");
	//++
    
	int tag = e.getTag();
	boolean possibleNewPvsFunction = false;

	switch (tag) {
      
	case TagConstants.SUBSTEXPR: {

	    //++
//	    System.out.println("printTerm::SUBSTEXPR");
	    //++

	    SubstExpr se = (SubstExpr)e;
	    // perform current substitution on expression
	    String expr = vc2Term(se.val, subst);
	    
	    // get old val, install new val
	    Object old = subst.put(se.var, expr);
	    // print
	    printTerm(out, subst, se.target);
        
	    // restore old state
	    if (old == null) subst.remove(se.var);
	    else subst.put(se.var, old);
        
	    break;
	}
      
	case TagConstants.DEFPREDAPPLEXPR: {

	    //++
//	    System.out.println("printTerm::DEFPREDAPPLEXPR");
	    //++


	    DefPredApplExpr dpae = (DefPredApplExpr)e;
	    out.print("(" + dpae.predId);
	    for (int i = 0; i < dpae.args.size(); i++) {
		out.print(" ");
		printTerm(out, subst, dpae.args.elementAt(i));
	    }
	    out.print(")");
	    break;
	}
      
	case TagConstants.TYPEEXPR: {

	    //++
//	    System.out.println("printTerm::TYPEEXPR");
	    //++

	    TypeExpr te = (TypeExpr)e;
	    
	    //System.out.println(BackPred.simplifyTypeName(te.type));
	    
	    // remove | around type name

	    out.print(replaceBadChar(BackPred.simplifyTypeName(te.type)));
	    break;
	}
      
	    // Literals
      
	    // This handles case 8 of ESCJ 23b:
	case TagConstants.STRINGLIT: {

	    //++
//	    System.out.println("printTerm::STRINGLIT");
	    //++

	    LiteralExpr le = (LiteralExpr)e;
	    String s = "S_" + UniqName.locToSuffix(le.loc);
	    s = Atom.printableVersion(s);
	    stringLiterals.add(s);
	    out.print(s);
	    break;
	}
      
	case TagConstants.BOOLEANLIT: {

	    //++
//	    System.out.println("printTerm::BOOLEANLIT");
	    //++

	    LiteralExpr le = (LiteralExpr)e;
	    //	    if (((Boolean)le.value).booleanValue()) out.print("|@true|");
	    if (((Boolean)le.value).booleanValue()) out.print("bool_true ");
	    //	    else out.print("|bool$false|");
	    else out.print("bool_false ");
	    break;
	}
      
	case TagConstants.CHARLIT:
	case TagConstants.INTLIT: {
	    LiteralExpr le = (LiteralExpr)e;
	    //out.print(integralPrintName(((Integer)le.value).intValue()));

	    /*
	     * In case of special value, this fonction can print
	     * neg2147483648 or pos2147483647 (and normal int like 1 or 2).
	     * But in the latter case, we have to convert it to 
	     * real_to_S(1) otherwise pvs will consider it as a real
	     */

	    String s = integralPrintName(((Integer)le.value).intValue());

	    /* lame way to determine if it's negXXXX or a 'normal int'  */
	    
	    if( s.charAt(0)=='n' || s.charAt(0)=='p' )
		out.print(s);
	    else
		out.print("real_to_S("+s+")");

	    break;
	}
      
	case TagConstants.LONGLIT: {
	    LiteralExpr le = (LiteralExpr)e;
	    out.print(integralPrintName(((Long)le.value).longValue()));
	    break;
	}
      
	case TagConstants.FLOATLIT: {
	    LiteralExpr le = (LiteralExpr)e;
	    out.print("|F_" + ((Float)le.value).floatValue() + "|");
	    break;
	}
      
	case TagConstants.DOUBLELIT: {
	    LiteralExpr le = (LiteralExpr)e;
	    out.print("|F_" + ((Double)le.value).doubleValue() + "|");
	    break;
	}
      
	case TagConstants.NULLLIT:
	    //out.print("null");
	    out.print("java_null");
	    break;
      
	case TagConstants.SYMBOLLIT: {
	    // Handles case 5 of ESCJ 23b:
	    LiteralExpr le = (LiteralExpr)e;
	    //String s = "|" + (String)le.value + "|";

	    String s = (String)le.value;
	    symbols.add(s);

	    //System.out.println("TagConstants.SYMBOLLIT:"+s);

	    // name of different path to reach end of fonction

	    out.print(s);
	    break;
	}
      
	case TagConstants.VARIABLEACCESS: {

	    //++
//	    System.out.println("VcToStringPvs::printTerm::VariableAccess");
	    //++

	    VariableAccess va = (VariableAccess)e;
	    String to = (String)subst.get(va.decl);

	    /* replace bad char in each case, because
	     * printVarDecl was modified too
	     */
	    
	    if (to != null) out.print(replaceBadChar(to));
	    else printVarDecl(out, va.decl);
	    break;
	}
      
	    // Nary functions
	case TagConstants.ALLOCLT:
	case TagConstants.ALLOCLE:
	case TagConstants.ARRAYLENGTH:
	case TagConstants.ARRAYFRESH:
	case TagConstants.ARRAYMAKE:
	case TagConstants.ARRAYSHAPEMORE:
	case TagConstants.ARRAYSHAPEONE:
	case TagConstants.ASELEMS:
	case TagConstants.ASFIELD:
	case TagConstants.ASLOCKSET:
	case TagConstants.BOOLAND:
	case TagConstants.BOOLANDX:
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
        
	case TagConstants.INTERN:
	case TagConstants.INTERNED:
	case TagConstants.IS:
	case TagConstants.ISALLOCATED:
	case TagConstants.ISNEWARRAY:
	case TagConstants.LOCKLE:
	case TagConstants.LOCKLT:
	case TagConstants.MAX:
	case TagConstants.METHODCALL:
	case TagConstants.REFEQ:
	case TagConstants.REFNE:
	case TagConstants.SELECT:
	case TagConstants.STORE:
	case TagConstants.STRINGCAT:
	case TagConstants.STRINGCATP:
	case TagConstants.TYPEEQ:
	case TagConstants.TYPENE:
	case TagConstants.TYPELE:
	case TagConstants.TYPEOF:
	case TagConstants.UNSET:
	case TagConstants.VALLOCTIME: {
	    NaryExpr ne = (NaryExpr)e;
	    String op = null; //compiler happy \o/
	    switch (tag) {
	    case TagConstants.INTEGRALADD:
		op = "+";
		break;
	    case TagConstants.INTEGRALMUL:
		op = "*";
		break;
	    case TagConstants.INTEGRALNEG:
		//op = "- 0";
		op = "-";
		break;
	    case TagConstants.INTEGRALSUB:
		op = "-";
		break;
	    case TagConstants.TYPELE:
		op = "<:";
		break;
	    case TagConstants.METHODCALL:
		
		/*
		 * Here you catch the call to function like
		 * java.lang.Throwable#_state
		 */

		//op = "|" + ne.methodName.toString() + "|";

		op = ne.methodName.toString();
		
		op = replaceBadChar(op);

		//++
//		System.out.println("printTerm::METHODCALL "+op);
		//++

		/* in order to declare the fonction after */
		possibleNewPvsFunction = true;

		break;
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

	    //++
//	     System.out.println("printTerm::default "+op);
	    //++

	    }

	    // (typeof X) => typeOf( )

	    //out.print("(" + op);
	    out.print(op + "(");
	    for (int i = 0; i < ne.exprs.size(); i++) {

		/* makes it more readable */
		if(i!=0)
		    out.print(", ");

		printTerm(out, subst, ne.exprs.elementAt(i));
	    }
	    out.print(")");

	    if(possibleNewPvsFunction) /* declaration of the function, by adding
					  it to the appropriate set */
		add2DeclFun(op,ne.exprs.size());

	    break;
	}
      
	case TagConstants.CAST: {

	    //++
//	    System.out.println("TagConstants.CAST");
	    //++

	    NaryExpr ne = (NaryExpr)e;
	    Assert.notFalse(ne.exprs.size() == 2);
	    Expr x = ne.exprs.elementAt(0);
	    TypeExpr t = (TypeExpr)ne.exprs.elementAt(1);
        
	    if (Types.isBooleanType(t.type)) {
		// Peephole optimization to remove casts to boolean, since
		// anything is a boolean in the underlying logic (it either
		// equals |@true| or it doesn't). The reason this peephole
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
      
	case TagConstants.DTTFSA: {
	    NaryExpr ne = (NaryExpr)e;
	    printDttfsa(out, subst, ne);
	    break;
	}
      
	default:
	    Assert.fail("Bad tag in GCTerm: " + "(tag=" + tag + ","
			+ TagConstants.toVcString(tag) + ") " + e);
	}
    }

  //@ requires ne.op == TagConstants.DTTFSA;
  protected void printDttfsa(/*@ non_null */PrintStream out, Hashtable subst,
      /*@ non_null */NaryExpr ne) {
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
      for (int i = 2; i < ne.exprs.size(); i++) {
        out.print(" ");
        Expr ei = ne.exprs.elementAt(i);
        printTerm(out, subst, ei);
      }
      out.print(")");
    }
  }
  
  // ======================================================================
  
  protected void printVarDecl(/*@ non_null */PrintStream out, GenericVarDecl decl) {
      // remove bacChar 

      //out.print(Atom.printableVersion(UniqName.variable(decl)));

      out.print(replaceBadChar(Atom.printableVersion(UniqName.variable(decl))));
  }
  
  // ======================================================================
  
  protected static final long MaxIntegral = 1000000;
  
  protected static final/*@ non_null */Long minThreshold = new Long(-MaxIntegral);
  
  protected static final/*@ non_null */Long maxThreshold = new Long(MaxIntegral);
  
  /**
   * * Convert an integral # into its printname according to the rules * of ESCJ
   * 23b, part 9.
   */
  
  protected/*@ non_null */String integralPrintName(long n) {
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
  
  protected void integralPrintNameOrder(/*@ non_null */PrintStream ps) {
    int n = integralPrintNames.size();
    Assert.notFalse(2 <= n); // should contain the two thresholds
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
    
    /* added for pvs */
    StringBuffer pvsDecl = new StringBuffer("\n");
    StringBuffer pvsAxiom = new StringBuffer("\n");
    boolean somethingToDeclare = false;

    String valueI = (String)integralPrintNames.get(keys[0]);

    /* loop invariant:  valueI == integralPrintNames.get(keys[i]); */
    for (int i = 0; i < n - 1; i++) {

	/* This loop can be runned even if their is no declaration
	   That's the need for somethingToDeclare raises */
		
      String valueNext = (String)integralPrintNames.get(keys[i + 1]);
      if (keys[i] == minThreshold) {
        Assert.notFalse(keys[i + 1] == maxThreshold);
      } else {

	  // Ugly hack to print it only the first time
	  if(!somethingToDeclare) {
	      pvsAxiom.append("integralAxiom : AXIOM(");
	      somethingToDeclare = true;
	  }

	  /* lame way to determine if it's negXXXX or 10000 
	     (as we must declare only negXXXX in this case) */

	  pvsAxiom.append("(");

	  if( valueI.charAt(0)=='n' || valueI.charAt(0)=='p' )
	      /* valueI is the int declared */
	      pvsDecl.append(valueI+ " : S");
	  else 
	      pvsDecl.append(valueNext+ " : S");

	  pvsAxiom.append(valueI + " < "+ valueNext);
	  
	  pvsDecl.append("\n");
	  pvsAxiom.append(")");

      }

      if( i < n - 2 && i != 0) /* another declaration after */
	  pvsAxiom.append("AND");

      valueI = valueNext;
    }

    if(somethingToDeclare)
	pvsAxiom.append(")");

    ps.print(pvsDecl.toString());
    ps.print(pvsAxiom.toString());
    
  }
  
  static {
    resetTypeSpecific();
  }

static private int add2Decl(String s){
	
	if( s.charAt(0) == '%')
	    return 0;

	Iterator i = listOfDecl.iterator();
	String temp = null; // make the compiler happy

	while(i.hasNext()){
	    
	    try{ temp = (String)i.next();}
	    catch(Exception e){
		System.out.println("VcToStringPvs::add2Decl *** error *** "+e);
	    }

	    if( s.compareTo(temp) == 0)
		return 0;
	}

	i = listOfDeclAdd.iterator();
	
	while(i.hasNext()){
	    
	    try{ temp = (String)i.next();}
	    catch(Exception e){
		System.out.println("VcToStringPvs::add2Decl *** error *** "+e);
	    }

	    if( s.compareTo(temp) == 0)
		return 0;
	}

	listOfDeclAdd.add(s);

	//++
//	System.out.println("Adding "+s+" to the listOfDecl");
	//++

	return 1;
    }

    static private int add2DeclFun(String s, int nbParameters){
	
	if( s.charAt(0) == '%')
	    return 0;

	Iterator i = listOfDeclFun.iterator();
	String temp = null; // make the compiler happy

	while(i.hasNext()){
	    
	    try{ temp = (String)i.next();}
	    catch(Exception e){
		System.out.println("VcToStringPvs::add2AdditionalDecl *** error *** "+e);
	    }

	    if( s.compareTo(temp) == 0)
		return 0;
	}

	listOfDeclFun.add(s);

	/* this is so crappy, not to be able to push an int direclty */
	listOfDeclFunNbParam.add(new Integer(nbParameters));

	//++
//	if(listOfDeclFun.size() != listOfDeclFunNbParam.size())
//	    System.out.println("Warning, inconsistency in declaration of new functions...");
	//++

	//++
//	System.out.println("Adding "+s+", with arity "+nbParameters+" to listOfDeclFun");
	//++

	return 1;
    }

    public static String replaceBadChar(String s){

	// remove beginning and ending |
	if(s.charAt(0) == '|')
	    s = s.substring(1,s.length()-1);

	if(s.charAt(s.length()-1) == '|')
	    s = s.substring(0,s.length()-2);

	s = s.replace('@','_').replace('#','_').replace('|','_').replace('.','_').replace(':','_').replace('<','_').replace('>','_').replace('-','_').replace('^','_').replace(',','_').replace('[','_').replace(']','_').replace('!','_').replace('(','_').replace(')','_').replace(' ','_');

	if(s.compareTo("o") == 0) /* handle the case where a variable
				     is named just 'o' which is a special
				     variable in pvs */
	    s = "o_";

	/* delete _ at the beginning of variable names, coz it's forbidden in pvs */
	while(s.charAt(0)=='_')
	    s = s.substring(1,s.length());

	add2Decl(s);
	
	return s;
    }

}
