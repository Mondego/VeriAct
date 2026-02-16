package escjava.sortedProver;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Stack;
import java.util.regex.Pattern;

import javafe.ast.ASTNode;
import javafe.ast.ArrayType;
import javafe.ast.ClassDecl;
import javafe.ast.ConstructorDecl;
import javafe.ast.Expr;
import javafe.ast.ExprVec;
import javafe.ast.FieldDecl;
import javafe.ast.FormalParaDecl;
import javafe.ast.GenericVarDecl;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.MethodDecl;
import javafe.ast.PrettyPrint;
import javafe.ast.PrimitiveType;
import javafe.ast.SimpleName;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.VariableAccess;
import javafe.tc.TypeCheck;
import javafe.tc.TypeSig;
import javafe.tc.Types;
import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Info;
import javafe.util.Location;
import escjava.ast.LabelExpr;
import escjava.ast.Modifiers;
import escjava.ast.NaryExpr;
import escjava.ast.QuantifiedExpr;
import escjava.ast.SubstExpr;
import escjava.ast.TagConstants;
import escjava.ast.TypeExpr;
import escjava.backpred.BackPred;
import escjava.backpred.FindContributors;
import escjava.translate.GC;
import escjava.translate.TrAnExpr;
import escjava.translate.UniqName;

/*@ non_null_by_default @*/
public class Lifter extends EscNodeBuilder
{
	final static boolean doTrace = false;
	final Stack quantifiedVars = new Stack();
	final Hashtable bgSymbolTypes = new Hashtable();
    Hashtable symbolTypes = new Hashtable();
	final Term[] emptyTerms = new Term[0];
	public final EscNodeBuilder builder;
	final SortedBackPred backPred = new SortedBackPred();	
	int pass;
	int methodNo = 0;
	final int lastPass = 3;
	private Hashtable baseFnTranslations = new Hashtable();
    
    /**
     * Print a warning message. Originally the messages went to
     * ErrorSet.caution but that is rather annoying for the user.
     * Anyway, the purpose of this method is to provide one place
     * for the debugger to redirect various rather internal error
     * messages.
     * @param s the error message
     */
    private static void warn(String s) {
        Info.out(s); // printed only in verbose mode
    }
    
    private void newMethod()
    {
        methodNo++;
        symbolTypes = new Hashtable();
    }
	
	// public interface
	public SPred convert(Expr main) 
	{
		newMethod();
		SPred res = doConvert(transform(main));
		symbolTypes = null;
		return res;
	}
	
	public Lifter(EscNodeBuilder b)
	{
		builder = b;
		// these symbols sometimes occur by-name:
		saveBgSymbol(symInterned);
	} 
	
	public SPred generateBackPred(FindContributors scope)
	{
		Assert.notFalse(methodNo == 0);
		
		backPred.genTypeBackPred(scope, System.err);

		Term[] terms = new Term[backPred.axioms.size()];
		terms = (Term[])backPred.axioms.toArray(terms);
		Term axioms = new FnTerm(giantBoolConective(TagConstants.BOOLAND, terms.length), terms);
		
		SPred and1 = doConvert(axioms);
		
		dumpBuilder = builder;
		SAny[] dist = new SAny[backPred.distinct.size()];
		for (int i = 0; i < dist.length; ++i)
			dist[i] = ((Term)backPred.distinct.get(i)).dumpAny();
		dumpBuilder = null;
		
		SPred and2 = builder.buildDistinct(dist);
		
		baseFnTranslations = fnTranslations;
		fnTranslations = new Hashtable();
		
		newMethod();
		return builder.buildAnd(new SPred[] { and1, and2 });
	}
	// end public interface
	
	/*@ non_null_by_default @*/
	public class SortVar extends NodeBuilder.Sort
	{
		private /*@ nullable @*/Sort ref;
		
		public SortVar()
		{
			super("sortVar", null, null, null);
		}
		
		void refSet()
		{
			if (ref == null) {
				if (dumpBuilder != null)
					ref = sortRef;
				else
					Assert.fail("ref == null");
			}
		}
		
		public /*@ nullable @*/Sort getSuperSort()
		{
			refSet();
			return ref.getSuperSort();
		}

		public /*@ nullable @*/Sort getMapFrom()
		{
			refSet();			
			return ref.getMapFrom();
		}

		public /*@ nullable @*/Sort getMapTo()
		{
			refSet();			
			return ref.getMapTo();
		}
		
		public boolean isFinalized()
		{
			if (ref == null) return false;
			if (ref instanceof SortVar)
				return ((SortVar)ref).isFinalized();
			return true;
		}
		
		boolean occurCheck(Sort s)
		{
			if (s == this)
				return true;
				
			if (s instanceof SortVar && !((SortVar)s).isFinalized())
			{
				return false;
			}
			else if (s.getMapFrom() != null) {
				return occurCheck(s.getMapFrom()) || occurCheck(s.getMapTo());
			} else return false;
		}
		
		public void assign(Sort s)
		{
			Assert.notFalse(ref == null);
			if (doTrace)
				trace("assign: ?" + id + " <- " + s);
			if (occurCheck(s))
				ErrorSet.error("cyclic sort found");
			else
				ref = s;
		}
		
		public Sort theRealThing()
		{
			if (dumpBuilder != null)
				refSet();
			
			if (ref != null && ref instanceof SortVar)
				return ref.theRealThing();
			return ref == null ? this : ref;
		}
		
		public String toString()
		{
			if (ref == null) return "?" + id;
			else return ref.toString();
		}
	}
	
	/*@ non_null_by_default @*/
	public abstract class Term
	{	
		abstract public Sort getSort();
		abstract public void infer();
		
		public Term subst(Term v, Term t) {
			return this;
		}
		public void printTo(StringBuffer sb)
		{
			sb.append(super.toString());		
		}
		
		public String toString()
		{
			StringBuffer sb = new StringBuffer();
			printTo(sb);
			return sb.toString();
		}
		
		abstract public STerm dump();
		
		public SPred dumpPred()
		{	
			//ErrorSet.caution("( dumpPred");
			Assert.notFalse(follow(getSort()) == sortPred);
			SPred p = (SPred)dump();
			//ErrorSet.caution(" dumpPred )");
			return p;
		}
		
		public SAny dumpAny()
		{
      // jgc: in mobius direct vc gen, the sort any contains pred
			//Assert.notFalse(follow(getSort()) != sortPred);
			return (SAny)dump();
		}
		
		public SValue dumpValue()
		{
			Assert.notFalse(getSort().isSubSortOf(sortValue));
			return (SValue)dump();
		}
		
		public SInt dumpInt()
		{
			Assert.notFalse(getSort().isSubSortOf(sortInt));
			return (SInt)dump();
		}
		
		public SBool dumpBool()
		{
			Assert.notFalse(getSort().isSubSortOf(sortBool));
			return (SBool)dump();
		}
		
		public SReal dumpReal()
		{
			Assert.notFalse(getSort().isSubSortOf(sortReal));
			return (SReal)dump();
		}
		
		public SRef dumpRef()
		{
			Assert.notFalse(getSort().isSubSortOf(sortRef));
			return (SRef)dump();
		}
		
		public void enforceLabelSense(int sense)
		{			
		}
	}
	
	/*@ non_null_by_default @*/
	public class QuantVariableRef extends Term
	{
		final public QuantVariable qvar;
		
		public QuantVariableRef(QuantVariable q) { qvar = q; }		
		public Sort getSort() { return qvar.type; }
		public void infer() { }
		
		public void printTo(StringBuffer sb)
		{
			sb.append("?" + qvar.name + ":" + qvar.type);
		}
		
		public STerm dump()
		{
			return dumpBuilder.buildQVarRef(qvar.qvar); 
		}
		public boolean equals(/*@nullable*/ Object o) {
			if(!(o instanceof QuantVariableRef)) {
				return false;
			}
			// FIXME: [bug #582] equals is declared pure, whereas hashCode 
			// is not, hence hashCode cannot be called here.
			// return o.hashCode() == this.hashCode();
			// Here is an attempt at fixing #582:
			return qvar.equals(((QuantVariableRef)o).qvar);
		}
		public int hashCode() {
			return qvar.hashCode();
		}
		
		public Term subst(Term v, Term subst) {
			if (v.equals(this))
				return subst;
			return this;
		}
	}
	
	public QuantVariableRef mkQuantVariableRef (QuantVariable q) {
		return new QuantVariableRef(q);
	}
	
	/*@ non_null_by_default @*/
	public class FnTerm extends Term
	{
		public FnSymbol fn;
		final public Term[] args;
		public int tag;
		final public Sort retType;
		public boolean isStringConst;
		public boolean isDistinctSymbol;
    
    public FnTerm(FnSymbol fn, Term[] args, Sort s) {
      this.fn = fn;
      this.args = args;
      
      if (s == null) {
        retType = fn.retType;
        //System.out.println(fn);
      }
      else {
        retType = s;
      }
      
    }

		public FnTerm(FnSymbol fn, Term[] args)
		{
			this.fn = fn;
			this.args = args;
			if (fn == symSelect || fn == symStore || fn == symAsField)
				retType = new SortVar();
			else
				retType = fn.retType;
		}
		
		public Sort getSort() { return retType; }


	    // apply a conversion function to a term
	    public FnTerm cast(FnSymbol fn, Term term)
	    {
		return new FnTerm(fn, new Term[] { term });
	    }
		
		void enforceArgType(int i, Sort r)
		{
			r = follow(r);
      if (args[i].getSort() == null) {
        System.out.println(args[i] + " " +  args[i].getClass());
          
      }
			Sort p = follow(args[i].getSort());
			
			if (isEarlySort (r, p))
				return;
			
			FnTerm conv = null;
			
			int minpass = 2;

			if (p != r) {
			    if (p == sortValue) {
				if (r == sortBool)
				    conv = cast(symValueToBool, args[i]);
				else if (r == sortInt)
				    conv = cast(symValueToInt, args[i]);
				else if (r == sortReal)
				    conv = cast(symValueToReal, args[i]);
				else if (r == sortRef)
				    conv = cast(symValueToRef, args[i]);
				else if (r == sortType)
				    conv = cast(symRefToType, (cast(symValueToRef, args[i])));
				else if (r == sortPred)
				    // TODO flag this with warning
				    conv = cast(symValueToPred, args[i]);
			    } else if (p == sortInt && r == sortReal) {
				conv = cast(symIntToReal, args[i]);
				minpass = 0;
			    } else if (p == sortRef && r == sortType) {
				conv = cast(symRefToType, args[i]);
			    } else if (p == sortBool && r == sortPred) {
				conv = cast(symIsTrue, args[i]);
				minpass = 1;
			    } else if (p == sortPred) {
				if (r == sortBool) {
				    warn("using pred -> bool conversion! in arg #" + (1+i) + " of " + fn + " / " + this);
				    conv = cast(symPredToBool, args[i]);
				} else if (r == sortValue) {
				    warn("using pred -> value conversion! in arg #" + (1+i) + " of " + fn + " / " + this);
				    conv = cast(symBoolToValue, cast(symPredToBool, args[i]));
				}
			    }
			}

			if (pass >= minpass && conv != null) {
			    args[i] = conv;
			} else if (!require(p, r, args[i]))
				ErrorSet.error("which is arg #" + (1+i) + " of " + fn + " / " + this);
		}
		
		public void infer()
		{
			if (doTrace)
				trace("start infer " + pass + ": " + fn + " / " + this + " -> " + retType);
			
			if (args.length != fn.argumentTypes.length) {
				ErrorSet.error("wrong number of parameters to " + fn + " ---> " + this);
				return;
			}
			
			for (int i = 0; i < args.length; ++i)
				args[i].infer();			
			
			boolean skip = pass < lastPass;
			
			if (fn == symSelect)
			{
				Sort idx = new SortVar();
				Sort map = registerMapSort(idx, retType);
				
				enforceArgType(0, map);
				enforceArgType(1, idx);
			}
			else if (fn == symStore)
			{
				Sort idx = new SortVar();
				Sort val = new SortVar();
				Sort map = registerMapSort(idx, val);
				
				if (!isEarlySort(retType, map))
					unify(retType, map, this);				
				enforceArgType(0, map);
				enforceArgType(1, idx);
				enforceArgType(2, val);
			}
			else if (fn == symAnyEQ || fn == symAnyNE) {
				Sort common = new SortVar();
				
				enforceArgType(0, common);
				enforceArgType(1, common);
				
				if (follow(args[0].getSort()) == sortPred ||
					follow(args[1].getSort()) == sortPred)
				{
					if (fn == symAnyEQ)
						fn = symIff;
					else
						fn = symXor;
					skip = false;
				}
			}
			else if (fn == symAsField) {
				Sort field = registerMapSort(new SortVar(), new SortVar(), sortField);
				
				enforceArgType(0, field);
				enforceArgType(1, sortType);				
				unify(field, retType, this);				
			}
			else if (fn == symFClosedTime) {
				Sort field = registerMapSort(new SortVar(), new SortVar(), sortField);
				
				enforceArgType(0, field);				
			} else skip = false;
			
			if (!skip)
				for (int i = 0; i < args.length; ++i)
					enforceArgType(i, fn.argumentTypes[i]);					
			
			if (doTrace)
				trace("infer " + pass + ": " + fn + " / " + this + " -> " + retType);
		}
		
		public Term subst(Term t, Term subst) {
			if (subst == null)
				throw new NullPointerException();
			if(t.equals(this))
				return subst;
			else {
				Term[] res = new Term[args.length];
				boolean bHasChanged = false;
				for(int i = 0; i < args.length; i++) {
					res[i] = args[i].subst(t, subst);
					bHasChanged |= (res[i] != args[i]); 
				}
				if (bHasChanged) {
					FnTerm f = new FnTerm(fn, res);
					f.tag = tag;
					return f;
				}
				else {
					return this;
				}
			}
		}
		
		public void printTo(StringBuffer sb)
		{
			sb.append(fn.name);
			if(tag != 0 && tag < tagsIds.length) 
				sb.append(" " + tagsIds[tag]);
			if (tag > tagsIds.length)
				sb.append(" " + TagConstants.toString(tag));
			if (args.length > 0) {
				sb.append("(");
				for (int i = 0; i < args.length; ++i) {
					args[i].printTo(sb);
					if (i != args.length - 1)
						sb.append(", ");
				}
				sb.append(")");
			}
			
			sb.append(":").append(retType);
		}
		
		public STerm dump()
		{
			boolean isPred = follow(fn.retType) == sortPred;
			FnSymbol tfn = mapFnSymbolTo(dumpBuilder, fn);
			if (tfn == null && fnTranslations.containsKey(fn))
				tfn = (FnSymbol)fnTranslations.get(fn);
				
			if (tfn != null)
				if (isPred)
					return dumpBuilder.buildPredCall((PredSymbol)tfn, dumpArray(args));
				else
					return dumpBuilder.buildFnCall(tfn, dumpArray(args));
			
			if (fn == symImplies)
				return dumpBuilder.buildImplies(args[0].dumpPred(), args[1].dumpPred());
			if (fn == symIff)
				return dumpBuilder.buildIff(args[0].dumpPred(), args[1].dumpPred());
			if (fn == symXor)
				return dumpBuilder.buildXor(args[0].dumpPred(), args[1].dumpPred());
			if (fn == symNot)
				return dumpBuilder.buildNot(args[0].dumpPred());
			if (fn.name == "%and")
				return dumpBuilder.buildAnd(dumpPredArray(args));
			if (fn.name == "%or")
				return dumpBuilder.buildOr(dumpPredArray(args));
			if (fn == symTermConditional)
				return dumpBuilder.buildITE(args[0].dumpPred(), args[1].dumpValue(), args[2].dumpValue());
			if (fn == symIntPred)
				return dumpBuilder.buildIntPred(tag, args[0].dumpInt(), args[1].dumpInt());
			if (fn == symIntFn)
				return dumpBuilder.buildIntFun(tag, args[0].dumpInt(), args[1].dumpInt());
			if (fn == symRealPred)
				return dumpBuilder.buildRealPred(tag, args[0].dumpReal(), args[1].dumpReal());
			if (fn == symRealFn)
				return dumpBuilder.buildRealFun(tag, args[0].dumpReal(), args[1].dumpReal());			
			if (fn == symIntegralNeg)
				return dumpBuilder.buildIntFun(funNEG, args[0].dumpInt());
			if (fn == symFloatingNeg)
				return dumpBuilder.buildRealFun(funNEG, args[0].dumpReal());
			if (fn == symSelect)
				return dumpBuilder.buildSelect((SMap)args[0].dump(), args[1].dumpValue());
			if (fn == symStore)
				return dumpBuilder.buildStore((SMap)args[0].dump(), args[1].dumpValue(), args[2].dumpValue());
			
			if (fn == symAnyEQ || fn == symAnyNE) {
				Sort t1 = args[0].getSort().theRealThing();
				Sort t2 = args[1].getSort().theRealThing();
				
				int tag = fn == symAnyEQ ? predEQ : predNE;
 
				if (t1.isSubSortOf(sortInt) && t2.isSubSortOf(sortInt))
					return dumpBuilder.buildIntPred(tag, args[0].dumpInt(), args[1].dumpInt());
				
				if (t1.isSubSortOf(sortReal) && t2.isSubSortOf(sortReal))
					return dumpBuilder.buildRealPred(tag, args[0].dumpReal(), args[1].dumpReal());
				
				if (fn == symAnyEQ)
					return dumpBuilder.buildAnyEQ(args[0].dumpAny(), args[1].dumpAny());
				else
					return dumpBuilder.buildAnyNE(args[0].dumpAny(), args[1].dumpAny());				
			}
			
			if (fn == symIsTrue)
				return dumpBuilder.buildIsTrue(args[0].dumpBool());
			
			if (fn == symValueToPred)
				return dumpBuilder.buildIsTrue(
						(SBool)dumpBuilder.buildValueConversion(
								dumpBuilder.sortValue, dumpBuilder.sortBool, 
								args[0].dumpValue()));
			
			if (fn == symPredToBool)
				return dumpBuilder.buildITE(args[0].dumpPred(),
						dumpBuilder.buildBool(true),
						dumpBuilder.buildBool(false));
			
			if (fn == symValueToBool || fn == symValueToInt ||
			    fn == symValueToReal || fn == symValueToRef ||
			    fn == symIntToReal ||
			    fn == symBoolToValue || fn == symIntToValue ||
			    fn == symRealToValue || fn == symRefToValue ||
			    fn == symRefToType || fn == symTypeToRef
			    )
				return dumpBuilder.buildValueConversion(mapSortTo(dumpBuilder, fn.argumentTypes[0]),
								mapSortTo(dumpBuilder, fn.retType), args[0].dumpValue());
			if (fn == symIntBoolFn) {
				return dumpBuilder.buildIntBoolFun(tag, args[0].dumpInt(), args[1].dumpInt());
			}
			if (fn == symRealBoolFn) {
				return dumpBuilder.buildRealBoolFun(tag, args[0].dumpReal(), args[1].dumpReal());
			}
			if (fn == symNewObj) {
				return dumpBuilder.buildNewObject((SMap)args[0].dump(), args[1].dumpAny(), (SMap)args[2].dump(),
						args[3].dumpValue());
			}
			if (fn == symDynSelect){
				return dumpBuilder.buildDynSelect((SMap)args[0].dump(), args[1].dumpValue(), args[2].dumpValue());
			}
      if (fn == symDynLoc){
        return dumpBuilder.buildDynLoc((SMap)args[0].dump(), args[1].dumpValue(), args[2].dumpValue());
      }
			if (fn == symDynStore){
				return dumpBuilder.buildDynStore((SMap)args[0].dump(), args[1].dumpValue(), args[2].dumpValue(), args[3].dumpValue());
			}
			
			if (fn == symArrSelect){
				return dumpBuilder.buildArrSelect((SMap)args[0].dump(), args[1].dumpRef(), args[2].dumpInt());
			}
			if (fn == symArrStore){
				return dumpBuilder.buildArrStore((SMap)args[0].dump(), args[1].dumpRef(), args[2].dumpInt(), args[3].dumpValue());
			}
			if (fn == symNewArray) {
				return dumpBuilder.buildNewArray((SMap)args[0].dump(), args[1].dumpAny(), (SMap)args[2].dump(),
						args[3].dumpRef(), args[4].dumpInt());
			}
			if (fn == symAssignCompat) {
				return dumpBuilder.buildAssignCompat((SMap)args[0].dump(),  args[1].dumpValue(),
							args[2].dumpAny());
			}
      if (fn == symInv) {
        return dumpBuilder.buildInv((SMap)args[0].dumpValue(), args[1].dumpValue(), args[2].dumpAny());
      }
      if (fn == symAssignPred) {
        return dumpBuilder.buildAssignPred((SMap)args[0].dumpValue(),(SMap)args[1].dumpValue(), args[2].dumpRef(), args[3].dumpRef());
      }
      if (fn == symRefBoolFn) {
        return dumpBuilder.buildRefBoolFun(tag, args[0].dumpRef(), args[1].dumpRef());
      }
      if (fn == symIsAlive) {
        return dumpBuilder.buildIsAlive((SMap) args[0].dump(), args[1].dumpRef());
      }
      
			if(fn.name.startsWith("%"))
				System.out.println(fn.name);
			Assert.notFalse(! fn.name.startsWith("%"));
			
			tfn = isPred ? dumpBuilder.registerPredSymbol(fn.name, mapSorts(fn.argumentTypes)) :
						   dumpBuilder.registerFnSymbol(fn.name, mapSorts(fn.argumentTypes), 
								   					mapSortTo(dumpBuilder, fn.retType));
			fnTranslations.put(fn, tfn);
			if (isStringConst) stringConstants.add(this);
			if (isDistinctSymbol) distinctSymbols.add(this);
			return dump();			
		}
		
		public void enforceLabelSense(int sense)
		{
			if (fn == symNot)
				args[0].enforceLabelSense(-sense);
			else if (fn == symImplies) {
				args[0].enforceLabelSense(-sense);
				args[1].enforceLabelSense(sense);
			} else if (fn == symXor || fn == symIff) {
				args[0].enforceLabelSense(0);
				args[1].enforceLabelSense(0);
			} else {
				for (int i = 0; i < args.length; ++i)
					args[i].enforceLabelSense(sense);
			}
		}
	}
	
	public FnTerm mkFnTerm(FnSymbol fn, Term[] args) {
		return new FnTerm(fn, args);
	}
	
	
	public FnTerm mkFnTerm(FnSymbol fn, Term[] args, int tag) {
		FnTerm res = new FnTerm(fn, args);
		res.tag = tag;
		return res;
	}
	
	/*@ non_null_by_default @*/
	public class QuantTerm extends Term
	{
		public final boolean universal;
		public final QuantVariable[] vars;
		public final Term[][] pats;
		public final Term[] nopats;
		public Term body;
		
		public QuantTerm(boolean universal, QuantVariable[] vars, Term body, Term[][] pats, Term[] nopats)
		{
			this.universal = universal;
			this.vars = vars;
			this.body = body;
			this.pats = pats;
			this.nopats = nopats;		
		}
		
		public Sort getSort() { return sortPred; } 		
		public void infer()		
		{			
			if (doTrace)
				trace("infer start q " + pass + ": " + this);
			body.infer();
			body = toPred(body);
			if (doTrace)
				trace("infer q " + pass + ": " + this);
		}
		
		//TODO: cbr: What about \max, \min, \sum etc...???
		public void printTo(StringBuffer sb)
		{
      if (universal){
        sb.append("%forAll [");
      }
      else {
        sb.append("%exists [");
      }
       
			int max = vars.length -1;
			int i;
			for (i = 0; i < max; ++i)
				sb.append(vars[i].name + ":" + vars[i].type + /*"/" + vars[i].var.type +*/ ", ");
			sb.append(vars[i].name + ":" + vars[i].type + "] ");

			body.printTo(sb);
		}
		
		public STerm dump()
		{
			QuantVar[] qvars = new QuantVar[vars.length];
			QuantVar[] prev = new QuantVar[vars.length];
			for (int i = 0; i < vars.length; ++i) {
				prev[i] = vars[i].qvar;
				vars[i].qvar = dumpBuilder.registerQuantifiedVariable(vars[i].name, 
											mapSortTo(dumpBuilder, vars[i].type));
				qvars[i] = vars[i].qvar;
			}
			SPred qbody = (SPred) body.dump();
			STerm[][] qpats = null;
			STerm[] qnopats = null;
			
			if (pats != null) {
			    // patterns might be SPred, e.g. for the 'is' predicate,
			    // so this must be STerm, not SAny,
				qpats = new STerm[pats.length][];
				for (int i = 0; i < pats.length; ++i)
					qpats[i] = dumpTermArray(pats[i]);				
			}
			
			if (nopats != null) qnopats = dumpTermArray(nopats);
			
			for (int i = 0; i < vars.length; ++i) {
				vars[i].qvar = prev[i];
			}
			
			if (universal)
				return dumpBuilder.buildForAll(qvars, qbody, qpats, qnopats);
			else if (qpats == null && qnopats == null)
				return dumpBuilder.buildExists(qvars, qbody);
			else
				return dumpBuilder.buildNot( 
						dumpBuilder.buildForAll(qvars,
								dumpBuilder.buildNot(qbody),
								qpats, qnopats));
		}
		
		public void enforceLabelSense(int sense)
		{
			body.enforceLabelSense(sense);
		}
		
		public Term subst(Term t, Term subst) {
			if (subst == null)
				throw new NullPointerException();
			if(t.equals(this))
				return subst;
			else {
				boolean bHasChanged = false;
				Term res = body.subst(t, subst);
				bHasChanged = (res != body); 
				if (bHasChanged) {
					QuantTerm f = new QuantTerm(universal, vars, res, pats, nopats);
					return f;
				}
				else {
					return this;
				}
			}
		}
	}
	
	public QuantTerm mkQuantTerm(boolean universal, QuantVariable[] vars, Term body, Term[][] pats, Term[] nopats) {
		return new QuantTerm(universal, vars, body, pats, nopats);
	}
	
	/*@ non_null_by_default @*/
	public class QuantVariable
	{
		public final GenericVarDecl var;
		public final String name;
		public final Sort type;
		
		public QuantVar qvar;
		
		public QuantVariable(GenericVarDecl v, String n)
		{
			var = v;
			name = n;
			type = typeToSort(v.type);
		}
		
		public QuantVariable(String n, Sort s) {
			var = null;
			type = s;
			name = n;
		}
		public boolean equals(/*@nullable*/ Object o) {
			if(o == this)
				return true;
			if(!(o instanceof QuantVariable)) {
				return false;
			}
			// FIXME: [bug #582] equals is declared pure, whereas hashCode 
			// is not, hence hashCode cannot be called here.
			// return this.hashCode() == o.hashCode();
			// 
			// Here is an attempt at fixing #582:
			// The comparison was previously based on name.hashCode
			// so I am assuming that name.equals() is equally valid:
			return name.equals(((QuantVariable)o).name);
		}
		
		public int hashCode() {
			return name.hashCode();
		}
	}
	public QuantVariable mkQuantVariable(String n, Sort s) {
		QuantVariable v = new QuantVariable(n, s);
		v.qvar = new QuantVar(n, s);
		return v;
	}
	
	public QuantVariable mkQuantVariable(GenericVarDecl g, String n) {
		QuantVariable v = new QuantVariable(g, n);
		v.qvar = new QuantVar(n, v.type);
		return v; 
	}

	/*@ non_null_by_default @*/
	public class LabeledTerm extends Term
	{
		public final boolean positive;
		public final String label;
		public Term body;
		boolean dirty;
		boolean skipIt;
		
		public LabeledTerm(boolean pos, String l, Term b)
		{
			positive = pos;
			label = l;
			body = b;
		}
		
		public Sort getSort() { return body.getSort(); } 		
		public void infer() 
		{
			body.infer();
			body = toPred(body);
		}		
		
		public void printTo(StringBuffer sb)
		{
			if (!positive)
				sb.append("~");
			sb.append(label).append(": ");
			body.printTo(sb);
		}
		
		public STerm dump()
		{
			if (skipIt)
				return body.dump();
			else
				return dumpBuilder.buildLabel(positive, label, (SPred)body.dump());
		}
		
		public void enforceLabelSense(int sense)
		{
			Assert.notFalse(!dirty);
			dirty = true;
			if ((positive && sense > 0) || (!positive && sense < 0)) {}
			else {
				//ErrorSet.error("label: " + label + " used with wrong sense s="+sense);
				//ErrorSet.caution("change positive: " + positive + " into " + (sense < 0));
				skipIt = true;
			}
			body.enforceLabelSense(sense);
		}
	}
	public LabeledTerm mkLabeledTerm(boolean pos, String l, Term b) {
		return new LabeledTerm(pos, l, b);
	}
  
  public PredSymbol mkPredSymbol(final String name, final Sort [] tab) {
    return new PredSymbol(name, tab);
  }
  public FnSymbol mkFnSymbol(final String name, final Sort [] tab,
                               final Sort ret) {
    return new FnSymbol(name, tab, ret);
  }
	/*@ non_null_by_default @*/
	public class IntLiteral extends Term
	{
		final public long value;
		public IntLiteral(long v) { value = v; }
		public Sort getSort() { return sortInt; }
		public void infer() { }
		public void printTo(StringBuffer sb) { sb.append(value); }
		public STerm dump() { return dumpBuilder.buildInt(value); }
	}
	public IntLiteral mkIntLiteral(long v) {
		return new IntLiteral(v);
	}
	
	/*@ non_null_by_default @*/
	public class RealLiteral extends Term
	{
		final public double value;
		public RealLiteral(double v) { value = v; }
		public Sort getSort() { return sortReal; } 
		public void infer() { }
		public STerm dump() { return dumpBuilder.buildReal(value); }
	}
	public RealLiteral mkRealLiteral(double v) {
		return new RealLiteral(v);
	}
	/*@ non_null_by_default @*/
	public class BoolLiteral extends Term
	{
		final public boolean value;
		public BoolLiteral(boolean v) {	value = v; }
		public Sort getSort() { return sortBool; }
		public void infer() { }
		public STerm dump() { return dumpBuilder.buildBool(value); }
		public void printTo(StringBuffer sb) {
			sb.append(value);
		}
	}
	public BoolLiteral mkBoolLiteral(boolean v) {
		return new BoolLiteral(v);
	}
	public FnTerm mkPredLiteral(boolean v) {
		return mkFnTerm(symIsTrue, new Term[] {mkBoolLiteral(v)});
	}
	/*@ non_null_by_default @*/
	public class NullLiteral extends Term
	{
		public NullLiteral() { }
		public Sort getSort() { return sortRef; }
		public void infer() { }
		public void printTo(StringBuffer sb) {
			sb.append("null");
		}
		public STerm dump() { return dumpBuilder.buildNull(); }
	}
	public NullLiteral mkNullLiteral() {
		return new NullLiteral();
	}
	
	public PredSymbol symImplies = registerPredSymbol("%implies", new Sort[] { sortPred, sortPred }, TagConstants.BOOLIMPLIES);
	public PredSymbol symOr = registerPredSymbol("%or", new Sort[] { sortPred, sortPred });
	public PredSymbol symAnd = registerPredSymbol("%and", new Sort[] { sortPred, sortPred });
	public PredSymbol symIff = registerPredSymbol("%iff", new Sort[] { sortPred, sortPred }, TagConstants.BOOLEQ);
	public PredSymbol symXor = registerPredSymbol("%xor", new Sort[] { sortPred, sortPred }, TagConstants.BOOLNE);
	public PredSymbol symNot = registerPredSymbol("%not", new Sort[] { sortPred }, TagConstants.BOOLNOT);
  public FnSymbol symTermConditional = registerFnSymbol("%ite", new Sort[] { sortPred, sortValue, sortValue }, sortValue, TagConstants.CONDITIONAL);
	public PredSymbol symIntPred = registerPredSymbol("%int-pred", new Sort[] { sortInt, sortInt });
	public PredSymbol symRealPred = registerPredSymbol("%real-pred", new Sort[] { sortReal, sortReal });
	public FnSymbol symBoolPred = registerFnSymbol("%bool-pred", new Sort[] { sortBool, sortBool }, sortPred);
	public FnSymbol symIntBoolFn = registerFnSymbol("%int-bool-fn", new Sort[] { sortInt, sortInt }, sortBool);
	public FnSymbol symRealBoolFn = registerFnSymbol("%real-bool-fn", new Sort[] { sortReal, sortReal }, sortBool);
  public FnSymbol symRefBoolFn = registerFnSymbol("%ref-bool-fn", new Sort[] { sortRef, sortRef }, sortBool);
	public FnSymbol symIntFn = registerFnSymbol("%int-fn", new Sort[] { sortInt, sortInt }, sortInt);
	public FnSymbol symRealFn = registerFnSymbol("%real-fn", new Sort[] { sortReal, sortReal }, sortReal);
	public FnSymbol symBoolFn = registerFnSymbol("%bool-fn", new Sort[] { sortBool, sortBool }, sortBool);
	public FnSymbol symBoolUnaryFn = registerFnSymbol("%bool-unary-fn", new Sort[] { sortBool}, sortBool);

  
	
	/** 
	 * the unary sub symbol for integrals, for Reals it is called Floating 
	 * @see #symFloatingNeg
	 */
	public FnSymbol symIntegralNeg = registerFnSymbol("%integralNeg", new Sort[] { sortInt }, sortInt, TagConstants.INTEGRALNEG);
    public FnSymbol symFloatingNeg = registerFnSymbol("%floatingNeg", new Sort[] { sortReal }, sortReal, TagConstants.FLOATINGNEG);    
	public FnSymbol symSelect = registerFnSymbol("%select", new Sort[] { sortMap, sortValue }, sortValue, TagConstants.SELECT);
	public FnSymbol symStore = registerFnSymbol("%store", new Sort[] { sortMap, sortValue, sortValue }, sortMap, TagConstants.STORE);

	
	public PredSymbol symAnyEQ = registerPredSymbol("%anyEQ", new Sort[] { sortValue, sortValue }, TagConstants.ANYEQ);
	public PredSymbol symAnyNE = registerPredSymbol("%anyNE", new Sort[] { sortValue, sortValue }, TagConstants.ANYNE);
	public PredSymbol symIsTrue = registerPredSymbol("%isTrue", new Sort[] { sortBool });
	
    public FnSymbol symValueToRef = registerFnSymbol("%valueToRef", new Sort[] { sortValue }, sortRef);
    public FnSymbol symValueToInt = registerFnSymbol("%valueToInt", new Sort[] { sortValue }, sortInt);
    public FnSymbol symValueToBool = registerFnSymbol("%valueToBool", new Sort[] { sortValue }, sortBool);
    public FnSymbol symValueToReal = registerFnSymbol("%valueToReal", new Sort[] { sortValue }, sortReal);
    public FnSymbol symValueToAny = registerFnSymbol("%valueToAny", new Sort[] { sortValue }, sortAny);

    
    public FnSymbol symRefToValue = registerFnSymbol("%refToValue", new Sort[] { sortRef}, sortValue);
    public FnSymbol symIntToValue = registerFnSymbol("%intToValue", new Sort[] { sortInt}, sortValue);
    public FnSymbol symBoolToValue = registerFnSymbol("%boolToValue", new Sort[] {sortBool}, sortValue);
    public FnSymbol symRealToValue = registerFnSymbol("%tealToValue", new Sort[] {sortReal}, sortValue);

    public FnSymbol symIntToReal = registerFnSymbol("%intToReal", new Sort[] { sortInt }, sortReal);

    // in some verification conditions types are used as references (meta classes),
    // so we need to be able to convert between the two.
    // e.g in the escjava test25 test case DirectType.java we get something like:
    //   getClass(_) = typeof(_)
    //   getClass(_) != null
    //   is(getClass(_), T_java.lang.Class)
    // so getClass returns the type (class) of an object,
    // which is compared with another type resp. a reference.
    public FnSymbol symRefToType = registerFnSymbol("%refToType", new Sort[] { sortRef}, sortType);
    public FnSymbol symTypeToRef = registerFnSymbol("%typeToRef", new Sort[] { sortType}, sortRef);

    public PredSymbol symValueToPred = registerPredSymbol("%valueToPred", new Sort[] { sortValue });
    public FnSymbol symPredToBool = registerFnSymbol("%predToBool", new Sort[] { sortPred }, sortBool);    
    
    
    // mobius direct vcgen specific constructs
    /** symbol to mean a new object has been created  oldHeap -> type -> obj -> heap -> newloc -> pred  */
    public FnSymbol symNewObj = registerFnSymbol("%newObj", new Sort[] { sortMap, sortType, sortMap, sortValue }, sortPred);
    /** symbol for dynamic fields select Map -> Ref -> Name -> Value */
    public FnSymbol symDynSelect = registerFnSymbol("%dynSelect", new Sort[] { sortMap, sortValue, sortRef }, sortValue);
    /** symbol for dynamic fields select Map -> Ref -> Name -> Value */
    public FnSymbol symDynLoc = registerFnSymbol("%dynLoc", new Sort[] { sortMap, sortValue, sortRef }, sortRef);
    /** symbol for dynamic fields store Map -> Ref -> Name -> Value -> Map */
    public FnSymbol symDynStore = registerFnSymbol("%dynStore", new Sort[] { sortMap, sortValue, sortRef, sortValue }, sortMap);
    /** symbol to mean a new array has been created */
    public FnSymbol symNewArray = registerFnSymbol("%newArray", new Sort[] { sortMap, sortType, sortMap, sortRef, sortInt}, sortPred);
    /** symbol for array select (heap -> ref -> int -> value) */
    public FnSymbol symArrSelect = registerFnSymbol("%arrSelect", new Sort[] { sortMap, sortRef, sortInt }, sortValue);
    /** symbol for array store (heap -> ref -> int -> value -> pred) */
    public FnSymbol symArrStore = registerFnSymbol("%arrStore", new Sort[] { sortMap, sortRef, sortInt, sortValue }, sortMap);
    /** bicolano special subtyping relation (heap -> value -> type -> pred) */
    public FnSymbol symAssignCompat = registerFnSymbol("%assignCompat", new Sort[] { sortMap, sortValue, sortType }, sortPred);
    /** cbr: used for invariants. \forall x,t : isAlive(heap, x) & typeof(x)=t -> inv(heap, x,t) */
    public PredSymbol symInv = registerPredSymbol("%inv", new Sort[]{sortMap, sortRef, sortType});
    /** cbr: symbol to denote that the object is alive in heap */
    public PredSymbol symIsAlive = registerPredSymbol("%isAlive", new Sort[] {sortMap, sortRef});
    /** cbr: symbol to state that field is of object in given heap*/
    public PredSymbol symAssignPred = registerPredSymbol("%assignPred", new Sort[] {sortMap, sortMap, sortRef, sortRef});
    
	// we just want Sort and the like, don't implement anything	
	static class Die extends RuntimeException { private static final long serialVersionUID = 1L; }
	public SAny buildFnCall(FnSymbol fn, SAny[] args) { throw new Die(); }
	public SAny buildConstantRef(FnSymbol c) { throw new Die(); }
	public SAny buildQVarRef(QuantVar v) { throw new Die(); }
	public SPred buildPredCall(PredSymbol fn, SAny[] args) { throw new Die(); }
	
	public SPred buildImplies(SPred arg1, SPred arg2) { throw new Die(); }
	public SPred buildIff(SPred arg1, SPred arg2) { throw new Die(); }
	public SPred buildXor(SPred arg1, SPred arg2) { throw new Die(); }
	public SPred buildAnd(SPred[] args) { throw new Die(); }
	public SPred buildOr(SPred[] args) { throw new Die(); }
	public SPred buildNot(SPred arg) { throw new Die(); }
	public SValue buildITE(SPred cond, SValue then_part, SValue else_part) { throw new Die(); }
	public SPred buildTrue() { throw new Die(); }
	public SPred buildDistinct(SAny[] terms) { throw new Die(); }
	public SPred buildLabel(boolean positive, String name, SPred pred) { throw new Die(); }
	public SPred buildForAll(QuantVar[] vars, SPred body, STerm[][] pats, STerm[] nopats) { throw new Die(); }
	public SPred buildExists(QuantVar[] vars, SPred body) { throw new Die(); }
	public SPred buildIntPred(int intPredTag, SInt arg1, SInt arg2) { throw new Die(); }
	public SInt buildIntFun(int intFunTag, SInt arg1, SInt arg2) { throw new Die(); }
	public SBool buildIntBoolFun(int intFunTag, SInt arg1, SInt arg2) { throw new Die(); }
	public SPred buildRealPred(int realPredTag, SReal arg1, SReal arg2) { throw new Die(); }
	public SBool buildRealBoolFun(int realPredTag, SReal arg1, SReal arg2) { throw new Die(); }
	public SReal buildRealFun(int realFunTag, SReal arg1, SReal arg2) { throw new Die(); }
  public SBool buildRefBoolFun(int refPredTag, SRef arg1, SRef arg2) { throw new Die(); }
  public SInt buildIntFun(int intFunTag, SInt arg1) { throw new Die(); }
	public SReal buildRealFun(int realFunTag, SReal arg1) { throw new Die(); }
	public SBool buildBool(boolean b) { throw new Die(); }
	public SInt buildInt(long n) { throw new Die(); }
	public SReal buildReal(double f) { throw new Die(); }
	public SRef buildNull() { throw new Die(); }
	public SValue buildSelect(SMap map, SValue idx) { throw new Die(); }
	public SMap buildStore(SMap map, SValue idx, SValue val) { throw new Die(); }
	public SPred buildAnyEQ(SAny arg1, SAny arg2) { throw new Die(); }
	public SPred buildAnyNE(SAny arg1, SAny arg2) { throw new Die(); }

	public SValue buildValueConversion(Sort from, Sort to, SValue val) { throw new Die(); }
	public SPred buildIsTrue(SBool val) { throw new Die(); }

	// Mobius specific stuff...
	public SPred buildNewObject(SMap oldh, SAny type, SMap heap, SValue r) { throw new Die(); }
	public SAny buildSort(Sort s) { throw new Die(); }
	public SValue buildDynSelect(SMap map, SValue obj, SAny field) {throw new Die(); }
	public SMap buildDynStore(SMap map, SValue obj, SAny field, SValue val) {throw new Die(); }
  public SRef buildDynLoc(SMap map, SValue obj, SAny field) {throw new Die(); }
  public SPred buildNewArray(SMap oldh, SAny type, SMap heap, SRef r, SInt len) { throw new Die(); }
	public SValue buildArrSelect(SMap map, SRef obj, SInt idx) {throw new Die(); }
	public SMap buildArrStore(SMap map, SRef obj, SInt idx, SValue val) {throw new Die(); }
	public SPred buildAssignCompat(SMap map, SValue val, SAny type) {throw new Die(); }
  public SPred buildInv(SMap map, SValue val, SAny type) {throw new Die(); }
  public SPred buildIsAlive(SMap map, SRef obj) {throw new Die(); }
  public SPred buildAssignPred(SMap map, SMap map_pre, SRef target, SRef loc) {throw new Die(); }
  
	boolean isEarlySort(Sort s, Sort p)
	{
		return isEarlySort(s) || isEarlySort(p);
	}
	
	boolean isEarlySort(Sort s)
	{
		s = follow(s);
		
		if (pass == 0)
			return s == sortAny || s == sortPred || s == sortValue || s == sortRef;
		else if (pass == 1)
			return s == sortAny || s == sortPred || s == sortValue;
		else if (pass == 2)
			return s == sortAny || s == sortPred;
		else
			return false;
	}
	
	public SPred doConvert(Term root) 
	{		
		pass = 0;
		while (pass <= lastPass) {
			root.infer();
			pass++;
		}
		
		return build(root);
	}
	
	public SPred build(Term root)
	{
		dumpBuilder = builder;
		stringConstants.clear();
		distinctSymbols.clear();
        fnTranslations = (Hashtable)baseFnTranslations.clone();
		
		root.enforceLabelSense(-1);
		SPred res = root.dumpPred();
		
		SPred[] assumptions = new SPred[stringConstants.size() * 2 + 1];
		
		for (int i = 0; i < stringConstants.size(); ++i) {
			SRef str = ((Term)stringConstants.get(i)).dumpRef();
			assumptions[2*i] = dumpBuilder.buildAnyNE(str, dumpBuilder.buildNull());
			assumptions[2*i+1] = dumpBuilder.buildAnyEQ(
					dumpBuilder.buildFnCall(symTypeOf, new SAny[] { str }),
					symbolRef("T_java.lang.String", sortType).dumpAny());
		}
		
		SAny[] terms = new SAny[distinctSymbols.size()];
		for (int i = 0; i < terms.length; ++i)
			terms[i] = ((Term)distinctSymbols.get(i)).dumpAny();
		assumptions[stringConstants.size()*2] = 
			terms.length == 0 ? dumpBuilder.buildTrue() : dumpBuilder.buildDistinct(terms);
		
		res = dumpBuilder.buildImplies(dumpBuilder.buildAnd(assumptions), res);
		
					
		dumpBuilder = null;
		return res;
	}
	
	Sort follow(Sort s)
	{
    if (s == null) {
      System.out.println("hell");
    }
		return s.theRealThing();
	}
	
	private boolean isFinalized(/*@ nullable @*/Sort s)
	{
		return s != null && !(s instanceof SortVar && !((SortVar)s).isFinalized());
	}
	
	// make sure s1<:s2
	private boolean require(Sort s1, Sort s2, Object where)
	{
		s1 = follow(s1);
		s2 = follow(s2);
		
		if (s1 == s2) return true;
		
		if (!isFinalized(s1))
			((SortVar)s1).assign(s2);
		else if (!isFinalized(s2))
			((SortVar)s2).assign(s1);
		else if (s1.isSubSortOf(s2))
		{}
		else if (s1.getMapFrom() != null && s2.getMapFrom() != null) {
			if (isFinalized(s2.getMapTo()) && s2.getMapTo().getMapFrom() != null) {
				return unify(sortRef, s1.getMapFrom(), where) &&
					   unify(sortRef, s2.getMapFrom(), where) &&
					   unify(sortArrayValue, s1.getMapTo(), where) &&
					   unify(sortArrayValue, s2.getMapTo(), where);
			}
			else {
				return unify(s1.getMapFrom(), s2.getMapFrom(), where) &&
					   unify(s1.getMapTo(), s2.getMapTo(), where);
			}
		}
		else {
			ErrorSet.error("the sort >" + s1 + "< is required to be subsort of >" + s2 + "< in " + where);
			return false;
		}
		
		return true;
	}
	
	private boolean unify(Sort s1, Sort s2, Object where)
	{
		return require(s1, s2, where) && require(s2, s1, where);
	}
	
	public FnTerm symbolRef(String name, /*@ nullable @*/Sort s)
	{
		FnSymbol fn = getFnSymbol(name, 0);
		if (s != null)
			if (!require(s, fn.retType, "symbol ref"))
				ErrorSet.error("symbol ref " + name);
    FnTerm f = new FnTerm(fn, emptyTerms, s);
    
		return f;
	}
	
	public FnTerm symbolRef(String name) {
		return symbolRef(name, null);
	}
	
	void saveBgSymbol(FnSymbol s)
	{
		String nameHt = s.name + "." + s.argumentTypes.length;
		bgSymbolTypes.put(nameHt, s);
	}
	
	public FnSymbol getFnSymbol(String name, int arity)
	{
        String nameHt = name + "." + arity;
        
		if (symbolTypes.containsKey(nameHt))
			return (FnSymbol)symbolTypes.get(nameHt);
		
		if (bgSymbolTypes.containsKey(nameHt))
			return (FnSymbol)bgSymbolTypes.get(nameHt);
		
		FnSymbol fn;
		if (arity == 0)
			fn = registerConstant(name, new SortVar());
		else {
			Sort[] args = new Sort[arity];
			for (int i = 0; i < arity; ++i)
				args[i] = new SortVar();
			fn = registerFnSymbol(name, args, new SortVar());
		}
		(methodNo == 0 ? bgSymbolTypes : symbolTypes).put(nameHt, fn);
		if (arity == 0 && name.startsWith("elems") && 
				(name.startsWith("elems<") || 
						name.equals("elems") ||
						name.startsWith("elems@") ||
						name.startsWith("elems-") ||
						name.startsWith("elems:")))
			unify(fn.retType, sortElems, "elems* hack");
		if (arity == 0 && name.startsWith("owner:") || name.equals("owner"))
			unify(fn.retType, sortOwner, "owner hack");
		return fn;
	}
	
	private void trace(String msg)
	{
		Assert.notFalse (doTrace);
		warn(msg);
	}
		
	public Sort typeToSort(Type t)
	{
		switch (t.getTag()) {
		case TagConstants.ARRAYTYPE:
			return sortArray;
			
		case TagConstants.BOOLEANTYPE:
			return sortBool;
		
		case TagConstants.DOUBLETYPE:
		case TagConstants.FLOATTYPE:
			return sortReal;
			
		case TagConstants.BYTETYPE:
		case TagConstants.SHORTTYPE:
		case TagConstants.INTTYPE:
		case TagConstants.CHARTYPE:
		case TagConstants.LONGTYPE:
		case TagConstants.BIGINTTYPE:
			return sortInt;
			
		case TagConstants.TYPESIG:
		case TagConstants.TYPENAME:
		case TagConstants.TYPECODE:
		case TagConstants.NULLTYPE:
		//case TagConstants.TYPE:
		//case TagConstants.TYPETYPE:
			return sortRef;
		
		// HACK! HACK! HACK! HACK! HACK! HACK!
		case TagConstants.VOIDTYPE:
			return sortRef;
    	// HACK! HACK! HACK! HACK! HACK! HACK! (end)
		
		case TagConstants.ANY:
			return new SortVar();
			
		default:
			warn("unknown type: " + TagConstants.toString(t.getTag()) + ":" + PrettyPrint.inst.toString(t));
			
			return new SortVar();
		}
	}
	
	private FnSymbol giantBoolConective(int tag, int arity) {
		FnSymbol fn;
		fn = getFnSymbol(tag == TagConstants.BOOLOR ? "%or" : "%and",
						 arity);
		for (int i = 0; i < fn.argumentTypes.length; ++i)
			unify(fn.argumentTypes[i], sortPred, "and/or");
		unify(fn.retType, sortPred, "and/or");
		return fn;
	}	
	private Pattern number = Pattern.compile("[0-9]+");	
	private Term doTransform(ASTNode n)
	{		
		// declarations & instancations
		/* int nbChilds = n.childCount();
		   Object o = null; */
		
		// all types checked are in alphabetical order
		if (n instanceof ArrayType) {			
			ArrayType m = (ArrayType) n;			
			return new FnTerm(symArray, new Term[] { transform(m.elemType) });
			
		} else if (n instanceof LiteralExpr) {
			LiteralExpr m = (LiteralExpr) n;
			
			switch (m.getTag()) {
			case TagConstants.STRINGLIT:
				String s = "S_" + UniqName.locToSuffix(m.loc);
				FnTerm f = symbolRef(s, sortString);
				f.isStringConst = true;
				return f;
			case TagConstants.BOOLEANLIT: 
				return new BoolLiteral(((Boolean) m.value).booleanValue());
			case TagConstants.INTLIT:
			case TagConstants.CHARLIT:
				return new IntLiteral(((Integer) m.value).intValue());
			case TagConstants.LONGLIT:
				return new IntLiteral(((Long) m.value).longValue());
			case TagConstants.FLOATLIT:
				return new RealLiteral(((Float) m.value).floatValue());
			case TagConstants.DOUBLELIT:
				return new RealLiteral(((Double) m.value).doubleValue());
			case TagConstants.NULLLIT:
				return new NullLiteral();
			case TagConstants.SYMBOLLIT: {
				String v = (String)m.value;
				if (number.matcher(v).matches())
					return new IntLiteral(Long.parseLong(v));
				else {
					//ErrorSet.caution("symbol lit " + v);
					FnTerm a = symbolRef(v);
					a.isDistinctSymbol = true;
					return a;
				}
			}
			default:
				ErrorSet.fatal("Instanceof LiteralExpr, case missed :"
						+ TagConstants.toString(m.getTag()));
				return null;
			}
		}
		else if (n instanceof LabelExpr) {
			LabelExpr l = ((LabelExpr)n);			
			return new LabeledTerm(l.positive, l.label.toString(), transform(l.expr));
		}
		// name of a method
		else if (n instanceof NaryExpr) {
			NaryExpr m = (NaryExpr) n;
			
			FnSymbol fn = null;
			int tag = 0;
			int arity = m.childCount() - 1;
			
			switch (m.getTag()) {
			// hack: REFEQ is used to compare integers
			case TagConstants.REFEQ:
				fn = symAnyEQ; break;
			case TagConstants.REFNE:
				fn = symAnyNE; break;
				
			case TagConstants.BOOLAND:
			case TagConstants.BOOLANDX:
			case TagConstants.BOOLOR:
				fn = giantBoolConective(m.getTag(), arity);
				break;
			
			// integral comparisons
			case TagConstants.INTEGRALEQ:
				fn = symIntPred; tag = predEQ; break;
			case TagConstants.INTEGRALGE:
				fn = symIntPred; tag = predGE; break;
			case TagConstants.INTEGRALGT:
				fn = symIntPred; tag = predGT; break;
			case TagConstants.INTEGRALLE:
				fn = symIntPred; tag = predLE; break;
			case TagConstants.INTEGRALLT:
				fn = symIntPred; tag = predLT; break;
			case TagConstants.INTEGRALNE:
				fn = symIntPred; tag = predNE; break;
			
			// int functions
			case TagConstants.INTEGRALADD:
				fn = symIntFn; tag = funADD; break;
			case TagConstants.INTEGRALDIV:
				fn = symIntFn; tag = funDIV; break;
			case TagConstants.INTEGRALMOD:
				fn = symIntFn; tag = funMOD; break;
			case TagConstants.INTEGRALMUL:
				fn = symIntFn; tag = funMUL; break;
			case TagConstants.INTEGRALSUB:
				fn = symIntFn; tag = funSUB; break;
				
			case TagConstants.INTSHIFTRU:
				fn = symIntFn; tag = funUSR32; break;
			case TagConstants.INTSHIFTR:
				fn = symIntFn; tag = funASR32; break;
			case TagConstants.INTSHIFTL:
				fn = symIntFn; tag = funSL32; break;
				
			case TagConstants.LONGSHIFTRU:
				fn = symIntFn; tag = funUSR64; break;
			case TagConstants.LONGSHIFTR:
				fn = symIntFn; tag = funASR64; break;
			case TagConstants.LONGSHIFTL:
				fn = symIntFn; tag = funSL64; break;
				
			// real comparisons
			case TagConstants.FLOATINGEQ:
				fn = symRealPred; tag = predEQ; break;
			case TagConstants.FLOATINGGE:
				fn = symRealPred; tag = predGE; break;
			case TagConstants.FLOATINGGT:
				fn = symRealPred; tag = predGT; break;
			case TagConstants.FLOATINGLE:
				fn = symRealPred; tag = predLE; break;
			case TagConstants.FLOATINGLT:
				fn = symRealPred; tag = predLT; break;
			case TagConstants.FLOATINGNE:
				fn = symRealPred; tag = predNE; break;
			
			// real functions
			case TagConstants.FLOATINGADD:
				fn = symRealFn; tag = funADD; break;
			case TagConstants.FLOATINGDIV:
				fn = symRealFn; tag = funDIV; break;
			case TagConstants.FLOATINGMOD:
				fn = symRealFn; tag = funMOD; break;
			case TagConstants.FLOATINGMUL:
				fn = symRealFn; tag = funMUL; break;
			case TagConstants.FLOATINGSUB:
				fn = symRealFn; tag = funSUB; break;
			
			case TagConstants.DTTFSA: {
				LiteralExpr lit = (LiteralExpr)n.childAt(2);
				String op = (String)lit.value;
				if (arity == 1) {
					//ErrorSet.caution("Dttfsa " + op);
					return symbolRef(op);
				} else if (op.equals("identity")) {
					Assert.notFalse(arity == 3);
					Term body = transform((ASTNode)n.childAt(3));
					return body;
				} else {
					arity -= 2;
					fn = getFnSymbol(op, arity);
					Term[] args = new Term[arity];
					for (int i = 0; i < arity; ++i)
						args[i] = transform((ASTNode)n.childAt(i+3));
					return new FnTerm(fn, args); 
				}
			}
			
			case TagConstants.SUM:
				Assert.fail("sum unhandled"); break;
				
			case TagConstants.METHODCALL:
				ASTNode sym = m.symbol;
				fn = getFnSymbol(m.methodName.toString(), arity); 
				if (sym == null) {
                    Info.out("no symbol stored in methodCall: " + m.methodName);
				} else if (sym instanceof FieldDecl) {
					Assert.notFalse(arity <= 2);
					Sort ft = typeToSort(((FieldDecl)sym).type);
					if (arity == 0) {
						Sort s = registerMapSort(sortRef, ft, sortField);
						unify(fn.retType, s, "mc");
					} else {
						unify(fn.retType, ft, "mc1");
						for (int i = 0; i < fn.argumentTypes.length; ++i)
							unify(fn.argumentTypes[i], sortRef, "mca");
					}
				} else if (sym instanceof GenericVarDecl) {
					warn("gvd in methodCall: " + m.methodName);
				} else if (sym instanceof MethodDecl) {
					MethodDecl md = (MethodDecl)sym;
					int off = arity - md.args.size(); 
					Assert.notFalse(off <= 2);
					for (int i = 0; i < arity; ++i) {
						Sort s = sortRef;
						if (i >= off)
							s = typeToSort(md.args.elementAt(i - off).type);
						unify(fn.argumentTypes[i], s, "mda");
					}
					unify(fn.retType, typeToSort(md.returnType), "mdr");									
				} else if (sym instanceof ConstructorDecl) {
					ConstructorDecl md = (ConstructorDecl)sym;
					int off = arity - md.args.size();
					// the first argument can be 'state'
					// then we can have up to two 'alloc'
					Assert.notFalse(off <= 3);
					for (int i = 0; i < arity; ++i) {
						Sort s = sortTime;
						if (off == 3 && i == 0)
							s = sortRef;
						if (i >= off)
							s = typeToSort(md.args.elementAt(i - off).type);
						unify(fn.argumentTypes[i], s, "cda");
					}
					unify(fn.retType, sortRef, "mdr");
				} else {
					ErrorSet.error("unknown symbol stored in methodcall: " + sym.getClass());
				}
				//ErrorSet.caution("MethodCall " + fn);
				break;
				
			default:
				Integer itag = new Integer(m.getTag());
				if (fnSymbolsByTag.containsKey(itag))
				{
					Object v = fnSymbolsByTag.get(itag); 
					if (v instanceof FnSymbol)
						fn = (FnSymbol)v;
					else if (v instanceof PredSymbol)
						fn = (PredSymbol)v;
					else
						Assert.fail("pff");					
				} else {
					ErrorSet.fatal("translating old gc tree, methodName not recognized "
						+ TagConstants.toString(m.getTag()) );
				}
			}
			
			Term[] args = new Term[arity];
			
			for (int i = 0; i < arity; ++i)
				args[i] = transform((ASTNode)n.childAt(i+1));
			FnTerm res = new FnTerm(fn, args);
			res.tag = tag;
			
			return res;
			
		} else if (n instanceof PrimitiveType) { // javafe/Type
			// this means this variable represent a primitive java type like
			// string, boolean or Java.lang.Object
			
			PrimitiveType m = (PrimitiveType) n;
			String s = escjava.ast.TagConstants.toString(m.getTag());
			return symbolRef(s, sortType);
			
		} else if (n instanceof QuantifiedExpr) {
			QuantifiedExpr m = (QuantifiedExpr) n;
			
			//String s = TagConstants.toString(m.getTag());
			
			boolean universal = false;
			
			if (m.getTag() == TagConstants.FORALL)
				universal = true;
			else if (m.getTag() == TagConstants.EXISTS)
				universal = false;
			else
				Assert.fail("QuantifiedExpr, unhandled tag");
			
			QuantVariable[] vars = new QuantVariable[m.vars.size()];
			for (int i = 0; i < m.vars.size(); ++i) {
				GenericVarDecl v = m.vars.elementAt(i);
				vars[i] = new QuantVariable(v, UniqName.variable(v));
				quantifiedVars.push(vars[i]);
			}
			
			Term[][] pats;
			Term[] nopats;
			
			if (m.pats != null) {
				pats = new Term[1][];
				pats[0] = new Term[pats.length];
				for (int i = 0; i < pats.length; ++i)
					pats[0][i] = transform(m.pats.elementAt(i));
			} else { 
				pats = new Term[0][];
			}
			
			if (m.nopats != null) {
				nopats = new Term[m.nopats.size()];
				for (int i = 0; i < nopats.length; ++i)
					nopats[i] = transform(m.nopats.elementAt(i));
			}else {
				nopats = new Term[0];
			}
			
			Term body = transform(m.expr);
			
			if (m.rangeExpr != null) {
				Term range = transform(m.rangeExpr);
				if (range instanceof BoolLiteral && ((BoolLiteral)range).value == true)
				{}
				else {
					body = new FnTerm(universal ? symImplies : symAnd, new Term[] { range, body });
				}
			}
			
			for (int i = 0; i < m.vars.size(); ++i)
				quantifiedVars.pop();
			
			return new QuantTerm(universal, vars, body, pats, nopats);
			
		} else if (n instanceof SimpleName) {
			SimpleName m = (SimpleName) n;			
			
			// it seems that this node is under a TypeName node all the time
			// and that all the information is in the TypeName node.
			// that's why we don't create a new node here.
			
			ErrorSet.fatal("SimpleName: "+m);
			return null;
			
		} else if (n instanceof SubstExpr) {
			//SubstExpr m = (SubstExpr) n;
			
			ErrorSet.fatal("SubstExpr viewed and not handled");
			return null;
			
		} else if (n instanceof TypeDecl) {						
			TypeDecl m = (TypeDecl) n;
			// this represents a type
			
			String s = new String(m.id.chars);
			
			ErrorSet.fatal("ignored TypeDecl " + s);
			return null;
			
		} else if (n instanceof TypeExpr) {
			TypeExpr m = (TypeExpr) n;
			return transform(m.type);
			
		} else if (n instanceof Type) { // javafe/Type
			Type t = (Type) n;
			return symbolRef(UniqName.type(t), sortType);
			
		} else if (n instanceof VariableAccess) {
			VariableAccess m = (VariableAccess) n;
			for (int i = 0; i < quantifiedVars.size(); ++i) {
				QuantVariable q = (QuantVariable)quantifiedVars.elementAt(i);
				if (q.var == m.decl)
					return new QuantVariableRef(q);
			}
			
			Sort s = null;
			String name = UniqName.variable(m.decl);
			
			GenericVarDecl decl = m.decl;
			
			while (decl instanceof LocalVarDecl && ((LocalVarDecl)decl).source != null)
				decl = ((LocalVarDecl)decl).source;				
			
			if (decl instanceof FieldDecl) {
				FieldDecl d = (FieldDecl)decl;
				if (Modifiers.isStatic(d.getModifiers()))
					s = typeToSort(d.type);
				else
					s = registerMapSort(sortRef, typeToSort(d.type), sortField);
				//ErrorSet.caution("VariableAccess " + name + " -> " + s);
			} else if (decl instanceof LocalVarDecl || decl instanceof FormalParaDecl) {
				GenericVarDecl g = (GenericVarDecl)decl;
				s = typeToSort(g.type);
				//ErrorSet.caution("VariableAccess local " + name + " -> " + s);
			} else {
				warn("unknown decl in VariableAccess " + m.decl.getClass());
			}
			
			return symbolRef (name, s);
			
		} else {
			ErrorSet.fatal("unhandled tag " + TagConstants.toString(n.getTag()));
			return null;
		}
	}
	public Term transform(ASTNode n)
	{
		//ErrorSet.caution("enter " + TagConstants.toString(n.getTag()) + " " + n);
		Term t = doTransform(n);
		//ErrorSet.caution("exit " + TagConstants.toString(n.getTag()));
		return t;
	}

	/*@ non_null_by_default @*/
	//TODO: maybe update the doc...
	public class SortedBackPred extends BackPred 
	{
		ArrayList axioms = new ArrayList();
		ArrayList distinct = new ArrayList();
		
		/**
		 * Return the type-specific background predicate as a formula.
		 */

		public void genTypeBackPred(FindContributors scope,
									PrintStream proverStream)
		{
			distinct.add(transform(Types.booleanType));
			distinct.add(transform(Types.charType));
			distinct.add(transform(Types.byteType));
			distinct.add(transform(Types.shortType));
			distinct.add(transform(Types.intType));
			distinct.add(transform(Types.longType));
			distinct.add(transform(Types.floatType));
			distinct.add(transform(Types.doubleType));
			distinct.add(transform(Types.voidType));
			distinct.add(transform(escjava.tc.Types.typecodeType));
						
			// Print them out, and add their contribution to the BP. 
			Info.out("[TypeSig contributors for "
					+Types.printName(scope.originType)+":");
			for( Enumeration typeSigs = scope.typeSigs();
			typeSigs.hasMoreElements(); )
			{
				TypeSig sig2 = (TypeSig)typeSigs.nextElement();
				Info.out("    "+Types.printName( sig2 ));
				addContribution( sig2.getTypeDecl(), proverStream );
				distinct.add(transform(sig2));
			}
			Info.out("]");
			
			// Handle constant fields' contribution:
			for( Enumeration fields = scope.fields();
			fields.hasMoreElements(); ) {
				FieldDecl fd = (FieldDecl)fields.nextElement();
				if (!Modifiers.isFinal(fd.modifiers) || fd.init==null)
					continue;
				
				int loc = fd.init.getStartLoc();
				VariableAccess f = VariableAccess.make(fd.id, loc, fd);
				
				if (Modifiers.isStatic(fd.modifiers)) {
					genFinalInitInfo(fd.init, null, null, f, fd.type, loc, 
							proverStream);
				} else {
					LocalVarDecl sDecl = UniqName.newBoundVariable('s');
					VariableAccess s = TrAnExpr.makeVarAccess(sDecl, Location.NULL);
					genFinalInitInfo(fd.init, sDecl, s, GC.select(f, s), fd.type, 
							loc, proverStream);
				}
			}
		}
		
		protected void produce(/*@ nullable @*/GenericVarDecl sDecl, /*@ nullable @*/Expr s,
				/*@ non_null */ Expr e,
				/*@ non_null */ PrintStream proverStream) {
			if (sDecl != null) {
				Expr ant = GC.nary(TagConstants.REFNE, s, GC.nulllit);
				ExprVec nopats = ExprVec.make(1);
				nopats.addElement(ant);
				e = GC.forall(sDecl, GC.implies(ant, e), nopats);
			}
			
			axioms.add(transform(e));
		}
		
		/**
		 * Add to b the contribution from a particular TypeDecl, which is
		 * a formula.
		 */
		
		protected void addContribution(/*@ non_null */ TypeDecl d,
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
			
			saySuper(d, proverStream);
		}
		
		
		FnTerm fn(FnSymbol f, Term a1)
		{
			return new FnTerm(f, new Term[] { a1 });
		}
		
		FnTerm fn(FnSymbol f, Term a1, Term a2)
		{
			return new FnTerm(f, new Term[] { a1, a2 });
		}
		
		void say(Term t)
		{
			axioms.add(t);
		}
		
		/** Record in the background predicate that x is a subclass of y. */
		
		private void saySubClass( Type x, Type y,
								  PrintStream proverStream ) 
		{
			saySubType( x, y, proverStream );
			Term tx = transform(x), ty = transform(y);
			say(fn(symAnyEQ, fn(symAsChild, tx, ty), tx));
		}
		
		/** Record in the background predicate that x is a subtype of y. */
		
		private void saySubType( Type x, Type y,
								PrintStream proverStream )
		{
			say(fn(symTypeLE, transform(x), transform(y)));
		}
		
		private void saySuper(TypeDecl d, /*@ non_null*/ PrintStream proverStream)
		{
			Expr sig = GC.typeExpr(TypeCheck.inst.getSig(d));
			
            LocalVarDecl boundVar = UniqName.newBoundVariable("t");
            Expr boundVarAccess = VariableAccess.make(boundVar.id, d.locId, boundVar);

            Expr subt = GC.nary(TagConstants.TYPEEQ, boundVarAccess, sig);
            
			if( d instanceof ClassDecl ) {
				ClassDecl cd = (ClassDecl)d;
				
				if( cd.superClass != null ) {
					subt = GC.or(subt,
							GC.nary(TagConstants.TYPELE, GC.typeExpr(cd.superClass), boundVarAccess));
				}
			} else {
				subt = GC.or(subt,
						GC.nary(TagConstants.TYPEEQ, GC.typeExpr(Types.javaLangObject()), 
													 boundVarAccess));
			}
			
			for( int i=0; i<d.superInterfaces.size(); i++ ) {
				subt = GC.or(subt,
						GC.nary(TagConstants.TYPELE, GC.typeExpr(d.superInterfaces.elementAt(i)), 
													 boundVarAccess));
			}
			
			Expr pat = GC.nary(TagConstants.TYPELE, sig, boundVarAccess); 
            Expr body = GC.nary(TagConstants.BOOLEQ, pat, subt);
			say(transform(GC.forallwithpats(boundVar,body, ExprVec.make(new Expr[] { pat }))));
		}
		
		/** Record in the background predicate that x is final. */
		
		private void sayIsFinal( Type x,
				/*@ non_null */ PrintStream proverStream ) 
		{
			Expr sig = GC.typeExpr(x);
			
            LocalVarDecl boundVar = UniqName.newBoundVariable("t");
            Expr boundVarAccess = VariableAccess.make(boundVar.id, x.getStartLoc(), boundVar);
            
			Expr pat = GC.nary(TagConstants.TYPELE, boundVarAccess, sig); 
            Expr body = GC.nary(TagConstants.BOOLEQ, pat, GC.nary(TagConstants.TYPEEQ, boundVarAccess, sig));
			say(transform(GC.forallwithpats(boundVar,body, ExprVec.make(new Expr[] { pat }))));
		}
		
	}

	// dump	
	// jgc: made it public but this is the WRONG way. a new build method (simpler) should be
	//  added instead
	public /*@ nullable @*/EscNodeBuilder dumpBuilder;
	Hashtable fnTranslations = new Hashtable();
	final ArrayList stringConstants = new ArrayList();
	final ArrayList distinctSymbols = new ArrayList();
	
	Sort[] mapSorts(Sort[] s)
	{
		Sort[] res = new Sort[s.length];
		for (int i = 0; i < s.length; ++i)
			res[i] = mapSortTo(dumpBuilder, s[i]);
		return res;
	}
	
	SAny[] dumpArray(Term[] args)
	{
		SAny[] params = new SAny[args.length];
		for (int i = 0; i < args.length; ++i)
			params[i] = args[i].dumpAny();
		return params;
	}
	
	SPred[] dumpPredArray(Term[] args)
	{
		SPred[] params = new SPred[args.length];
		for (int i = 0; i < args.length; ++i)
			params[i] = args[i].dumpPred();
		return params;
	}
	
	STerm[] dumpTermArray(Term[] args)
	{
		STerm[] params = new STerm[args.length];
		for (int i = 0; i < args.length; ++i)
			params[i] = args[i].dump();
		return params;
	}
	
	Term toPred(Term body)
	{
		if (follow(body.getSort()) == sortBool)
			body = new FnTerm(symIsTrue, new Term[] { body });
		if (follow(body.getSort()) == sortValue)
			// TODO warning
			body = new FnTerm(symValueToPred, new Term[] { body });			
		unify(body.getSort(), sortPred, this);
		return body;		
}


	
}
