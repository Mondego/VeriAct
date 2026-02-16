package escjava.sortedProver.auflia;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.regex.Pattern;

import javafe.util.Assert;
import escjava.prover.Atom;
import escjava.sortedProver.EscNodeBuilder;
import escjava.sortedProver.NodeBuilder.FnSymbol;
import escjava.sortedProver.NodeBuilder.PredSymbol;
import escjava.sortedProver.NodeBuilder.SPred;
import escjava.sortedProver.NodeBuilder.Sort;

/*@ non_null_by_default @*/
public class AufliaNodeBuilder extends EscNodeBuilder
{
	StringBuffer extrafuns;
	Hashtable nameMap;
	Hashtable revNameMap;
	HashSet funsDefined;
		
	/*@ non_null_by_default @*/
	public class Sx implements SMap, SBool, SInt, SReal, SPred
	{
		private static final int LIMIT = 100;
		public final String head;
		public final Sx[] args;
		public final int size;
		public final int depth;
		
		public Sx(String name, Sx[] args)
		{
			this.head = name.equals("_array") ? "array_" : name;
			this.args = args;
			int depth = 0;
			int size = name.length() + 1;
			for (int i = 0; i < args.length; ++i) {
				if (args[i].depth > depth) depth = args[i].depth;
				size += args[i].size + 1;
			}
			depth++;
			this.depth = depth;
			this.size = size;
			if (name.equals("and") || name.equals("or"))
				Arrays.sort(args, new Comparator() {
					public int compare(Object a, Object b) {
						return ((Sx)a).compareTo((Sx)b);
					}
				});
		}
		
		public int compareTo(Sx other)
		{
			if (this == other) return 0;
			if (this.depth != other.depth) return this.depth - other.depth;
			if (this.size != other.size) return this.size - other.size;
			if (this.args.length != other.args.length) return this.args.length - other.args.length;
			if (!this.head.equals(other.head)) this.head.compareTo(other.head);
			for (int i = 0; i < args.length; ++i) {
				int res = args[i].compareTo(other.args[i]);
				if (res != 0) return res;
			}
			return 0;
		}

		public boolean isSubSortOf(Sort s)
		{
			return true;
		}
		
		public int length(int limit)
		{
			if (limit < 0) return 1000;
			
			int sz = head.length() + 2;
			for (int i = 0; i < args.length; ++i)
				sz += 1 + args[i].length(limit - sz);
			
			return sz;
		}
		
		public void dump(int ind, StringBuffer sb)
		{
			String head = this.head;
			
			for (int i = 0; i < ind; ++i)
				sb.append(' ');
			
			if (head.startsWith(":::")) {
				sb.append(head.substring(2)).append(" {");
				for (int i = 0; i < args.length; ++i) {
					sb.append(" ");
					args[i].dump(-1, sb);
				}
				sb.append(" } ");
				if (ind >= 0) sb.append("\n");
				return;
			}
			
			if (head != "") head = encodeName(head);
			
			if (args.length == 0) {
				sb.append(head);
			} else {
				sb.append("(").append(head);
				int limit = LIMIT - ind;
				if (ind < 0 || length(limit) <= limit /*|| head.equals("FORALL")*/) {
					for (int i = 0; i < args.length; ++i) {
						sb.append(" ");
						args[i].dump(-1, sb);
					}
				} else {
					sb.append("\n");
					for (int i = 0; i < args.length; ++i) {						
						args[i].dump(ind + 1, sb);
					}					
					for (int i = 0; i < ind; ++i)
						sb.append(" ");
				}
				sb.append(")");
			}
			if (ind >= 0) sb.append("\n");
		}		
	}
	
	protected Sx sx(String name, STerm[] args)
	{
		//if (args instanceof Sx[])
		//	return new Sx(name, (Sx[])args);
		
		Sx[] temp = new Sx[args.length];
		for (int i = 0; i < args.length; ++i)
			temp[i] = (Sx)args[i];
		return new Sx(name, temp);		
	}	
	protected Sx sx(String name) { return new Sx(name, new Sx[0]); }	
	protected Sx sx(String name, STerm a1) { return new Sx(name, new Sx[] { (Sx)a1 }); }	
	protected Sx sx(String name, STerm a1, STerm a2) { return new Sx(name, new Sx[] { (Sx)a1, (Sx)a2 }); }
	
	protected Sx trueConst = sx("Smt.true");
	protected Sx intConst = sx("Int");
	
	protected void earlyInit()
	{
		if (extrafuns == null) {
			nameMap = new Hashtable();
			revNameMap = new Hashtable();
			extrafuns = new StringBuffer();
			funsDefined = new HashSet();
		}
	} 
	
	private boolean isLetter(char c)
	{
		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');		
	}
	
	private String encodeName(String orig_s)
	{
		String s = orig_s;
		if (nameMap.containsKey(s))
			return (String)nameMap.get(s);
	
		if (s.equals("<:")) s = "subtypes";
		
		StringBuffer sb = new StringBuffer();
		if (s.charAt(0) == '|')
			s = s.substring(1, s.length() - 1);
		int start = 0;
		if (s.charAt(0) == '?') {  }
		switch (s.charAt(0)) {
		case '?': 
			sb.append('?'); 
			start++;
			if (!isLetter(s.charAt(start)))
				sb.append("a'");
			break;
		case '<':
		case '>':
		case '=':
		case '-':
		case '+':
		case '*':
			nameMap.put(s,s);
			revNameMap.put(s,s);
			return s;
		default:
			if (!isLetter(s.charAt(start)) && !number.matcher(s).matches())
				s = "a'" + s;
			break;
		}
		
		for (int i = start; i < s.length(); ++i) {
			char c = s.charAt(i);
			if (isLetter(c) || (c >= '0' && c <= '9') || c == '.')
				sb.append(c);
			else
				sb.append('_');			
		}
		
		s = sb.toString();
		String res = s;
		int i = 1;
		while (revNameMap.containsKey(res)) {
			res = s + "." + i;
			i++;
		}
		nameMap.put(orig_s, res);
		revNameMap.put(res, orig_s);
		return res;
	}
	
	public String dump(SPred p)
	{
		StringBuffer sb = new StringBuffer();
		
		((Sx)p).dump(0, sb);
		return sb.toString();
	}
	
	public SAny buildFnCall(FnSymbol fn, SAny[] args)
	{
		return sx(fn.name, args);
	}

	public SAny buildConstantRef(FnSymbol c) 
	{
		return sx(c.name);
	}

	public SAny buildQVarRef(QuantVar v)
	{
		return sx("?" + v.name);
	}

	public SPred buildPredCall(PredSymbol fn, SAny[] args)
	{
		if (fn == symRefNE || fn == symTypeNE) 
			return sx("not", sx("=", args));
		
		else if (fn == symRefEQ || fn == symTypeEQ) 
			return sx("=", args);
		
		//if (fn == symAllocLT || fn == symAllocLE || fn == symTypeLE || fn == symArrayFresh ||
			//	fn == symIsAllocated)
		if (fn.retType == sortPred)
			return sx(fn.name, args);
			
		return sx("=", sx(fn.name, args), trueConst);
	}
	
	public PredSymbol registerPredSymbol(String name, Sort[] args, int tag)
	{
		earlyInit();
		String ename = encodeName(name);
		if (! AufliaPrelude.predefined.contains(ename)) {
			extrafuns.append(":extrapreds (( ").append(ename);
			for (int i = 0; i < args.length; ++i)
				extrafuns.append(" Int");
			extrafuns.append(" ))\n");
		}
		
		return super.registerPredSymbol(name, args, tag);
	}
	
	public FnSymbol registerFnSymbol(String name, Sort[] args, Sort ret_type, int tag)
	{
		earlyInit();
		String ename = encodeName(name);
		if (! AufliaPrelude.predefined.contains(ename)) {
			/*
			if (name.indexOf("intern") >= 0) {
				int x = 0;
				x++;
			}*/
			extrafuns.append(":extrafuns (( ").append(ename);
			for (int i = 0; i < args.length + 1; ++i)
				extrafuns.append(" Int");
			extrafuns.append(" ))\n");
		}
		
		return super.registerFnSymbol(name, args, ret_type, tag);
	}

	public SPred buildImplies(SPred arg1, SPred arg2) { return sx("implies", arg1, arg2); }
	public SPred buildIff(SPred arg1, SPred arg2) { return sx("iff", arg1, arg2); }
	public SPred buildXor(SPred arg1, SPred arg2) { return sx("not", sx("iff", arg1, arg2)); }
	public SPred buildAnd(SPred[] args) { return sx("and", args); }
	public SPred buildOr(SPred[] args) { return sx("or", args); }
	public SPred buildNot(SPred arg) { return sx("not", arg); }
	
	Hashtable labelMap = new Hashtable();
	
	public SPred buildLabel(boolean positive, String name, SPred pred)
	{
		String evname = (positive ? "ErrVarPos_" : "ErrVarNeg_") + name;
		String ename = encodeName(evname);
		labelMap.put(ename, name);
		if (!labelMap.containsKey(ename)) {
			extrafuns.append(":extrapreds (( ").append(ename).append(" ))\n");
			funsDefined.add(evname);
		}
		return sx(positive ? "and" : "or", sx(evname), pred);
	}
	
	public String errVarToLabel(String name)
	{
		return (String)labelMap.get(name);
	}


	public SValue buildITE(SPred cond, SValue then_part, SValue else_part)
	{
		return sx("ite", new STerm[] { cond, then_part, else_part });		
	}

	private SPred buildQuant(String name, QuantVar[] vars, SPred body, STerm[][] pats, STerm[] nopats)
	{
		if (pats != null && pats.length == 0) pats = null;		
		if (nopats != null && nopats.length == 0) nopats = null;
		
		ArrayList args = new ArrayList();
		int p = 0;
		for (int i = 0; i < vars.length; ++i)	
			args.add(sx("?" + vars[i].name, intConst));
	
		args.add(body);
		
		if (nopats != null)
			for (int i = 0; i < nopats.length; ++i)
				args.add(sx(":::nopat", nopats[i]));
		
		if (pats != null)
			for (int i = 0; i < pats.length; ++i)
				args.add(sx(":::pat", pats[i]));
		
		STerm[] args2 = new STerm[args.size()];
		for (int i = 0; i < args2.length; ++i)
			args2[i] = (STerm) args.get(i);
		return sx(name, args2);
	}

	public SPred buildForAll(QuantVar[] vars, SPred body, STerm[][] pats,
			STerm[] nopats) 
	{
		return buildQuant("forall", vars, body, pats, nopats);
	}
		
	public SPred buildExists(QuantVar[] vars, SPred body)
	{
		return buildQuant("exists", vars, body, null, null);		
	}
	
	private Sx[] getVars(QuantVar[] vars)
	{
		Sx[] vs = new Sx[vars.length];
		for (int i = 0; i < vars.length; ++i)
			vs[i] = sx(vars[i].name);
		return vs;
	}

	public SPred buildIntPred(int intPredTag, SInt arg1, SInt arg2) {
		String sym;
		
		switch (intPredTag) {
		case predGE: sym = ">="; break;
		case predLE: sym = "<="; break;
		case predGT: sym = ">"; break;
		case predLT: sym = "<"; break;
		case predEQ: sym = "="; break;
		case predNE: return sx("not", sx("=", arg1, arg2));
		default: Assert.fail(""); sym = null;
		}
		
		return sx(sym, arg1, arg2);
	}
	
	private static Pattern number = Pattern.compile("[\\-0-9]+");
	private boolean isConst(Sx s)
	{		
		return (s.args.length == 0 && number.matcher(s.head).matches());
	}

	public SInt buildIntFun(int intFunTag, SInt arg1, SInt arg2)
	{
		String sym;
		
		switch (intFunTag) {
		case funADD: sym = "+"; break;
		case funSUB: sym = "-"; break;
		case funMUL:
			if (isConst((Sx)arg1) || isConst((Sx)arg2))
				sym = "*";
			else				
				sym = "integralMul";
			break;
		case funDIV: sym = "integralDiv"; break;
		case funMOD: sym = "integralMod"; break;
		case funASR32: sym = "intShiftAR"; break;
		case funSL32: sym = "intShiftL"; break;
		case funUSR32: sym = "intShiftUR"; break;
		case funASR64: sym = "longShiftAR"; break;
		case funSL64: sym = "longShiftL"; break;
		case funUSR64: sym = "longShiftUR"; break;
		default: Assert.fail(""); sym = null;
		}
		
		return sx(sym, arg1, arg2);
	}

	public SPred buildRealPred(int realPredTag, SReal arg1, SReal arg2) 
	{
		String sym;
		
		switch (realPredTag) {
		case predGE: sym = "floatingGE"; break;
		case predLE: sym = "floatingLE"; break;
		case predGT: sym = "floatingGT"; break;
		case predLT: sym = "floatingLT"; break;
		case predEQ: sym = "floatingEQ"; break;
		case predNE: sym = "floatingNE"; break;
		default: Assert.fail(""); sym = null;
		}
		
		return sx("=", sx(sym, arg1, arg2), trueConst);
	}

	public SReal buildRealFun(int realFunTag, SReal arg1, SReal arg2) 
	{
		String sym;
		
		switch (realFunTag) {
		case funADD: sym = "floatingADD"; break;
		case funSUB: sym = "floatingSUB"; break;
		case funMUL: sym = "floatingMUL"; break;
		case funDIV: sym = "floatingDiv"; break;
		case funMOD: sym = "floatingMOD"; break;
		default: Assert.fail(""); sym = null;
		}
		
		return sx(sym, arg1, arg2);
	}

	public SInt buildIntFun(int intFunTag, SInt arg1)
	{		
		switch (intFunTag) {
		case funNEG: return sx("~", arg1); 
		default: Assert.fail(""); return null;
		}
	}

	public SReal buildRealFun(int realFunTag, SReal arg1)
	{
		switch (realFunTag) {
		case funNEG: return sx("floatingNEG", arg1); 
		default: Assert.fail(""); return null;
		}
	}

	public SBool buildBool(boolean b)
	{
		if (b) return trueConst;
		else return sx("Smt.false");
	}

	public SInt buildInt(long n)
	{
		return sx(String.valueOf(n));
	}

	public SReal buildReal(double f) 	
	{
		String name = "F_" + f ;
		if (!funsDefined.contains(name)) {
			extrafuns.append(":extrafuns (( ").append(encodeName(name)).append(" Int ))\n");
			funsDefined.add(name);
		}
		return sx(name);
	}


	public SRef buildNull() 
	{
		return sx("null");
	}

	public SValue buildSelect(SMap map, SValue idx) 
	{
		return sx("select1", map, idx);
	}

	public SMap buildStore(SMap map, SValue idx, SValue val)
	{
		return sx("store1", new SAny[] { map, idx, val });
	}

	public SPred buildAnyEQ(SAny arg1, SAny arg2)
	{
		return sx("=", arg1, arg2);
	}

	public SPred buildAnyNE(SAny arg1, SAny arg2) 
	{
		return sx("not", sx("=", arg1, arg2));
	}

	public SValue buildValueConversion(Sort from, Sort to, SValue val) 
	{
		return val;
	}

	public SPred buildIsTrue(SBool val) 
	{
		return sx("=", trueConst, val);
	}
	
	public SPred buildDistinct(SAny[] terms)
	{
		return sx("distinct", terms);
	}
	
	public SPred buildTrue()
	{
		return sx("true");
	}
	
	public SBool buildIntBoolFun(int intFunTag, SInt arg1, SInt arg2) {
		return (SBool) buildIntPred(intFunTag, arg1, arg2);
	}  
  public SBool buildRefBoolFun(int intFunTag, SRef arg1, SRef arg2) {
    return buildIntBoolFun(intFunTag, (SInt)arg1, (SInt)arg2);
  }
	public SBool buildRealBoolFun(int realPredTag, SReal arg1, SReal arg2) {
		return (SBool) buildRealPred(realPredTag, arg1, arg2);
	}
	
	// Mobius stuff
	public SPred buildNewObject(SMap oldh, SAny type, SMap heap, SValue r) {
		throw new UnsupportedOperationException();
	}
	public SAny buildSort(Sort s) {
		throw new UnsupportedOperationException();
	}
	public SValue buildDynSelect(SMap map, SValue obj, SAny field) {
		throw new UnsupportedOperationException();
	}
  public SRef buildDynLoc(SMap map, SValue obj, SAny field) {
    throw new UnsupportedOperationException();
  }
	public SMap buildDynStore(SMap map, SValue obj, SAny field, SValue val) {
		throw new UnsupportedOperationException();
  }
	public SValue buildArrSelect(SMap map, SRef obj, SInt idx) {
		throw new UnsupportedOperationException();
	}
	public SMap buildArrStore(SMap map, SRef obj, SInt idx, SValue val) {
		throw new UnsupportedOperationException();
	}
	public SPred buildNewArray(SMap oldh, SAny type, SMap heap, SRef r, SInt len) {
		throw new UnsupportedOperationException();
	}
	public SPred buildAssignCompat(SMap map, SValue val, SAny type) {
		throw new UnsupportedOperationException();
	}
  public SPred buildInv(SMap map, SValue val, SAny type) {
    throw new UnsupportedOperationException();
  }
  public SPred buildIsAlive(SMap map, SRef ref) {
    throw new UnsupportedOperationException();
  }
  public SPred buildAssignPred(SMap map, SMap map_pre, SRef target, SRef loc) {
    throw new UnsupportedOperationException();
  }

}
