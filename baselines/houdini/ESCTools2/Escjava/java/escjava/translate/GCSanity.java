/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;

/** This class contains a number of sanity checks of guarded commands.
 **/

import java.util.Enumeration;

import javafe.util.Set;
import javafe.util.Assert;

import javafe.ast.*;
import javafe.ast.GenericVarDecl;
import escjava.ast.*;
import escjava.ast.TagConstants;

import escjava.Main;


public class GCSanity {
  public static void check(/*@ non_null */ GuardedCmd g) {
    if (! Main.options().noVarCheckDeclsAndUses) {
      Set edci = new Set();
      Set cdni = new Set();
      Set euei = new Set();
      Set uuei = new Set();
      Set sp = new Set();
      checkDeclsAndUses(g, edci, cdni, euei, uuei, "", sp);
    }
  }

  /** Checks that there are no duplicate definitions of local
    * variables, including implicit outermost declarations and
    * considering dynamic inflections.  Also checks that
    * dynamic-inflection prefixes are unique in nesting, i.e. there
    * are no two dynamic commands with the same inflections where
    * one would be inside another.
    *
    * @param edci       ever declared with current inflection
    * @param cdni       currently declared with nonempty inflection
    * @param euei       ever used with empty inflection
    * @param uuei       unusable with empty inflection
    * @param inflection current inflection
    * @param sp         seen prefixes (except "")
    **/

  private static void checkDeclsAndUses(/*@ non_null */ GuardedCmd g,
					/*@ non_null */ Set edci,
					/*@ non_null */ Set cdni,
					/*@ non_null */ Set euei,
					/*@ non_null */ Set uuei,
					/*@ non_null */ String inflection,
					/*@ non_null */ Set sp) {
    switch (g.getTag()) {
    case TagConstants.SKIPCMD:
    case TagConstants.RAISECMD:
      break;

    case TagConstants.LOOPCMD:
      {
	LoopCmd lp = (LoopCmd)g;
	checkDeclsAndUses(lp.desugared,
			  edci, cdni, euei, uuei,
			  inflection, sp);
	break;
      }

    case TagConstants.CALL:
      {
	Call call = (Call)g;
	checkDeclsAndUses(call.desugared,
			  edci, cdni, euei, uuei,
			  inflection, sp);
	break;
      }

    case TagConstants.ASSERTCMD:
    case TagConstants.ASSUMECMD:
      {
	ExprCmd ec = (ExprCmd)g;
	checkUses(ec.pred, cdni, euei, uuei);
	break;
      }

    case TagConstants.GETSCMD:
    case TagConstants.RESTOREFROMCMD:
      {
	AssignCmd ac = (AssignCmd)g;
	checkUses(ac.v, cdni, euei, uuei);
	if (ac.rhs != null) checkUses(ac.rhs, cdni, euei, uuei);
	break;
      }

    case TagConstants.SUBGETSCMD:
      {
	SubGetsCmd ac = (SubGetsCmd)g;
	checkUses(ac.v, cdni, euei, uuei);
	checkUses(ac.index, cdni, euei, uuei);
	if (ac.rhs != null) checkUses(ac.rhs, cdni, euei, uuei);
	break;
      }

    case TagConstants.SUBSUBGETSCMD:
      {
	SubSubGetsCmd ac = (SubSubGetsCmd)g;
	checkUses(ac.v, cdni, euei, uuei);
	checkUses(ac.index1, cdni, euei, uuei);
	checkUses(ac.index2, cdni, euei, uuei);
	if (ac.rhs != null) checkUses(ac.rhs, cdni, euei, uuei);
	break;
      }

    case TagConstants.VARINCMD:
      {
	VarInCmd vc = (VarInCmd)g;
	boolean emptyInflection = inflection.length() == 0;
	/*-@ uninitialized */ boolean[] previouslyInCdni = null;
	if (! emptyInflection) {
	  previouslyInCdni = new boolean[vc.v.size()];
	}
	for (int i = 0; i < vc.v.size(); i++) {
	  GenericVarDecl vd = (GenericVarDecl)vc.v.elementAt(i);
	  Assert.notFalse(! edci.add(vd));
	  if (emptyInflection) {
	    Assert.notFalse(! euei.contains(vd));
	  } else {
	    previouslyInCdni[i] = cdni.add(vd);
	  }
	}
	checkDeclsAndUses(vc.g, edci, cdni, euei, uuei,
			  inflection, sp);
	for (int i = 0; i < vc.v.size(); i++) {
	  GenericVarDecl vd = (GenericVarDecl)vc.v.elementAt(i);
	  if (emptyInflection) {
	    uuei.add(vd);
	  } else if (! previouslyInCdni[i]) {
	    cdni.remove(vd);
	  }
	}
	break;
      }

    case TagConstants.DYNINSTCMD:
        {
            DynInstCmd dc = (DynInstCmd)g;
            Set edciNew = new Set();
            String infl = inflection + "-" + dc.s;
            String internInfl = infl.intern();
            Assert.notFalse(!sp.add(internInfl));
            checkDeclsAndUses(dc.g, edciNew, cdni, euei, uuei, infl, sp);
            sp.remove(internInfl);
            break;
        }

    case TagConstants.SEQCMD:
      {
	SeqCmd sc = (SeqCmd)g;
	for (int i = 0; i < sc.cmds.size(); i++) {
	  checkDeclsAndUses(sc.cmds.elementAt(i),
			    edci, cdni, euei, uuei,
			    inflection, sp);
	}
	break;
      }

      case TagConstants.TRYCMD:
      case TagConstants.CHOOSECMD:
	{
	  CmdCmdCmd tc = (CmdCmdCmd)g;
	  checkDeclsAndUses(tc.g1, edci, cdni, euei, uuei,
			    inflection, sp);
	  checkDeclsAndUses(tc.g2, edci, cdni, euei, uuei,
			    inflection, sp);
	  break;
	}

      default:
	//@ unreachable;
	Assert.fail("Fall thru on "+g);
	break;
    }
  }

  private static void checkUses(/*@ non_null */ Expr e,
				/*@ non_null */ Set cdni,
				/*@ non_null */ Set euei,
				/*@ non_null */ Set uuei) {
    for (Enumeration freeVars = Substitute.freeVars(e).elements();
	 freeVars.hasMoreElements(); ) {
      GenericVarDecl v = (GenericVarDecl)freeVars.nextElement();
      if (! cdni.contains(v)) {
	Assert.notFalse(! uuei.contains(v));
	euei.add(v);
      }
    }
  }
}
