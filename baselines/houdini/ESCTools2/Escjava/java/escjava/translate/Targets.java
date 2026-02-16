/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.util.Set;
import javafe.util.Assert;

import escjava.ast.*;
import escjava.ast.TagConstants;


/**
 ** Provides methods for computing various kinds of syntactic targets
 **/

public final class Targets {
  
  /** Returns the set of normal targets of <code>gc</code>.
    * In ESCJ 16 speak, this method returns <code>NTargets[[ gc, {} ]]</code>.
    **/

  //@ ensures \result != null;
  public static Set normal(/*@ non_null */ GuardedCmd gc) {
    // use the SimpleTargets option (see ESCJ 16)
    Set answer = new Set();
    simpleTargets(gc, answer, true);
    return answer;
  }

  /** Returns the set of variables that are direct normal targets in
    * <code>gc</code>, that is, that are modified in some assignment
    * statement, not call statement, in <code>gc</code>.
    **/

  //@ ensures \result != null;
  public static Set direct(/*@ non_null */ GuardedCmd gc) {
    // use the SimpleTargets option (see ESCJ 16)
    Set answer = new Set();
    simpleTargets(gc, answer, false);
    return answer;
  }

  /** Adds <code>SimpleTargets[[ gc ]]</code> (as defined in ESCJ 16)
    * to <code>set</code>.  Assumes that no local variable declared in
    * <code>gc</code> is in <code>set</code>.  If
    * <code>includeCallTargets</code> is <code>true</code>, the targets of
    * calls are included; otherwise, they are not.
    **/

  private static void simpleTargets(/*@ non_null */ GuardedCmd gc, Set set,
				    boolean includeCallTargets) {
    int tag = gc.getTag();
    switch (tag) {
    case TagConstants.SKIPCMD:
    case TagConstants.RAISECMD:
    case TagConstants.ASSERTCMD:
    case TagConstants.ASSUMECMD:
      return;

    case TagConstants.GETSCMD: {
      GenericVarDecl vd = ((GetsCmd)gc).v.decl;
      set.add(vd);
      return;
    }

    case TagConstants.SUBGETSCMD: {
      GenericVarDecl vd = ((SubGetsCmd)gc).v.decl;
      set.add(vd);
      return;
    }

    case TagConstants.SUBSUBGETSCMD: {
      GenericVarDecl vd = ((SubSubGetsCmd)gc).v.decl;
      set.add(vd);
      return;
    }

    case TagConstants.RESTOREFROMCMD: {
      // no targets
      return;
    }

    case TagConstants.VARINCMD: {
      VarInCmd vc = (VarInCmd)gc;
      simpleTargets(vc.g, set, includeCallTargets);
      for (int i = 0; i < vc.v.size(); i++) {
	GenericVarDecl vd = vc.v.elementAt(i);
	set.remove(vd);
      }
      return;
    }

    case TagConstants.DYNINSTCMD: {
      DynInstCmd dc = (DynInstCmd)gc;
      simpleTargets(dc.g, set, includeCallTargets);
      return;
    }

    case TagConstants.SEQCMD: {
      SeqCmd sc = (SeqCmd)gc;
      int len = sc.cmds.size();
      for (int i = 0; i < len; i++) {
	simpleTargets(sc.cmds.elementAt(i), set, includeCallTargets);
      }
      return;
    }

    case TagConstants.CALL: {
	Call call = (Call)gc;
	if (call.inlined) {
	    simpleTargets(call.desugared, set, includeCallTargets);
	} else if (includeCallTargets) {
	    Enumeration e = call.spec.preVarMap.keys();
	    while (e.hasMoreElements()) {
		set.add(((GenericVarDecl)e.nextElement()));
	    }
	}
	return;
    }

    case TagConstants.TRYCMD: {
      CmdCmdCmd tc = (CmdCmdCmd)gc;
      simpleTargets(tc.g1, set, includeCallTargets);
      simpleTargets(tc.g2, set, includeCallTargets);
      return;
    }

    case TagConstants.LOOPCMD: {
      LoopCmd lp = (LoopCmd)gc;
      simpleTargets(lp.guard, set, includeCallTargets);
      simpleTargets(lp.body, set, includeCallTargets);
      for (int i = 0; i < lp.tempVars.size(); i++) {
	GenericVarDecl vd = lp.tempVars.elementAt(i);
	set.remove(vd);
      }
      return;
    }

    case TagConstants.CHOOSECMD: {
      CmdCmdCmd cc = (CmdCmdCmd)gc;
      simpleTargets(cc.g1, set, includeCallTargets);
      simpleTargets(cc.g2, set, includeCallTargets);
      return;
    }

    default:
      //@ unreachable;
      Assert.fail("UnknownTag<" + tag + ":" +
		  TagConstants.toString(tag) + ">");
      return;
    }
  }
  
}

