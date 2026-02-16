/* Copyright Hewlett-Packard, 2002 */

package escjava.sp;


import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import escjava.translate.GC;
import escjava.translate.UniqName;
import escjava.translate.Substitute;

import escjava.Main;

import javafe.util.Location;
import javafe.util.Assert;

import java.util.Hashtable;
import java.util.Enumeration;


public class DSA {

  public static GuardedCmd dsa(/*@ non_null */ GuardedCmd g) {
    VarMap[] out = new VarMap[2];
    return dsa( g, out );
  }

  public static GuardedCmd dsa(/*@ non_null */ GuardedCmd g, VarMap[] out) {
    RefInt preOrderCount;
    Hashtable lastVarUse;
    if (Main.lastVarUseOpt) {
      preOrderCount = new RefInt(0);
      lastVarUse = new Hashtable();  // mapping GenericVarDecl to RefInt
      //@ set lastVarUse.keyType = type(GenericVarDecl);
      //@ set lastVarUse.elementType = type(RefInt);
      computeLastVarUses(g, preOrderCount, lastVarUse);
      // reset the pre-order count
      preOrderCount.value = 0;
    } else {
      preOrderCount = null;
      lastVarUse = null;
    }
    return dsa(g, VarMap.identity(), out, "", preOrderCount, lastVarUse);
  }

  /** Parameters <code>preOrderCount</code> and <code>lastVarUse</code>
    * are used to perform a dead-variable analysis on variables, so that
    * merges of variables can be smaller.  These parameters should either
    * both be non-<code>null</code> or both be <code>null</code>.  If
    * they are <code>null</code>, the dead-variable analysis and optimization
    * will not be performed.
    **/

  //@ requires 2 <= out.length;
  //@ requires typeof(out) == type(VarMap[]);
  //@ requires (preOrderCount == null) == (lastVarUse == null);
  /*@ requires lastVarUse != null ==>
               lastVarUse.keyType == type(GenericVarDecl) &&
	       lastVarUse.elementType == type(RefInt); */
  //@ modifies out[0], out[1];
  //@ ensures out[0] != null && out[1] != null;
  private static GuardedCmd dsa(/*@ non_null */ GuardedCmd g,
				/*@ non_null */ VarMap map, 
				/*@ non_null */ VarMap[] out,
				String dynInstPrefix,
				RefInt preOrderCount,
				Hashtable lastVarUse) {
    if (map.isBottom()) {
      if (preOrderCount != null) {
	// Note that we must still update "preOrderCount" appropriately.
	doPreOrderCount(g, preOrderCount);
      }
      out[0] = map;
      out[1] = map;
      return GC.skip();
    }

    if (preOrderCount != null) {
      preOrderCount.value++;
    }

    switch (g.getTag()) {
    case TagConstants.SKIPCMD:
      /* dsa[[ Skip, M ]] ==
           Skip, M, bottom
      */
      out[0] = map;
      out[1] = VarMap.bottom();
      return g;

    case TagConstants.RAISECMD:
      /* dsa[[ Raise, M  ]] ==
           Raise, bottom, M
      */
      out[0] = VarMap.bottom();
      out[1] = map;
      return g;

    case TagConstants.YIELDCMD:
      // "dsa" only cares about the desugared form of yield.
      {
	YieldCmd yc = (YieldCmd)g;
	return dsa(yc.desugared, map, out, dynInstPrefix, preOrderCount, lastVarUse );
      }

    case TagConstants.LOOPCMD:
      // "dsa" only cares about the desugared form of loop.
      {
	LoopCmd lp= (LoopCmd)g;
	return dsa(lp.desugared, map, out, dynInstPrefix, preOrderCount, lastVarUse);
      }
    
    case TagConstants.CALL:
      // "dsa" only cares about the desugared form of call.
      {
	Call call= (Call)g;
	return dsa(call.desugared, map, out, dynInstPrefix, preOrderCount, lastVarUse );
      }

    case TagConstants.ASSERTCMD:
      /* dsa[[ Assert x, M ]] ==
	   Assert M[[e]], M, bottom
      */
      { 
	ExprCmd ec = (ExprCmd)g;
        out[0] = map;
	out[1] = VarMap.bottom();
	return ExprCmd.make(TagConstants.ASSERTCMD,
			    map.apply(ec.pred), ec.loc);
      }

    case TagConstants.ASSUMECMD:
      /* dsa[[ Assume x, M ]] ==
	   Assume M[[e]], M, bottom
      */
      { 
	ExprCmd ec = (ExprCmd)g;
        out[0] = map;
	out[1] = VarMap.bottom();
	return GC.assume(map.apply(ec.pred));
      }

    case TagConstants.GETSCMD:
    case TagConstants.SUBGETSCMD:
    case TagConstants.SUBSUBGETSCMD:
    case TagConstants.RESTOREFROMCMD:
	{
	  /* dsa[[ x = e, M ]] ==
               assume x' = M[[e]], M[x->x'], bottom
             dsa[[ x[e0] = e, M ]] ==
               assume x' = M[[ store(x, e0, e) ]], M[x->x'], bottom
             dsa[[ x[e0][e1] = e, M ]] ==
               assume x' = M[[ store(x, e0, store(select(x, e0), e1, e)) ]],
               M[x->x'], bottom
	  
             (where "x'" is a fresh inflected form of "x").
	  */

	  AssignCmd gc = (AssignCmd)g;

	  // Create the declaration for "x'".
	  LocalVarDecl xpDecl = UniqName.newIntermediateStateVar(gc.v, dynInstPrefix);
	  VariableAccess xpRef	= VariableAccess.make( gc.v.id, gc.v.loc,
						       xpDecl );

	  // Calculate the new value of "x'".
	  Expr nuv;
	  switch( g.getTag() ) {
	    case TagConstants.GETSCMD:
	    case TagConstants.RESTOREFROMCMD:
	      {
		nuv = gc.rhs;
		break;
	      }

	    case TagConstants.SUBGETSCMD:
	      {
		SubGetsCmd sgc = (SubGetsCmd)g;
		nuv = GC.nary(Location.NULL, Location.NULL,
			      TagConstants.STORE, sgc.v, sgc.index, sgc.rhs);
		break;
	      }

	    case TagConstants.SUBSUBGETSCMD:
	      {
		SubSubGetsCmd ssgc = (SubSubGetsCmd)g;
		
		Expr innermap = GC.nary(Location.NULL,
					Location.NULL,
					TagConstants.SELECT, ssgc.v, ssgc.index1);
		Expr newinnermap = GC.nary(Location.NULL,
					   Location.NULL,
					   TagConstants.STORE, innermap, ssgc.index2,
					   ssgc.rhs);
		nuv = GC.nary(Location.NULL,Location.NULL,
			      TagConstants.STORE, ssgc.v, ssgc.index1,
			      newinnermap);
		break;
	      }

	    default:
	      Assert.fail("Unreachable");
	      nuv = null; // dummy assignment
	  }

          out[1] = VarMap.bottom();
	  nuv = map.apply( nuv );
	  if( GC.isSimple( nuv ) ) {
	    out[0] = map.extend( gc.v.decl, nuv );
	    return GC.skip();
	  } else {
	    out[0] = map.extend(gc.v.decl, xpRef);
	    return GC.assume(GC.nary( gc.v.loc, gc.v.loc,
				      TagConstants.ANYEQ, xpRef, nuv ));
	  }
	}

    case TagConstants.VARINCMD:
      /* dsa[[ var v in S, M ]] ==
           S', N[v->v], X[v->v]
         where dsa[[ S, M ]] == S', N, X .
      */

      {
	VarInCmd vc = (VarInCmd)g;
        VarMap[] nx = new VarMap[2];

	Hashtable h1 = new Hashtable();
	Hashtable h2 = new Hashtable();
	for (int i = 0; i < vc.v.size(); i++) {
	    GenericVarDecl v = vc.v.elementAt(i);
	    LocalVarDecl decl = UniqName.newIntermediateStateVar(v, dynInstPrefix);
	    VariableAccess xpRef = VariableAccess.make( decl.id, decl.locId, decl );
	    h1.put(v, xpRef);
	    Expr oldExpr = (Expr) map.get(v);
	    if (oldExpr != null) {
		h2.put(v, oldExpr);
	    }
	}

        GuardedCmd sp = dsa(vc.g, map.extend(h1), nx, dynInstPrefix, preOrderCount, lastVarUse);

	out[0] = nx[0].unmap(vc.v).extend(h2);
	out[1] = nx[1].unmap(vc.v).extend(h2);

        return sp;
      }

    case TagConstants.DYNINSTCMD:
      {
	DynInstCmd dc = (DynInstCmd)g;
	
	return dsa(dc.g, map, out, dynInstPrefix + "-" + dc.s, preOrderCount, lastVarUse);
      }

    case TagConstants.SEQCMD:
      /* dsa[[ A ; B , M ]] ==
           let A', AN, AX == dsa[[ A , M ]]
               B', BN, BX == dsa[[ B , AN ]]
               AXR, BXR, X == merge(AX, BX)
           in
               ((A' ! (AXR ; raise)) ; (B' ! (BXR ; raise))), BN, X
      */
      {
	SeqCmd sc = (SeqCmd)g;
	GuardedCmd[] ap = new GuardedCmd[sc.cmds.size()];
	VarMap[] xmap = new VarMap[sc.cmds.size()];
	VarMap[] temp = new VarMap[2];

	for (int i = 0; i < sc.cmds.size(); i++) {
	  ap[i] = dsa(sc.cmds.elementAt(i), map, temp, dynInstPrefix, preOrderCount, lastVarUse);
	  map = temp[0];
	  xmap[i] = temp[1];
	}

	out[0] = map;
	GuardedCmdVec[] rename = new GuardedCmdVec[sc.cmds.size()];
	int p = (preOrderCount == null ? 0 : preOrderCount.value);
	out[1] = VarMap.merge(xmap, rename, sc.getStartLoc(), p, lastVarUse);
      
	GuardedCmdVec res = GuardedCmdVec.make(sc.cmds.size());
	for (int i = 0; i < sc.cmds.size(); i++) {
	  if (rename[i].size() > 0) {
	    ap[i] = GC.trycmd(ap[i], GC.seq(GC.seq(rename[i]),GC.raise()));
	  }
        }
	return GC.seq(GuardedCmdVec.make(ap));
      }

      case TagConstants.TRYCMD:
      /* dsa[[ A ! B , M ]] ==
           let A', AN, AX == dsa[[ A , M ]]
               B', BN, BX == dsa[[ B , AX ]]
               ANR, BNR, N == merge(AN, BN)
           in
               ((A' ; AN) ! (B' ; BN)), N, BX
      */
	{
	  CmdCmdCmd tc = (CmdCmdCmd)g;
          VarMap[] amaps = new VarMap[2];
          VarMap[] bmaps = new VarMap[2];
	  GuardedCmd ap = dsa(tc.g1, map, amaps, dynInstPrefix, preOrderCount, lastVarUse);
	  GuardedCmd bp = dsa(tc.g2, amaps[1], bmaps, dynInstPrefix, preOrderCount, lastVarUse);
	  out[1] = bmaps[1];
	  GuardedCmdVec[] rename = new GuardedCmdVec[2];
	  int p = (preOrderCount == null ? 0 : preOrderCount.value);
	  out[0] = VarMap.merge(amaps[0], bmaps[0], rename, tc.getStartLoc(),
				p, lastVarUse);
          return GC.trycmd( GC.seq(ap, GC.seq(rename[0])),
			    GC.seq(bp, GC.seq(rename[1])));
	}

      case TagConstants.CHOOSECMD:
      /* dsa[[ A [] B , M ]] ==
           let A', AN, AX == dsa[[ A , M ]]
               B', BN, BX == dsa[[ B , M ]]
               ANR, BNR, N == merge(AN, BN)
               AXR, BXR, X == merge(AX, BX)
           in
               (((A' ; AN) ! (AX ; raise)) [] ((B' ; BN) ! (BX ; raise))),
	       N, X
      */
	{
	  CmdCmdCmd cc = (CmdCmdCmd)g;
          VarMap[] amaps = new VarMap[2];
          VarMap[] bmaps = new VarMap[2];
	  GuardedCmd ap = dsa(cc.g1, map, amaps, dynInstPrefix, preOrderCount, lastVarUse);
	  GuardedCmd bp = dsa(cc.g2, map, bmaps, dynInstPrefix, preOrderCount, lastVarUse);

	  GuardedCmdVec[] nrename = new GuardedCmdVec[2];
	  GuardedCmdVec[] xrename = new GuardedCmdVec[2];
	  int p = (preOrderCount == null ? 0 : preOrderCount.value);
	  out[0] = VarMap.merge(amaps[0], bmaps[0], nrename, cc.getStartLoc(),
				p, lastVarUse);
	  out[1] = VarMap.merge(amaps[1], bmaps[1], xrename, cc.getStartLoc(),
				p, lastVarUse);
          return GC.choosecmd(GC.trycmd(GC.seq(ap, GC.seq(nrename[0])),
					GC.seq(GC.seq(xrename[0]), GC.raise())),
			      GC.trycmd(GC.seq(bp, GC.seq(nrename[1])),
					GC.seq(GC.seq(xrename[1]), GC.raise())));
	}

      default:
	//@ unreachable
	Assert.fail("Fall thru on "+g);
	return null;
    }
  }

  //@ requires lastVarUse.keyType == type(GenericVarDecl);
  //@ requires lastVarUse.elementType == type(RefInt);
  private static void computeLastVarUses(/*@ non_null */ GuardedCmd g,
					 /*@ non_null */ RefInt preOrderCount,
					 /*@ non_null */ Hashtable lastVarUse) {
    int id = preOrderCount.value;
    preOrderCount.value++;

    switch (g.getTag()) {
    case TagConstants.SKIPCMD:
    case TagConstants.RAISECMD:
      break;

    case TagConstants.YIELDCMD:
      {
	YieldCmd yc = (YieldCmd)g;
	computeLastVarUses(yc.desugared, preOrderCount, lastVarUse);
	break;
      }

    case TagConstants.LOOPCMD:
      {
	LoopCmd lp= (LoopCmd)g;
	computeLastVarUses(lp.desugared, preOrderCount, lastVarUse);
	break;
      }
    
    case TagConstants.CALL:
      {
	Call call= (Call)g;
	computeLastVarUses(call.desugared, preOrderCount, lastVarUse);
	break;
      }

    case TagConstants.ASSERTCMD:
    case TagConstants.ASSUMECMD:
      { 
	ExprCmd ec = (ExprCmd)g;
	RefInt pi = new RefInt(id);
	for (Enumeration e = Substitute.freeVars(ec.pred).elements();
	     e.hasMoreElements(); ) {
	  GenericVarDecl v = (GenericVarDecl)e.nextElement();
	  lastVarUse.put(v, pi);
	}
	break;
      }

    case TagConstants.GETSCMD:
    case TagConstants.SUBGETSCMD:
    case TagConstants.SUBSUBGETSCMD:
    case TagConstants.RESTOREFROMCMD:
	{
	  AssignCmd gc = (AssignCmd)g;

	  // Calculate the rhs of the assignment
	  Expr nuv;
	  switch( g.getTag() ) {
	    case TagConstants.GETSCMD:
	    case TagConstants.RESTOREFROMCMD:
	      {
		nuv = gc.rhs;
		break;
	      }

	    case TagConstants.SUBGETSCMD:
	      {
		SubGetsCmd sgc = (SubGetsCmd)g;
		nuv = GC.nary(Location.NULL, Location.NULL,
			      TagConstants.STORE, sgc.v, sgc.index, sgc.rhs);
		break;
	      }

	    case TagConstants.SUBSUBGETSCMD:
	      {
		SubSubGetsCmd ssgc = (SubSubGetsCmd)g;
		
		Expr innermap = GC.nary(Location.NULL,
					Location.NULL,
					TagConstants.SELECT, ssgc.v, ssgc.index1);
		Expr newinnermap = GC.nary(Location.NULL,
					   Location.NULL,
					   TagConstants.STORE, innermap, ssgc.index2,
					   ssgc.rhs);
		nuv = GC.nary(Location.NULL,Location.NULL,
			      TagConstants.STORE, ssgc.v, ssgc.index1,
			      newinnermap);
		break;
	      }

	    default:
	      Assert.fail("Unreachable");
	      nuv = null; // dummy assignment
	  }

	  RefInt pi = new RefInt(id);
	  for (Enumeration e = Substitute.freeVars(nuv).elements();
	       e.hasMoreElements(); ) {
	    GenericVarDecl v = (GenericVarDecl)e.nextElement();
	    lastVarUse.put(v, pi);
	  }
	  break;
	}

    case TagConstants.VARINCMD:
      {
	VarInCmd vc = (VarInCmd)g;
	computeLastVarUses(vc.g, preOrderCount, lastVarUse);
	break;
      }

    case TagConstants.DYNINSTCMD:
      {
	DynInstCmd dc = (DynInstCmd)g;
	computeLastVarUses(dc.g, preOrderCount, lastVarUse);
	break;
      }

    case TagConstants.SEQCMD:
      {
	SeqCmd sc = (SeqCmd)g;
	for (int i = 0; i < sc.cmds.size(); i++) {
	  computeLastVarUses(sc.cmds.elementAt(i), preOrderCount, lastVarUse);
	}
	break;
      }

      case TagConstants.TRYCMD:
      case TagConstants.CHOOSECMD:
	{
	  CmdCmdCmd tc = (CmdCmdCmd)g;
	  computeLastVarUses(tc.g1, preOrderCount, lastVarUse);
	  computeLastVarUses(tc.g2, preOrderCount, lastVarUse);
	  break;
	}

      default:
	//@ unreachable
	Assert.fail("Fall thru on "+g);
	break;
    }
  }

  private static void doPreOrderCount(/*@ non_null */ GuardedCmd g,
				      /*@ non_null */ RefInt preOrderCount) {
    preOrderCount.value++;

    switch (g.getTag()) {
    case TagConstants.SKIPCMD:
    case TagConstants.RAISECMD:
      break;

    case TagConstants.YIELDCMD:
      {
	YieldCmd yc = (YieldCmd)g;
	doPreOrderCount(yc.desugared, preOrderCount);
	break;
      }

    case TagConstants.LOOPCMD:
      {
	LoopCmd lp= (LoopCmd)g;
	doPreOrderCount(lp.desugared, preOrderCount);
	break;
      }
    
    case TagConstants.CALL:
      {
	Call call= (Call)g;
	doPreOrderCount(call.desugared, preOrderCount);
	break;
      }

    case TagConstants.ASSERTCMD:
    case TagConstants.ASSUMECMD:
      break;

    case TagConstants.GETSCMD:
    case TagConstants.SUBGETSCMD:
    case TagConstants.SUBSUBGETSCMD:
    case TagConstants.RESTOREFROMCMD:
      break;

    case TagConstants.VARINCMD:
      {
	VarInCmd vc = (VarInCmd)g;
	doPreOrderCount(vc.g, preOrderCount);
	break;
      }

    case TagConstants.DYNINSTCMD:
      {
	DynInstCmd dc = (DynInstCmd)g;
	doPreOrderCount(dc.g, preOrderCount);
	break;
      }

    case TagConstants.SEQCMD:
      {
	SeqCmd sc = (SeqCmd)g;
	for (int i = 0; i < sc.cmds.size(); i++) {
	  doPreOrderCount(sc.cmds.elementAt(i), preOrderCount);
	}
	break;
      }

      case TagConstants.TRYCMD:
      case TagConstants.CHOOSECMD:
	{
	  CmdCmdCmd tc = (CmdCmdCmd)g;
	  doPreOrderCount(tc.g1, preOrderCount);
	  doPreOrderCount(tc.g2, preOrderCount);
	  break;
	}

      default:
	//@ unreachable
	Assert.fail("Fall thru on "+g);
	break;
    }
  }
}
