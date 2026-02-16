/* Copyright 2000, 2001, Compaq Computer Corporation */

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
import javafe.util.Set;

import java.util.Hashtable;
import java.util.Enumeration;


public class DSA {

  public static GuardedCmd dsa(/*@ non_null */ GuardedCmd g) {
    VarMapPair out = new VarMapPair();
    return dsa( g, out );
  }

  public static GuardedCmd dsa(/*@ non_null */ GuardedCmd g, VarMapPair out) {
    RefInt preOrderCount;
    Hashtable lastVarUse;
    if (Main.options().lastVarUseOpt) {
      preOrderCount = new RefInt(0);
      lastVarUse = new Hashtable();  // mapping GenericVarDecl to RefInt
      //+@ set lastVarUse.keyType = \type(GenericVarDecl);
      //+@ set lastVarUse.elementType = \type(RefInt);
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

  //@ requires (preOrderCount == null) == (lastVarUse == null);
  /*+@ requires lastVarUse != null ==>
               lastVarUse.keyType == \type(GenericVarDecl) &&
	       lastVarUse.elementType == \type(RefInt); */
  //@ modifies out.n, out.x;
  //@ ensures out.n != null && out.x != null;
  private static GuardedCmd dsa(/*@ non_null */ GuardedCmd g,
				/*@ non_null */ VarMap map, 
				/*@ non_null */ VarMapPair out,
				String dynInstPrefix,
				RefInt preOrderCount,
				Hashtable lastVarUse) {
    if (map.isBottom()) {
      if (preOrderCount != null) {
	// Note that we must still update "preOrderCount" appropriately.
	doPreOrderCount(g, preOrderCount);
      }
      out.n = map;
      out.x = map;
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
      out.n = map;
      out.x = VarMap.bottom();
      return g;

    case TagConstants.RAISECMD:
      /* dsa[[ Raise, M  ]] ==
           Raise, bottom, M
      */
      out.n = VarMap.bottom();
      out.x = map;
      return g;

    case TagConstants.LOOPCMD:
      // "dsa" only cares about the desugared form a loop.
      {
	LoopCmd lp= (LoopCmd)g;
	return dsa(lp.desugared, map, out, dynInstPrefix, preOrderCount, lastVarUse);
      }
    
    case TagConstants.CALL:
      // "dsa" only cares about the desugared form a call.
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
        out.n = map;
	out.x = VarMap.bottom();
	return ExprCmd.make(TagConstants.ASSERTCMD,
			    map.apply(ec.pred), ec.loc);
      }

    case TagConstants.ASSUMECMD:
      /* dsa[[ Assume x, M ]] ==
	   Assume M[[e]], M, bottom
      */
      { 
	ExprCmd ec = (ExprCmd)g;
        out.n = map;
	out.x = VarMap.bottom();
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
	  Expr nuv = null;
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
		if (sgc.rhs == null) break;
		nuv = GC.nary(Location.NULL, Location.NULL,
			      TagConstants.STORE, sgc.v, sgc.index, sgc.rhs);
		break;
	      }

	    case TagConstants.SUBSUBGETSCMD:
	      {
		SubSubGetsCmd ssgc = (SubSubGetsCmd)g;
		if (ssgc.rhs == null) break;
		
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

          out.x = VarMap.bottom();
	  if (nuv == null) {
	    out.n = map.extend( gc.v.decl, xpRef);
	    return GC.skip();
	  } else if( GC.isSimple( nuv ) ) {
	    nuv = map.apply( nuv );
	    out.n = map.extend( gc.v.decl, nuv );
	    return GC.skip();
	  } else {
	    nuv = map.apply( nuv );
	    out.n = map.extend(gc.v.decl, xpRef);
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
        VarMapPair nx = new VarMapPair();

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

	out.n = nx.n.unmap(vc.v).extend(h2);
	out.x = nx.x.unmap(vc.v).extend(h2);

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
	VarMapPair temp = new VarMapPair();

	for (int i = 0; i < sc.cmds.size(); i++) {
	  ap[i] = dsa(sc.cmds.elementAt(i), map, temp, dynInstPrefix, preOrderCount, lastVarUse);
	  map = temp.n;
	  xmap[i] = temp.x;
	}

	out.n = map;
	GuardedCmdVec[] rename = new GuardedCmdVec[sc.cmds.size()];
	int p = (preOrderCount == null ? 0 : preOrderCount.value);
	out.x = VarMap.merge(xmap, rename, sc.getStartLoc(), p, lastVarUse);
      
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
          VarMapPair amaps = new VarMapPair();
          VarMapPair bmaps = new VarMapPair();
	  GuardedCmd ap = dsa(tc.g1, map, amaps, dynInstPrefix, preOrderCount, lastVarUse);
	  GuardedCmd bp = dsa(tc.g2, amaps.x, bmaps, dynInstPrefix, preOrderCount, lastVarUse);
	  out.x = bmaps.x;
	  GuardedCmdVec[] rename = new GuardedCmdVec[2];
	  int p = (preOrderCount == null ? 0 : preOrderCount.value);
	  out.n = VarMap.merge(amaps.n, bmaps.n, rename, tc.getStartLoc(),
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
          VarMapPair amaps = new VarMapPair();
          VarMapPair bmaps = new VarMapPair();
	  GuardedCmd ap = dsa(cc.g1, map, amaps, dynInstPrefix, preOrderCount, lastVarUse);
	  GuardedCmd bp = dsa(cc.g2, map, bmaps, dynInstPrefix, preOrderCount, lastVarUse);

	  GuardedCmdVec[] nrename = new GuardedCmdVec[2];
	  GuardedCmdVec[] xrename = new GuardedCmdVec[2];
	  int p = (preOrderCount == null ? 0 : preOrderCount.value);
	  out.n = VarMap.merge(amaps.n, bmaps.n, nrename, cc.getStartLoc(),
				p, lastVarUse);
	  out.x = VarMap.merge(amaps.x, bmaps.x, xrename, cc.getStartLoc(),
				p, lastVarUse);
          return GC.choosecmd(GC.trycmd(GC.seq(ap, GC.seq(nrename[0])),
					GC.seq(GC.seq(xrename[0]), GC.raise())),
			      GC.trycmd(GC.seq(bp, GC.seq(nrename[1])),
					GC.seq(GC.seq(xrename[1]), GC.raise())));
	}

      default:
	//@ unreachable;
	Assert.fail("Fall thru on "+g);
	return null;
    }
  }

  //+@ requires lastVarUse.keyType == \type(GenericVarDecl);
  //+@ requires lastVarUse.elementType == \type(RefInt);
  private static void computeLastVarUses(/*@ non_null */ GuardedCmd g,
					 /*@ non_null */ RefInt preOrderCount,
					 /*@ non_null */ Hashtable lastVarUse) {
    int id = preOrderCount.value;
    preOrderCount.value++;

    switch (g.getTag()) {
    case TagConstants.SKIPCMD:
    case TagConstants.RAISECMD:
      break;

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
	  Expr nuv = null;
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
		if (sgc.rhs == null) break;
		nuv = GC.nary(Location.NULL, Location.NULL,
			      TagConstants.STORE, sgc.v, sgc.index, sgc.rhs);
		break;
	      }

	    case TagConstants.SUBSUBGETSCMD:
	      {
		SubSubGetsCmd ssgc = (SubSubGetsCmd)g;
		if (ssgc.rhs == null) break;
		
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

	  if (nuv != null) {
	      RefInt pi = new RefInt(id);
	      for (Enumeration e = Substitute.freeVars(nuv).elements();
		   e.hasMoreElements(); ) {
		GenericVarDecl v = (GenericVarDecl)e.nextElement();
		lastVarUse.put(v, pi);
	      }
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
	//@ unreachable;
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
	//@ unreachable;
	Assert.fail("Fall thru on "+g);
	break;
    }
  }
}
