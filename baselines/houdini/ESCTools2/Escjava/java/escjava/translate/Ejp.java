/* Copyright 2000, 2001, Compaq Computer Corporation */


package escjava.translate;

import javafe.ast.*;
import javafe.ast.GenericVarDecl;
import javafe.ast.LocalVarDecl;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.sp.VarMap;

import javafe.util.Location;
import javafe.util.Assert;

public abstract class Ejp {

  public static Expr compute(/*@ non_null */ GuardedCmd g,
			     /*@ non_null */ Expr normal,
			     /*@ non_null */ Expr exceptional) {
    VarMap dynInstMap = VarMap.identity();
    return compute(g, normal, exceptional, "", dynInstMap);
  }

  /*@ requires (* "normal" and "exceptional" have been as inflected by
                   "dynInstMap" as they need to be, that is, this method
		   is not to apply "dynInstMap" to "normal" and
		   "exceptional" *);
       ensures (* \result has been as inflected by "dynInstMap" as it needs
                  to be *);
   */
  private static Expr compute(/*@ non_null */ GuardedCmd g,
			      /*@ non_null */ Expr normal,
			      /*@ non_null */ Expr exceptional,
			      /*@ non_null */ String dynInstPrefix,
			      /*@ non_null */ VarMap dynInstMap) {
    switch (g.getTag()) {
    case TagConstants.SKIPCMD:
      // ejp[[ Skip ]](N,X) = N
      return normal;

    case TagConstants.LOOPCMD:
    {
      LoopCmd lp= (LoopCmd)g;
      return compute(lp.desugared, normal, exceptional,
		     dynInstPrefix, dynInstMap);
    }
    
    case TagConstants.RAISECMD:
      // ejp[[ Raise ]](N,X) = X
      return exceptional;

    case TagConstants.ASSERTCMD: {
      // ejp[[ Assert x ]](N,X) = x /\ P
      ExprCmd ec = (ExprCmd)g;
      return GC.and(dynInstMap.apply(ec.pred), normal);
    }

    case TagConstants.ASSUMECMD: {
      // ejp[[ Assume x ]](N,X) = x ==> N
      ExprCmd ec = (ExprCmd)g;
      return GC.implies(dynInstMap.apply(ec.pred), normal);
    }

    case TagConstants.CALL:
      {
	Call call = (Call)g;
	return compute(call.desugared, normal, exceptional,
		       dynInstPrefix, dynInstMap);
      }

    case TagConstants.GETSCMD:
    case TagConstants.SUBGETSCMD:
    case TagConstants.SUBSUBGETSCMD:
    case TagConstants.RESTOREFROMCMD:
	{
	  // ejp[[ v = x ]](N,X) =
	  //   (forall t :: t==x ==> N[dynInstMap(v):=t])
	  // and variants for the other kinds of assignment

	  AssignCmd gc = (AssignCmd)g;

	  // Create t
	  LocalVarDecl tDecl = UniqName.newIntermediateStateVar(gc.v,
								dynInstPrefix);
	  VariableAccess tRef = VariableAccess.make(tDecl.id, gc.v.loc, tDecl);

	  // Calculate the new value of v
	  Expr nuv;
	  switch( g.getTag() ) {
	    case TagConstants.GETSCMD:
	    case TagConstants.RESTOREFROMCMD:
	      {
		// ejp[[ v = x ]](N,X) =
		//   (forall t :: t==x ==> N[dynInstMap(v):=t])
		// ejp[[ RESTORE v FROM x ]] == ejp[[ v = x ]]

		nuv = gc.rhs;
		break;
	      }

	    case TagConstants.SUBGETSCMD:
	      {
		// ejp[[ v[i] = x ]](N,X) =
		//   (forall t :: t==store(v,i,x) ==> N[dynInstMap(v):=t])

		SubGetsCmd sgc = (SubGetsCmd)g;
		nuv = GC.nary(Location.NULL, Location.NULL,
			      TagConstants.STORE, sgc.v, sgc.index, sgc.rhs);
		break;
	      }

	    case TagConstants.SUBSUBGETSCMD:
	      {
		// ejp[[ v[i1][i2] = x]](N,X) =
		//    (forall t :: t==store(v, i1, store(select(v,i1), i2, x))
		//                 ==>
		//                 N[dynInstMap(v):=t])

		SubSubGetsCmd ssgc = (SubSubGetsCmd)g;

		Expr innermap = GC.nary(Location.NULL, Location.NULL,
					TagConstants.SELECT,
					ssgc.v, ssgc.index1);
		Expr newinnermap = GC.nary(Location.NULL, Location.NULL,
					   TagConstants.STORE,
					   innermap, ssgc.index2, ssgc.rhs);
		nuv = GC.nary(Location.NULL, Location.NULL,
			      TagConstants.STORE,
			      ssgc.v, ssgc.index1, newinnermap);
		break;
	      }

	    default:
	      Assert.fail("Unreachable");
	      nuv = null; // dummy assignment
	  }

	  // vv = dynInstMap(v)
	  VariableAccess vva = (VariableAccess)dynInstMap.get(gc.v.decl);
	  GenericVarDecl vv;
	  if (vva == null) {
	    vv = gc.v.decl;
	  } else {
	    vv = vva.decl;
	  }

	  Expr normal2 = GC.subst(vv, tRef, normal);
	  int locStart = g.getStartLoc();
	  int locEnd = g.getEndLoc();
	  Expr equals = GC.nary(locStart, locEnd,
				TagConstants.ANYEQ,
				tRef, dynInstMap.apply(nuv));
	  Expr implies = GC.implies(locStart, locEnd, equals, normal2);
	  return GC.forall(locStart, locEnd, tDecl, null, implies);
	}

    case TagConstants.VARINCMD: {
      // ejp[[ var v in S ]](N,X) = (forall v . ejp[[S]](N,X))
      VarInCmd vc = (VarInCmd)g;

      LocalVarDecl[] newNames = new LocalVarDecl[vc.v.size()];
      for (int i = 0; i < vc.v.size(); i++) {
	GenericVarDecl v = vc.v.elementAt(i);

	// create a new variable with a unique name...
	LocalVarDecl decl = UniqName.newIntermediateStateVar(v, dynInstPrefix);
	newNames[i] = decl;
	VariableAccess xpRef = VariableAccess.make(decl.id, decl.locId, decl);
	// ...and map "v" to it
	dynInstMap = dynInstMap.extend(v, xpRef);
      }

      Expr result = Ejp.compute(vc.g, normal, exceptional, dynInstPrefix,
				dynInstMap);
      for (int i = newNames.length-1; 0 <= i; i--) {
	result = GC.forall(newNames[i], result);
      }

      return result;
    }

    case TagConstants.DYNINSTCMD: {
      // ejp[[ DynamicInstanceCommand s in S end]](N,X) = ejp[[S]](N,X))
      DynInstCmd dc = (DynInstCmd)g;
      return Ejp.compute(dc.g, normal, exceptional,
			 dynInstPrefix + "-" + dc.s, dynInstMap);
    }

    case TagConstants.SEQCMD: {
      // ejp[[ S1 ; S2 ]](N,X) = ejp[[S1]](ejp[[S2]](N,X), X)
      SeqCmd sc = (SeqCmd)g;
      for (int i = sc.cmds.size()-1; 0 <= i; i--)
	normal = Ejp.compute(sc.cmds.elementAt(i), normal, exceptional,
			     dynInstPrefix, dynInstMap);
      return normal;
    }

    case TagConstants.TRYCMD: {
      // ejp[[ S1 ! S2 ]](N,X) = ejp[[S1]](N, ejp[[S2]](N,X))
      CmdCmdCmd tc = (CmdCmdCmd)g;
      return Ejp.compute(tc.g1,
			 normal, Ejp.compute(tc.g2, normal, exceptional,
					     dynInstPrefix, dynInstMap),
			 dynInstPrefix, dynInstMap);
    }

    case TagConstants.CHOOSECMD: {
      // ejp[[ S1 [] S2 ]](N,X) = ejp[[S1]](N,X) /\ ejp[[S2]](N,X)
      CmdCmdCmd cc = (CmdCmdCmd)g;
      return GC.and(Ejp.compute(cc.g1, normal, exceptional,
				dynInstPrefix, dynInstMap),
		    Ejp.compute(cc.g2, normal, exceptional,
				dynInstPrefix, dynInstMap));
    }

    default:
      //@ unreachable;
      Assert.fail("Fall thru on "+g);
      return null;
    }
  }
}
