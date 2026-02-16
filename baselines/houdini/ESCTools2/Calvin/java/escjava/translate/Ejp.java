/* Copyright Hewlett-Packard, 2002 */


package escjava.translate;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import javafe.util.Location;
import javafe.util.Assert;

public abstract class Ejp {

  public static Expr compute(GuardedCmd g, Expr normal, Expr exceptional) {
    switch (g.getTag()) {
    case TagConstants.SKIPCMD: {
      // ejp[[ Skip ]](N,X) = N
      return normal;
    }

    case TagConstants.YIELDCMD: {
	YieldCmd yc = (YieldCmd) g;
	return compute( yc.desugared, normal, exceptional );
    }

    case TagConstants.LOOPCMD:
    {
      LoopCmd lp= (LoopCmd)g;
      return compute( lp.desugared, normal, exceptional );
    }
    
    case TagConstants.RAISECMD:
      // ejp[[ Raise ]](N,X) = X
      return exceptional;

    case TagConstants.ASSERTCMD:
      // ejp[[ Assert x ]](N,X) = x /\ P
      return GC.and(((ExprCmd)g).pred, normal);

    case TagConstants.ASSUMECMD:
      // ejp[[ Assume x ]](N,X) = x ==> N
      return GC.implies(((ExprCmd)g).pred, normal);

    case TagConstants.CALL:
      {
	Call call = (Call)g;
	return compute( call.desugared, normal, exceptional );
      }

    case TagConstants.GETSCMD:
    case TagConstants.SUBGETSCMD:
    case TagConstants.SUBSUBGETSCMD:
    case TagConstants.RESTOREFROMCMD:
	{
	  // ejp[[ v = x ]](N,X) = forall t.(x==t => N/t for x)
	  // and variants for the other kinds of assignment  

	  AssignCmd gc = (AssignCmd)g;

	  // Create t
	  LocalVarDecl tDecl = UniqName.newIntermediateStateVar(gc.v, "");
	  VariableAccess tRef	= VariableAccess.make( tDecl.id, gc.v.loc,
						       tDecl);

	  // Calculate the new value of v
	  Expr nuv;
	  switch( g.getTag() ) {
	    case TagConstants.GETSCMD:
	    case TagConstants.RESTOREFROMCMD:
	      {
		// ejp[[ v = x ]](N,X) = forall t.(x==t => N/t for x)
		// ejp[[ RESTORE v FROM x ]] == ejp[[ v = x ]]

		nuv = gc.rhs;
		break;
	      }

	    case TagConstants.SUBGETSCMD:
	      {
		// ejp[[ v[i] = x ]](N,X) = forall t. (t==store(v,i,x) => N/t for x)

		SubGetsCmd sgc = (SubGetsCmd)g;
		nuv = GC.nary(Location.NULL, Location.NULL,
			      TagConstants.STORE, sgc.v, sgc.index, sgc.rhs);
		break;
	      }

	    case TagConstants.SUBSUBGETSCMD:
	      {
		// ejp[[ v[i1][i2] = x]](N,X) =
		//    N/"store(v, i1, store(select(v,i1), i2, x)) for v"

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

	  Expr normal2 = GC.subst(gc.v.decl, tRef, normal);
	  Expr equals = GC.nary( gc.v.loc, gc.v.loc,
				 TagConstants.ANYEQ, tRef, nuv );
	  Expr implies = GC.implies( gc.v.loc, gc.v.loc,
				     equals, normal2 );
	  return GC.forall( g.getStartLoc(), g.getEndLoc(),
			    tDecl,
			    implies );
	}

    case TagConstants.VARINCMD:
      // ejp[[ var v in S ]](N,X) = (forall v . ejp[[S]](N,X))
      VarInCmd vc = (VarInCmd)g;
      Expr result = Ejp.compute(vc.g, normal, exceptional);
      for (int i = vc.v.size()-1; 0 <= i; i--)
	result = GC.forall(vc.v.elementAt(i), result);
      return result;

    case TagConstants.DYNINSTCMD:
      // ejp[[ DynamicInstanceCommand s in S end]](N,X) = ejp[[S]](N,X))
      DynInstCmd dc = (DynInstCmd)g;
      return Ejp.compute(dc.g, normal, exceptional);

    case TagConstants.SEQCMD:
      // ejp[[ S1 ; S2 ]](N,X) = ejp[[S1]](ejp[[S2]](N,X), X)
      SeqCmd sc = (SeqCmd)g;
      for (int i = sc.cmds.size()-1; 0 <= i; i--)
	normal = Ejp.compute(sc.cmds.elementAt(i), normal, exceptional);
      return normal;

    case TagConstants.TRYCMD:
      // ejp[[ S1 ! S2 ]](N,X) = ejp[[S1]](N, ejp[[S2]](N,X))
      CmdCmdCmd tc = (CmdCmdCmd)g;
      return Ejp.compute(tc.g1, normal, Ejp.compute(tc.g2, normal, exceptional));

    case TagConstants.CHOOSECMD:
      // ejp[[ S1 [] S2 ]](N,X) = ejp[[S1]](N,X) /\ ejp[[S2]](N,X)
      CmdCmdCmd cc = (CmdCmdCmd)g;
      return GC.and(Ejp.compute(cc.g1, normal, exceptional),
		 Ejp.compute(cc.g2, normal, exceptional));

    default:
      //@ unreachable
      Assert.fail("Fall thru on "+g);
      return null;
    }
  }
}
