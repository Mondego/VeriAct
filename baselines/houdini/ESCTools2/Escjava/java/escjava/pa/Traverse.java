/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.pa;

import escjava.*;
import escjava.prover.*;
import java.io.*;

import javafe.util.Set;
import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.translate.*;

import javafe.util.Location;
import javafe.util.Assert;

public class Traverse
{
    public static void compute(GuardedCmd g, InitialState initState, Translate tr) {
	ProverManager.push(initState.getInitialState());
/*
        PrintStream ps = Main.prover.subProcessToStream();
        ps.print("\n(BG_PUSH ");
        VcToString.compute(initState.getInitialState(), ps);
        ps.println(")");
        Main.prover.sendCommands("");
*/
        Set env = new Set();
        GuardedCmd h = computeHelper(g, GC.skip(), env);
	ProverManager.pop();
        //Main.prover.sendCommand("(BG_POP)");
        desugarLoops(g, tr);
    }

    private static void desugarLoops(GuardedCmd g, Translate tr) {
	switch (g.getTag()) {
            case TagConstants.SKIPCMD:
            case TagConstants.RAISECMD:
            case TagConstants.ASSERTCMD:
            case TagConstants.ASSUMECMD:
            case TagConstants.GETSCMD:
            case TagConstants.SUBGETSCMD:
            case TagConstants.SUBSUBGETSCMD:
            case TagConstants.RESTOREFROMCMD:
                return;
            case TagConstants.CALL: {
                desugarLoops(((Call) g).desugared, tr);
                return;
            }
            case TagConstants.VARINCMD: {
                desugarLoops(((VarInCmd) g).g, tr);
                return;
            }
            case TagConstants.DYNINSTCMD: {
                desugarLoops(((DynInstCmd) g).g, tr);
                return;
            }
            case TagConstants.SEQCMD: {
                SeqCmd sc = (SeqCmd) g;
                for (int i = 0; i < sc.cmds.size(); i++) {
                    desugarLoops(sc.cmds.elementAt(i), tr);
                }
                return;
            }
            case TagConstants.TRYCMD: {
                CmdCmdCmd tc = (CmdCmdCmd)g;
                desugarLoops(tc.g1, tr);
                desugarLoops(tc.g2, tr);
                return;
            }
            case TagConstants.CHOOSECMD: {
                CmdCmdCmd tc = (CmdCmdCmd)g;
                desugarLoops(tc.g1, tr);
                desugarLoops(tc.g2, tr);
                return;
            }
            case TagConstants.LOOPCMD: {
                LoopCmd loop = (LoopCmd) g;
                PredicateAbstraction pa = (PredicateAbstraction) PredicateAbstraction.paDecoration.get(loop);
                Assert.notNull(pa);

                ExprVec invs = pa.invariants;

                System.out.println("Loop invariant at "+
                                   Location.toString( loop.getStartLoc() )+
                                   ", " + invs.size() +" invariants, "+
                                   pa.milliSecsUsed+" ms, "+
                                   pa.perfCount.report() +" = "
                                   );

                for(int i=0; i<invs.size(); i++) {
                    Expr inv = invs.elementAt(i);
                    System.out.println("    "+PrettyPrint.inst.toString(inv));
                    Condition cond = GC.condition(TagConstants.CHKLOOPINVARIANT,
                                                  inv,
                                                  loop.getStartLoc());
                    loop.invariants.addElement(cond);
                }

                //	    System.out.println("Loop invariant compact: "+pa.set.concretize2());
                //System.out.println("Loop invariant as expr: "+PrettyPrint.inst.toString(pa.set.get()));


                tr.desugarLoopSafe(loop,ExprVec.make());
                desugarLoops(loop.desugared, tr);
                return;
            }
	    
            default:
                //@ unreachable;
                Assert.fail("Fall through on "+g);
                return;
	}	
    }

    static /*@ non_null @*/ GuardedCmd computeHelper(/*@ non_null @*/ GuardedCmd g, 
						  GuardedCmd context, Set env) 
    {
        switch (g.getTag()) {
            case TagConstants.SKIPCMD:
            case TagConstants.RAISECMD:
            case TagConstants.ASSERTCMD:
            case TagConstants.ASSUMECMD:
            case TagConstants.GETSCMD:
            case TagConstants.SUBGETSCMD:
            case TagConstants.SUBSUBGETSCMD:
            case TagConstants.RESTOREFROMCMD:
                return g;

            case TagConstants.CALL:
                {
                    Call call = (Call)g;
                    return computeHelper( call.desugared, context, env );
                }

            case TagConstants.VARINCMD:
                {
                    VarInCmd vc = (VarInCmd)g;
                    Set env2 = new Set(env.elements());
                    for(int i=0; i<vc.v.size(); i++) {
                        env2.add(vc.v.elementAt(i));
                    }
                    return VarInCmd.make(vc.v, computeHelper(vc.g, context, env2));
                }

            case TagConstants.DYNINSTCMD:
                {
                    DynInstCmd dc = (DynInstCmd)g;
                    return DynInstCmd.make(dc.s, computeHelper(dc.g, context, env));
                }

            case TagConstants.SEQCMD:
                {
                    SeqCmd sc = (SeqCmd)g;
                    GuardedCmd t = GC.skip();
                    for (int i = 0; i < sc.cmds.size(); i++) {
                        GuardedCmd r = computeHelper(sc.cmds.elementAt(i), GC.seq(context, t), env);
                        t = GC.seq(t, r);	      
                    }
                    return t;
                }

            case TagConstants.TRYCMD:
                {
                    CmdCmdCmd tc = (CmdCmdCmd)g;
                    GuardedCmd t1 = computeHelper(tc.g1, context, env);
                    GuardedCmd t2 = computeHelper(tc.g2, GC.seq(context, GC.trycmd(GC.seq(t1, GC.fail()), GC.skip())), env);
                    return GC.trycmd(t1, t2);
                }

            case TagConstants.CHOOSECMD:
                {
                    CmdCmdCmd cc = (CmdCmdCmd)g;
                    return GC.choosecmd(computeHelper(cc.g1, context, env), computeHelper(cc.g2, context, env));
                }

            case TagConstants.LOOPCMD:
                {
                    LoopCmd lp = (LoopCmd)g;
                    return PredicateAbstraction.abstractLoop( lp, context, env );
                }
    
            default:
                //@ unreachable;
                Assert.fail("Fall through on "+g);
                return null;
        }
    }
}
