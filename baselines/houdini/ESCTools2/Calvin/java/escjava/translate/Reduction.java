/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;
import java.io.*;

import javafe.ast.*;
import javafe.util.Location;
import javafe.util.StackVector;
import javafe.util.Assert;
import javafe.util.Set;
import javafe.util.ErrorSet;
import javafe.util.Info;
import javafe.tc.ConstantExpr;
import javafe.tc.TypeSig;

import escjava.Main;

import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.tc.FlowInsensitiveChecks;

import escjava.backpred.FindContributors;

import escjava.tc.Types;
import escjava.tc.TypeCheck;
import escjava.prover.*;
import escjava.sp.*;

public class Reduction {
    public static int NONMOVER = 0;
    public static int RIGHTMOVER = 1;
    public static int LEFTMOVER = 2;
    public static int ENDOFEXE = 3;

    static TrYield yieldObj;
    static PrintStream ps;
    static RoutineDecl rd;

    public static void reduce(RoutineDecl r, 
			      InitialState initState, 
			      TrYield obj, 
			      GuardedCmd gc) {
	yieldObj = obj;
	rd = r;

	ps = Main.prover.subProcessToStream();
	ps.print("\n(BG_PUSH ");
	VcToString.compute(initState.getInitialState(), ps);
	ps.println(")");
	Main.prover.sendCommands("");
	int[] out = new int[2];
	//	prove(yieldObj.getAssignCode(), gc, 0, out);
	//prove(GC.skip(), gc, RIGHTMOVER, out);
	prove(obj.getHavocAssumes(GenericVarDeclVec.make()), gc, RIGHTMOVER, out);
	Main.prover.sendCommand("(BG_POP)");
    }

    /* 
       out[0] : normal outcome
       out[1] : exceptional outcome
    */
    static void prove(GuardedCmd context, GuardedCmd stmt, int in, int[] out) {
	Assert.notFalse(in != ENDOFEXE);

	int tag = stmt.getTag();
	switch (tag) {
  	  case TagConstants.SKIPCMD:
	  case TagConstants.ASSERTCMD:
 	  case TagConstants.ASSUMECMD:    
          case TagConstants.GETSCMD: 
	  case TagConstants.SUBGETSCMD: 
 	  case TagConstants.SUBSUBGETSCMD:
	  case TagConstants.RESTOREFROMCMD: {
	      out[0] = in;
	      out[1] = ENDOFEXE;
	      return;
	  }
    
	  case TagConstants.RAISECMD : {
	      out[0] = ENDOFEXE;
	      out[1] = in;
	      return;
	  }

	  case TagConstants.YIELDCMD: {
	      YieldCmd yc = (YieldCmd) stmt;
	      if (in == RIGHTMOVER) {
		  if (isRightMover(context, yc)) {
		      out[0] = yc.mark = RIGHTMOVER;
		  } else {
		      out[0] = yc.mark = NONMOVER;
		  }
		  yc.desugared = yc.code;
	      } else { /* in == NONMOVER || in == LEFTMOVER */
		  GenericVarDeclVec threadLocalVars = TrYield.genThreadLocalVariables(yc.localVarDecls);
		  GuardedCmdVec gcVec = GuardedCmdVec.make();
		  gcVec.addElement(context);
		  gcVec.addElement(yieldObj.getHavocAssumes(threadLocalVars));
		  GuardedCmd newContext = GC.seq(gcVec);
 
		  if (isLeftMover(newContext, yc)) {
		      out[0] = yc.mark = LEFTMOVER;
		      yc.desugared = yc.code;
		  } else {
		      if (isRightMover(newContext, yc)) {
			  out[0] = yc.mark = RIGHTMOVER;
		      } else {
			  out[0] = yc.mark = NONMOVER;
		      }
		      yc.desugared = GC.seq(yieldObj.getHavocAssumes(threadLocalVars), yc.code);
		  }
	      }
	      out[1] = ENDOFEXE;

	      return;		  
	  }

	  case TagConstants.CALL : {
	      Call call = (Call)stmt;
	      prove(context, call.desugared, in, out);
	      return;
	  }

	  case TagConstants.VARINCMD : {
	      VarInCmd vc = (VarInCmd)stmt;
	      prove(context, vc.g, in, out);	    
	      return;
	  }

  	  case TagConstants.DYNINSTCMD : { 
	      DynInstCmd dc = (DynInstCmd)stmt;
	      prove(context, dc.g, in, out);
	      return;
	  }

 	  case TagConstants.SEQCMD : {
	      SeqCmd sc = (SeqCmd)stmt;
	      GuardedCmd c = context;

	      int[] tout = new int[2];
	      out[0] = in;
	      out[1] = ENDOFEXE;

	      for (int i = 0; i < sc.cmds.size(); i++) {
		  if (out[0] == ENDOFEXE) 
		      break;
		  prove(c, sc.cmds.elementAt(i), out[0], tout);
		  c = GC.seq(c, sc.cmds.elementAt(i));
		  out[0] = tout[0];
		  out[1] = combine(out[1], tout[1]);
	      }
	      return;
	  }

	  case TagConstants.TRYCMD: {
	      CmdCmdCmd tc = (CmdCmdCmd)stmt;

	      int[] tout = new int[2];

	      prove(context, tc.g1, in, tout);

	      out[0] = tout[0];
	      out[1] = tout[1];
	      
	      if (out[1] != ENDOFEXE) {
		  prove(GC.seq(context, GC.trycmd(GC.seq(tc.g1, GC.fail()), GC.skip())), 
			tc.g2, out[1], tout);

		  out[0] = combine(out[0], tout[0]);
		  out[1] = tout[1];
	      }
	      return;
	  }

	  case TagConstants.CHOOSECMD: {
	      CmdCmdCmd cc = (CmdCmdCmd)stmt;

	      int[] tout1 = new int[2];
	      prove(context, cc.g1, in, tout1);
	      
	      int[] tout2 = new int[2];
	      prove(context, cc.g2, in, tout2);

	      out[0] = combine(tout1[0], tout2[0]);
	      out[1] = combine(tout1[1], tout2[1]);
	      return;
	  }

	  case TagConstants.LOOPCMD : {
	      /* This currently works only for loop unrolling */
	      LoopCmd loop = (LoopCmd) stmt;
	      prove(context, loop.desugared, in, out);
	      return;
	  }

	  default : {
	      Assert.fail("Fall through on " + stmt);
	  }
	}
    }

    static int combine(int in1, int in2) {
	if (in1 == ENDOFEXE) return in2;
	if (in2 == ENDOFEXE) return in1;
	if (in1 == RIGHTMOVER && in2 == RIGHTMOVER) return RIGHTMOVER;
	if (in1 == LEFTMOVER && in2 == LEFTMOVER) return LEFTMOVER;
	return NONMOVER;
    }

    static boolean isLeftMover(GuardedCmd context, YieldCmd yc) {
	GuardedCmdVec gcVec = GuardedCmdVec.make();
	gcVec.addElement(context);
	gcVec.addElement(yieldObj.getAssignCode());
	gcVec.addElement(yc.code);

	for (int i = 0; i < yc.conds.size(); i++) {
	    gcVec.addElement(GC.check(yc.getStartLoc(), yc.conds.elementAt(i)));
	}
	for (int i = 0; i < yc.lmConds.size(); i++) {
	    gcVec.addElement(GC.check(yc.getStartLoc(), yc.lmConds.elementAt(i)));
	}
	GuardedCmd gc = GC.seq(gcVec);

	if (escjava.Main.showReduce) {
	    System.out.println("\n\nPrinting guarded command in isLeftMover");
	    ((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, gc);
	    System.out.println("");
	}

	GuardedCmd gcDsa = DSA.dsa(gc);
	Expr vc = SPVC.compute(gcDsa);

	Main.prover.startProve();
	VcToString.compute(vc, ps);
	Enumeration results = Main.prover.streamProve();
	return processSimplifyOutput(results);
    }

    static boolean isRightMover(GuardedCmd context, YieldCmd yc) {
	GuardedCmdVec gcVec = GuardedCmdVec.make();
	gcVec.addElement(context);
	gcVec.addElement(yieldObj.getAssignCode());
	gcVec.addElement(yc.code);

	for (int i = 0; i < yc.conds.size(); i++) {
	    gcVec.addElement(GC.check(yc.getStartLoc(), yc.conds.elementAt(i)));
	}
	for (int i = 0; i < yc.rmConds.size(); i++) {
	    gcVec.addElement(GC.check(yc.getStartLoc(), yc.rmConds.elementAt(i)));
	}
	GuardedCmd gc = GC.seq(gcVec);
	
	if (escjava.Main.showReduce) {
	    System.out.println("\n\nPrinting guarded command in isRightMover");
	    ((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, gc);
	    System.out.println("");
	}

	GuardedCmd gcDsa = DSA.dsa(gc);
	Expr vc = SPVC.compute(gcDsa);

	Main.prover.startProve();
	VcToString.compute(vc, ps);
	Enumeration results = Main.prover.streamProve();
	return processSimplifyOutput(results);
    }

    static boolean processSimplifyOutput(Enumeration results) {
	boolean status = true;
	boolean nextWarningNeedsPrecedingLine = true;
	while (results.hasMoreElements()) {
	  SimplifyOutput so = (SimplifyOutput)results.nextElement();
	  switch (so.getKind()) {
	    case SimplifyOutput.VALID:
	      break;
	    case SimplifyOutput.INVALID:
	      break;
	    case SimplifyOutput.UNKNOWN:
	      break;
	    case SimplifyOutput.COMMENT: {
	      SimplifyComment sc = (SimplifyComment)so;
	      System.out.println("SIMPLIFY: " + sc.getMsg());
	      break;
	    }
	    case SimplifyOutput.COUNTEREXAMPLE: {
		SimplifyResult sr = (SimplifyResult)so;
		boolean printing = false;
		/* first see if anything will be printed */
		try {
		    SList labelList = sr.getLabels();
		    int cLabels = labelList.length();
		    for (int i = 0; i < cLabels; i++) {
			String s = labelList.at(i).getAtom().toString();
			if (ErrorMsg.isErrorLabel(s)) {		      
			    if (ErrorMsg.isMoverLabel(s)) {
				status = false;
			    } else {
				printing = true;
			  }
			  break;		  
		      }
		  }	
		} catch (escjava.prover.SExpTypeError s) {
		    Assert.fail("unexpected S-expression exception");
		}

		if (printing && nextWarningNeedsPrecedingLine) {
		    escjava.translate.ErrorMsg.printSeparatorLine(System.out);
		    nextWarningNeedsPrecedingLine = false;
		}
	      try {
		  SList labelList = sr.getLabels();
		  int cLabels = labelList.length();
		  for (int i = 0; i < cLabels; i++) {
		      String s = labelList.at(i).getAtom().toString();
		      if (ErrorMsg.isErrorLabel(s)) {		      
			  if (ErrorMsg.isMoverLabel(s)) {
			      status = false;
			  } else {
			      ErrorMsg.print(TypeCheck.inst.getName(rd),
					     labelList, sr.getContext(),
					     rd, null, System.out); // don't turn Main.suggest on 
			  }
			  break;		  
		      }
		  }	
	      } catch (escjava.prover.SExpTypeError s) {
		  Assert.fail("unexpected S-expression exception");
	      }
	      break;
	    }
	    case SimplifyOutput.EXCEEDED_PROVER_KILL_TIME: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check " +
			       TypeCheck.inst.getName(rd) +
			       " of type " + TypeSig.getSig(rd.parent) +
			       " completely because too much time required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.EXCEEDED_PROVER_KILL_ITER: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check " +
			       TypeCheck.inst.getName(rd) +
			       " of type " + TypeSig.getSig(rd.parent) +
			       " completely because" +
			       " too many iterations required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.REACHED_CC_LIMIT:
	      ErrorSet.caution("Not checking " +
			       TypeCheck.inst.getName(rd) +
			       " of type " + TypeSig.getSig(rd.parent) +
			       " completely because" +
			       " warning limit (PROVER_CC_LIMIT) reached");
	      break;
	    case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_TIME: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check subgoal of " +
			       TypeCheck.inst.getName(rd) +
			       " of type " + TypeSig.getSig(rd.parent) +
			       " completely because too much time required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_ITER: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check subgoal of " +
			       TypeCheck.inst.getName(rd) +
			       " of type " + TypeSig.getSig(rd.parent) +
			       " completely because" +
			       " too many iterations required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.WARNING_TRIGGERLESS_QUANT: {
	      TriggerlessQuantWarning tqw = (TriggerlessQuantWarning)so;
	      int loc = tqw.getLocation();
	      String msg = "Unable to use quantification because " +
                           "no trigger found: " + tqw.e1;
	      if (loc != Location.NULL) {
		ErrorSet.caution(loc, msg);
	      } else {
		ErrorSet.caution(msg);
	      }
	      if (Info.on && tqw.getLabels() != null) {
		Info.out("Current labels: " + tqw.getLabels());
	      }
	      break;
	    }
	    default:
	      Assert.fail("unexpected type of Simplify output");
	      break;
	  }
	}
	
	return status;
    }

}

	/*
	switch (c) {
	case -1 : { // checking does not finish for some reason 
	      Assert.fail("Don't know how to handle this case in isRightMover");
	  }

	case 0 : {  // a right mover 
	      return true;
	  } 
	    
	case 1 : {  // not a right mover but satisfies both pre and post constraints 
	      return false;
	  }

	case 2 : { // does not satisfy pre constraint 
	      return false;
	  }

	case 3 : { // does not satisfy post constraint 
	      return false;
	  }

	case 4 : {  // some other assertion breaks 
	      return false;
	  }
	  
	  default : {
	      Assert.fail("Unexpected case in isRightMover");
	  }
	  }
	*/
