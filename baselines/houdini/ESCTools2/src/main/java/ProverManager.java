/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava;

import escjava.backpred.FindContributors;
import escjava.prover.*;
import escjava.sortedProver.CounterExampleResponse;
import escjava.sortedProver.Lifter;
import escjava.sortedProver.SortedProver;
import escjava.sortedProver.SortedProverCallback;
import escjava.sortedProver.SortedProverResponse;
import escjava.sortedProver.NodeBuilder.SPred;
import escjava.translate.VcToString;
import javafe.ast.ASTNode;
import javafe.ast.Expr;
import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.FatalError;
import javafe.util.Info;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Properties;

public class ProverManager {
  
  public static interface Listener {
    void stateChanged(int s);
  }
  public static /*@ nullable */ Listener listener = null;
  
  final static private int NOTSTARTED = 0;
  final static private int STARTED = 1;
  final static private int PUSHED = 2;
  
  /*@ spec_public */ static private int status = 0;
  /*@ spec_public */ static private boolean isStarted = false;
  //@ private invariant status != NOTSTARTED <==> isStarted;
  
  static private /*@ nullable */ FindContributors savedScope = null;
  
  public static boolean useSimplify = false;
  public static boolean useSammy = false;
  public static boolean useHarvey = false;
  public static boolean useCvc3 = false;
  public static boolean useSorted = false;
  public static String[] sortedProvers;
  
  static escjava.backpred.BackPred backPred = new escjava.backpred.BackPred();

  //@ ensures isStarted && prover != null;
  synchronized
  static public void start() {
    
    if (isStarted) return;
    
    if( useSammy && (sammy == null || !sammy.started) ){
      long startTimeSammy = java.lang.System.currentTimeMillis();
      
      System.out.println("Launching demo of sammy...");
      
      Sammy.main(new String[0]);
      
      System.out.println("exiting...");
      System.exit(0);
      
    }
    
    if(!useSorted && useSimplify && (simplify == null)) { 
      
      long startTime = java.lang.System.currentTimeMillis();
      simplify = new Simplify();
      
      if (!Main.options().quiet)
        System.out.println("  Prover started:" + Main.timeUsed(startTime));
      
      backPred.genUnivBackPred(simplify.subProcessToStream());
      simplify.sendCommands("");
      
    }

    if (useCvc3 && (cvc3 == null || !cvc3.started)) {
      long startTime = java.lang.System.currentTimeMillis();
      if (cvc3 == null)
        cvc3 = new Cvc3(true);

      if (!Main.options().quiet) 
        System.out.println("  Prover started:" + Main.timeUsed(startTime));

      // set any flags here (-tcc and timeout, for example)
      Properties flags = new Properties();
      flags.setProperty(" ","-timeout 0");
      cvc3.set_prover_resource_flags(flags);

      cvc3.start_prover();
      // note: the backpred is included as part of the vcGen code
    } 
    
    if (useSorted && sortedProver == null) {
        long startTime = java.lang.System.currentTimeMillis();
        
        Assert.notFalse (sortedProvers != null && sortedProvers.length != 0);
        
        if (sortedProvers.length > 1)
        	ErrorSet.caution("sorry, only a single prover at once supported, using the first one");
        sortedProver = SortedProver.getProver(sortedProvers[0]);
        if (sortedProver == null) {
        	ErrorSet.fatal("cannot find prover: " + sortedProvers[0]);
        }

        if (!Main.options().quiet) 
          System.out.println("  Sorted prover started:" + Main.timeUsed(startTime));

        sortedProver.setProverResourceFlags(Main.options().svcgProverResourceFlags);
        
        lifter = new Lifter(sortedProver.getNodeBuilder());

        sortedProver.startProver();
        sortedProver.sendBackgroundPredicate();
    }
    	
    
    if (listener != null) listener.stateChanged(1);
    isStarted = true;
    status = STARTED;
    
  }
  
  synchronized
  static public /*@ non_null */ Simplify prover() {
	Assert.notFalse(useSimplify);
    start();
    return simplify;
  }
  
  //@ ensures status == NOTSTARTED && prover == null;
  synchronized
  static public void kill() {
    
    if(useSimplify) {
      if (simplify != null) 
        simplify.close();
      
      simplify = null;
    }
    
    if(useSammy) {
      sammy.stop_prover();
      sammy = null;
    }

    if(useCvc3) {
      cvc3.stop_prover();
      cvc3 = null;
    }
    
    if (useSorted && sortedProver != null) {
    	sortedProver.stopProver();
    	sortedProver = null;
    }
    
    if (listener != null) 
      listener.stateChanged(0);
    
    isStarted = false;
    status = NOTSTARTED;
  }
  
  synchronized
  static public void died() {
    
    if(useSimplify) {
      if (simplify != null) 
        simplify.close();
      
      simplify = null;
    }
    
    if(useSammy) {
      sammy.stop_prover();
      sammy = null;
    }
    
    if (listener != null) 
      listener.stateChanged(0);

    if(useCvc3) {
      cvc3.stop_prover();
      cvc3 = null;
    }
    
    if (useSorted) {
    	sortedProver.stopProver();
    	sortedProver = null;
    }
    
    isStarted = false;
    status = NOTSTARTED;
  }
  
  /*
   * Specific to simplify
   */
  synchronized
  static public void push(/*@ non_null @*/ Expr vc) {
	  if (useSorted) {
		  sortedProver.makeAssumption(lifter.convert(vc));
	  }
	  else {
		  Assert.notFalse(useSimplify);
		  PrintStream ps = simplify.subProcessToStream();
		  ps.print("\n(BG_PUSH ");
		  VcToString.computePC(vc, ps);
		  ps.println(")");
		  simplify.sendCommands("");
	  }
  }
  
  synchronized
  static public void push(/*@ non_null */ FindContributors scope) {
    start();
    if (useSorted) {
    	sortedProver.makeAssumption(lifter.generateBackPred(scope));
        savedScope = scope;
        status = PUSHED;
    }
    else if (simplify != null) {
      PrintStream ps = simplify.subProcessToStream();
      ps.print("\n(BG_PUSH ");
      backPred.genTypeBackPred(scope, ps);
      ps.println(")");
      simplify.sendCommands("");
      savedScope = scope;
      status = PUSHED;
    }
  }
  
  synchronized
  static public /*@ non_null */ Enumeration prove(/*@ non_null */ Expr vc, 
		  									      /*@ nullable */ FindContributors scope, 
		  									      /*@ non_null */ String problemName) 
  {
    if (!(useSimplify || useSorted))
			return null;
    start();

    if (scope == null) {
    	if (savedScope != null && status != PUSHED)
    		push(savedScope);
    } else {
    	if (status == PUSHED) {
    		if (savedScope != scope) {
    			pop();
    			push(scope);
    		}
    	} else {
    		push(scope);
    	}
    }

    if (listener != null) listener.stateChanged(2);
      
    try {
    	if (useSorted)
    		return proveUsingSortedProver_helper(vc, problemName);

    	simplify.startProve();
    	VcToString.compute(vc, simplify.subProcessToStream());

    	Enumeration en;
    	if (PROVER_IN_ITS_OWN_THREAD) {
        	SimplifyProverRunnable r = new SimplifyProverRunnable(simplify);
        	Thread simplifyProverThread = new Thread(r);
        	simplifyProverThread.start();
        	simplifyProverThread.join(PROVER_KILL_TIME*1000); // can throw InterruptedException.
        	en = r.getProverResult();
        	if (en == null) {
        		if (THREADING_TRACE_OUTPUT > 0) System.out.println("\tProverManager: interrupting simplify");
        		simplifyProverThread.interrupt(); // in case prover is still running.
        		if (THREADING_TRACE_OUTPUT > 9) System.out.println("\tProverManager: waiting for simplify to return");
        		simplifyProverThread.join(WAIT_FOR_PROVER_TO_DIE_TIME*1000); // wait for child process to die
        		if (THREADING_TRACE_OUTPUT > 9) System.out.println("\tProverManager: killing simplify");
        		died(); // kill simplify prover process since it is taking too much time.
        		en = new SimplifyTimeoutResultEnum();
        	}
    	} else {
    		en = simplify.streamProve();
    	}

    	if (listener != null) listener.stateChanged(1);
    	return en;
    } catch (FatalError e) {
    	died();
    	return null;
    } catch (InterruptedException e) { // Should not occur, but just in case ...
    	died();
    	return null;
	}
  }

  private static Enumeration proveUsingSortedProver_helper(Expr vc, String problemName) {
	  final ArrayList responses = new ArrayList();
	  SortedProverCallback cb = new SortedProverCallback() {
		  public void processResponse(SortedProverResponse resp) {
			  if (resp.getTag() == SortedProverResponse.COUNTER_EXAMPLE) {
				  String[] labels = ((CounterExampleResponse) resp)
				  .getLabels();
				  SExp[] labels2 = new SExp[labels.length];
				  for (int i = 0; i < labels.length; ++i)
					  labels2[i] = SExp.fancyMake(labels[i]);
				  responses.add(new SimplifyResult(
						  SimplifyOutput.COUNTEREXAMPLE, SList
						  .fromArray(labels2), null));
			  }
		  }
	  };

	  Properties props = new Properties();
	  props.setProperty("ProblemName", problemName);

	  SortedProverResponse resp = liftAndProve(vc, cb, props);

	  if (resp.getTag() == SortedProverResponse.FAIL)
		  died();

	  responses
	  .add(new SimplifyOutput(
			  resp.getTag() == SortedProverResponse.YES ? SimplifyOutput.VALID
					  : SimplifyOutput.INVALID));

	  return new Enumeration() {
		  int pos = 0;

		  public boolean hasMoreElements() {
			  return (pos < responses.size());
		  }

		  public Object nextElement() {
			  if (pos >= responses.size())
				  throw new java.util.NoSuchElementException();
			  return responses.get(pos++);
		  }
	  };
  }
  
  static int cnt = 0;

  private static SortedProverResponse liftAndProve(Expr vc, SortedProverCallback cb, Properties props)
  {
	  //long startTime = Main.currentTime();
	  Info.out ("[running lifter]");
	  
	  //System.err.println("[pre-lifter time " + Main.timeUsed(startTime) + "]");
	  //Runtime.getRuntime().gc();
	  //System.err.println("[pre-lifter, after GC time " + Main.timeUsed(startTime) + "]");
	  SPred pred = lifter.convert(vc);
	  
	  //System.err.println("[lifter time " + Main.timeUsed(startTime) + "]");
	  
	  Info.out ("[calling prover]");
	  SortedProverResponse resp = sortedProver.isValid(pred, cb, props);
	  Info.out ("[prover done]");
	  
	  //System.err.println("[prover time " + Main.timeUsed(startTime) + "]");
	  pred = null;
	  //Runtime.getRuntime().gc();
	  //System.err.println("[after gc " + Main.timeUsed(startTime) + "]");
	  
	  //if (cnt++ > 2) System.exit(0);
	  
	  return resp;
  }
  
  static synchronized
  public boolean isValid(Expr vc, Properties props)
  {
      if (savedScope != null && status != PUSHED) push(savedScope);
      
      if (useSorted) {
          SortedProverCallback cb = new SortedProverCallback() {
              public void processResponse(SortedProverResponse resp) {}
          };
          start();
          SortedProverResponse resp = liftAndProve(vc, cb, props); 
          if (resp.getTag() == SortedProverResponse.FAIL)
        	  died();
          
          return resp.getTag() == SortedProverResponse.YES;
      } else {
          Assert.notFalse(useSimplify);
          PrintStream ps = simplify.subProcessToStream();
          simplify.startProve();
          VcToString.compute(vc, ps);
          Enumeration results = simplify.streamProve();
          while (results.hasMoreElements()) {
              SimplifyOutput so = (SimplifyOutput)results.nextElement();
              //System.err.println(so.getKind());
              if (so.getKind() == SimplifyOutput.VALID) {
                  Assert.notFalse(!results.hasMoreElements());
                  return true;
              }
          }
          return false;
      }
  }
  
  /*
   * Specific to simplify
   */
  synchronized
  static public void pop() {
	  if (sortedProver != null)
		  sortedProver.retractAssumption(1);
	  else  if (simplify != null)
		  simplify.sendCommand("\n(BG_POP)");
	  savedScope = null;
	  status = STARTED;
  }
  
  /**
   * Our Simplify instance.
   */
  //-@ monitored
  public static /*@ nullable */ Simplify simplify;
  //@ private invariant isStarted ==> prover != null;
  
  /*
   * Our Sammy instance \\o \o/ o//
   */
  public static /*@ nullable */ Sammy sammy;
  //@ public static model Object prover;
  //@ static represents prover <- sammy;

  /*
   * Our Cvc3 instance.
   */
  public static /*@ nullable */ Cvc3 cvc3;
  
  /*
   * Our generic sorted prover instance.
   */
  public static /*@ nullable */ SortedProver sortedProver;
  public static /*@ nullable */ Lifter lifter;

  public static final boolean PROVER_IN_ITS_OWN_THREAD;
  public static final int THREADING_TRACE_OUTPUT;
  public static final int PROVER_KILL_TIME; // in seconds.
  public static final int WAIT_FOR_PROVER_TO_DIE_TIME = 1; // in seconds.

  static {
	  String proverInItsOwnThread = System.getProperty("PROVER_IN_ITS_OWN_THREAD");
	  PROVER_IN_ITS_OWN_THREAD = "1".equals(proverInItsOwnThread);
	  
	  THREADING_TRACE_OUTPUT = PROVER_IN_ITS_OWN_THREAD ? 5 : 0;
	  
	  String pktEnvVar = System.getProperty("PROVER_KILL_TIME");
	  int pkt;
	  try {
		  pkt = Integer.parseInt(pktEnvVar);
	  } catch (NumberFormatException e) {
		  pkt = -1;
	  }
	  if ((pkt < 0 || pkt > 1000) && pktEnvVar != null && !"".equals(pktEnvVar)) {
		  System.err.println("ERROR: illegal value for PROVER_KILL_TIME:"
				  + pktEnvVar);
		  pkt = 5;
	  }
	  PROVER_KILL_TIME = pkt;
	  if (THREADING_TRACE_OUTPUT > 0) System.out.println("\tProverManager: using PROVER_KILL_TIME " + PROVER_KILL_TIME);
  }
}
