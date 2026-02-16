package escjava.prover;

import java.util.Properties;
import org.apache.xmlrpc.*;
import java.util.Vector;
import java.util.Enumeration;
import java.util.Iterator;
import java.lang.Thread;
import escjava.prover.jniw.Cvc3Wrapper;
import escjava.prover.jniw.Cvc3WrapperException;

public class Cvc3 extends NewProver {
  
  /*
   * README
   * This class and the new other ones like NewProver, ProverResponse
   * have been specified with jml. Even if we cannot use jmlc and jmlrac for
   * the whole escjava2 program, these new classes have been tested
   * apart from the rest of the program. Using jmlrac on a few test seems 
   * to show that these classes are respecting there specification.
   * Anyway nothing has been done with escjava2.
   */ 
  
  //@ public invariant started ==> wrapper != null;
  
  private /*@spec_public*/ static /*@non_null*/Cvc3Wrapper wrapper = new Cvc3Wrapper();
  
  static boolean debug = false;
  
  // special constructor for debugging
  public Cvc3(boolean debug){
    Cvc3.debug = debug;
  }
  
  /** Starts a new instance of CVC3.  This is necessary in order to redefine
   * any functions or variables used.
   *
   * @return OK (or exit)
   */
  /*@
   @ also
   @ assignable wrapper;
   @*/
  public /*@non_null*/ProverResponse start_prover() {
    
    //++
    if(debug) System.out.println("Cvc3::start_prover");
    //++
    
    try { wrapper.startSolver(); }
    catch (Exception e) {
      System.err.println("Cvc3Wrapper::"+e);
      System.exit(0);
    }
    
    ProverResponse res = ProverResponse.OK;
    
    //++
    if(debug)
      if( res != ProverResponse.OK )
        System.out.println("Failed to init cvc");
      //++
      
    started = true;
    
    return res;
    
  }
  
  /**
   * Sets command line flags for CVC3.  Flag are currently just strings in
   * the properties.  
   *
   * @param properties is a table of the flags.  Keys and entries are strings,
   * and are concatenated together to form each flag (so there is no 
   * functional difference between ["a b",""] and ["a","b"]).
   *
   * @return OK or SYNTAX_ERROR, as appropriate.
   *
   */
  public /*@non_null*/ProverResponse set_prover_resource_flags(/*@non_null*/Properties properties) {
    
    //++
    if(debug) System.out.println("Cvc3::set_prover_resource_flags");
    //++
    
    /* we have to create a big string containing
     * all the flags ( = key in the hashtable of Properties)
     * and their values ( = value)
     */
    Enumeration flags = properties.keys();
    
    String value = ""; // command string
    String currentFlag = null;
    
    for (  ; flags.hasMoreElements() ;) {
      
      try {
        currentFlag = (String)flags.nextElement();
        value += currentFlag+" "+properties.getProperty( currentFlag );
      }
      
      catch (Exception e) {
        System.err.println("Cvc3::Failed to inspect properties");
        System.exit(0);
      }
    }
   
    ProverResponse res = ProverResponse.OK;
 
    try {
      wrapper.setFlags(value);
    }
    catch (Exception e) {
      res = ProverResponse.SYNTAX_ERROR;
    }
    
    //++
    if(debug)
      if( res!= ProverResponse.OK )
        System.out.println("Failed to set flags");
      //++
      
    return res;
  }
 
  /**
   * Declares a signature to CVC3.  The signature is assumed to be a string
   * of a sequence of CVC-readable declaration commands (each 
   * terminated by a semicolon).
   *
   * @return ProverResponse.OK if no problems encountered, otherwise
   * ProverResponse.FAIL.
   */ 
  public /*@non_null*/ProverResponse signature(/*@non_null*/Signature signature) {
    //++
    if(debug)
      System.out.print("Cvc3::signature");
    //++
    
    /*
     * This requires the signature be able to be transformed into a list
     * (Vector) of strings for each declaration command (without the ;)
     * res can only degrade from "OK".
     */

    // We first need to decompose the signature into commands

    ProverResponse res = ProverResponse.OK;
    String [] cmd = signature.toString().split(";");
    int sz = cmd.length;
 
    // now we cycle through each command (replacing the semicolon):
    // this assumes a CVC presentation language format!

    try {
      for (int i = 0; i < sz; i++) {
        if (cmd[i].indexOf("TYPE") > -1) {
          // tyedefs
          wrapper.declType(cmd[i]+";");
        } else if (cmd[i].matches("BOOLEAN\\s*\\z")) {
          // predicates must have BOOLEAN domain
          wrapper.declPred(cmd[i]+";");
        } else {
          // everything else is a function
          wrapper.declFun(cmd[i]+";");
        }
      }
    } catch (Exception e) {
      res = ProverResponse.FAIL;
    }

    	//++
     	if(debug)
     	    if( res!= ProverResponse.OK )
     		System.out.println("Failed to set signature definitions");

    signature_defined = true;
    
    return res;
  }

  /**
   * Same as make_asumption.
   */ 
  public /*@non_null*/ProverResponse declare_axiom(/*@non_null*/Formula formula) {
    
    /*
     * This is currently treated tha same as make_assumption
     */ 
    
    return make_assumption(formula);
  }
 
  /**
   * Asserts a formula to cvc3.
   *
   * @param The fomula to be asserted, as a string.  (not a "command")
   *
   * @return OK, FAIL, or INCONSISTENCY_WARNING.  INCONSISTENCY_WARNING will
   * indicate the current (post-assertion) context has been detected as
   * inconsistent.
   */
  public /*@non_null*/ProverResponse make_assumption(/*@non_null*/Formula formula) {
    
    /*
     * This is currently treated the same as declare_axiom.
     * Assume the formula is just a formula, not a command.
     * The assertion goes through even if the context becomes inconsistent
     */
    
    ProverResponse res = ProverResponse.OK;

    String cmd = "ASSERT " + formula.toString() + ";";
    
    try {
      wrapper.assertFormula(cmd);
      if (wrapper.contextInconsistent()) {
        res = ProverResponse.INCONSISTENCY_WARNING;
        res.info = new Properties();
        res.formula = new Formula("");
      }
    } catch (Exception e) {
      res = ProverResponse.FAIL;
    }

    return res;
  }
  
  /**
   * Retracts some number of previous assumptions.
   *
   * @param count the number of undos.  If count &lt = 0 then do nothing.  
   * If count &gt the number of assertions, all assertions will be undone.
   *
   * @return OK.
   */
  public /*@non_null*/ProverResponse retract_assumption(int count) {
    
    wrapper.undoAssert(count);   
 
    return ProverResponse.OK;
  }
  
  /**
   * Perform a query on a formula.
   *
   * @param formula the formula to be checked.
   * @param properties not currently used
   *
   * @return YES (is valid), COUNTEREXAMPLE (possibly spurious), or TIMEOUT.
   * Note is is possible to also distinguish truly invalid operations from
   * invalid-but-possibly-incomplete ones, but there does not appear to be
   * a ProverResponse for "don't know" responses that allows the (possible)
   * counterexample.
   */
  public /*@non_null*/ProverResponse is_valid(/*@non_null*/Formula formula,
      Properties properties) {
    
    ProverResponse res = ProverResponse.OK;

    /*
     * assume the query is not in command form;
     * Properties are currently ignored
     * a "Don't know" response is returned as a timeout
     */

    try {
      String cmd = "QUERY " + formula.toString() + ";";
      String r = wrapper.queryFormula(cmd);
      if (r.startsWith("Abort")) {
        res = ProverResponse.TIMEOUT; //should this be something else?
      } else if (r.startsWith("Valid")) {
        res = ProverResponse.YES; // is VALID
      } else { // is INVALID or DON'T KNOW
        int colon = r.indexOf(":");
        String counterex = r.substring(colon+1);
        res = ProverResponse.COUNTER_EXAMPLE;
        res.formula = new Formula(counterex);
      }
    } catch (Exception e) { // PROBLEM
      res = ProverResponse.SYNTAX_ERROR;
      res.info = new Properties();
      res.info.setProperty("ERROR",e.toString());
    }    
    
    return res;
  }
  

  /**
   * Destroys the current instance of cvc3.  It is necessary to call
   * start_solver() to do further processing.
   *
   * Returns OK or FAIL.
   */
    //@ also
    //@ assignable signature_defined, started;
  public /*@non_null*/ProverResponse stop_prover() {
    
    //++
    if(debug) System.out.println("Cvc3::stop_prover");
    //++
    
    int result;
    
    ProverResponse res = ProverResponse.OK;

    try {
      wrapper.stopSolver();
    } catch (Exception e) {
      res = ProverResponse.FAIL;
    }
    
    if(debug)
      if( res != ProverResponse.OK )
        System.out.println("Cvc3::Failed to stop");
      
    started = false;
    signature_defined = false;
    
    return res;
  }
  
  /*
   * Test
   */
  static void printResponse(ProverResponse r) {
    String s;
    System.out.println();
    if (r.equals(ProverResponse.OK)) {s = "OK";}
    else if (r.equals(ProverResponse.FAIL)) {s = "FAIL";}
    else if (r.equals(ProverResponse.YES)) {s = "YES";}
    else if (r.equals(ProverResponse.NO)) {s = "NO";}
    else if (r.equals(ProverResponse.COUNTER_EXAMPLE)) {s = "COUNTER_EXAMPLE="+r.formula.toString();}
    else if (r.equals(ProverResponse.SYNTAX_ERROR)) {s = "SYNTAX_ERROR";}
    else if (r.equals(ProverResponse.PROGRESS_INFORMATION)) {s = "PROGRESS_INFORMATION";}
    else if (r.equals(ProverResponse.TIMEOUT)) {s = "TIMEOUT";}
    else if (r.equals(ProverResponse.INCONSISTENCY_WARNING)) {s = "INCONSISTENCY_WARNING";}
    else {s = "UNKNOWN RESPONSE";}
    System.out.println(s);
  }


  static public void main(String[] argv){
    
    long startTime = java.lang.System.currentTimeMillis();
    Cvc3 cvc = new Cvc3(true);
    
    /* Test 1 */
    cvc.start_prover();
    Properties flags = new Properties();
    Properties p = new Properties();
    flags.setProperty(" ","-tcc");
    ProverResponse r;
    cvc.set_prover_resource_flags(flags);
    cvc.signature(new Signature("f:(INT,INT)->INT;x,y,z:INT;a,b,c:BOOLEAN;"));
    cvc.make_assumption(new Formula("a AND b AND NOT c"));
    cvc.make_assumption(new Formula("x=y"));
    cvc.make_assumption (new Formula("f(x,x)=y"));
    cvc.make_assumption(new Formula("f(x,x)=z"));
    r = cvc.is_valid(new Formula("x=z"),p);
    printResponse(r);
    cvc.retract_assumption(2);
    r = cvc.is_valid(new Formula("f(x+1,x+1)=y"),p);
    printResponse(r);
    cvc.stop_prover();
    flags = new Properties();
    flags.setProperty(" ","-lang presentation -output-lang smtlib");
    cvc.set_prover_resource_flags(flags);
    cvc.start_prover();
    cvc.signature(new Signature("f:(INT,INT)->INT;a,b,c:INT;x,y,z:BOOLEAN;"));
    cvc.make_assumption(new Formula("NOT(x AND NOT y AND z)"));
    cvc.make_assumption(new Formula("a/=b"));
    cvc.make_assumption(new Formula("f(a+1,a+1)=b+1"));
    cvc.make_assumption(new Formula("f(a,a)=c"));
    r = cvc.is_valid(new Formula("NOT a=c"),p);
    printResponse(r);
    r = cvc.is_valid(new Formula("f(a,b)=b"),p);
    printResponse(r);
    r = cvc.is_valid(new Formula("EXISTS(c:INT):f(a,c)=b"),p);
    printResponse(r);
    cvc.start_prover();

    long endTime = java.lang.System.currentTimeMillis();
    
    long totalTime = endTime - startTime;
    
    System.out.println("Time spend : "+totalTime+" milliseconds");
    
  }
  
}
