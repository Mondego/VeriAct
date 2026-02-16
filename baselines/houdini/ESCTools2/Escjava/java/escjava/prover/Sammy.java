package escjava.prover;

import java.util.Properties;
import org.apache.xmlrpc.*;
import java.util.Vector;
import java.util.Enumeration;
import java.util.Iterator;
import java.lang.Thread;

public class Sammy extends NewProver {
  
  /*
   * README
   * This class and the new other ones like NewProver, ProverResponse
   * have been specified with jml. Even if we cannot use jmlc and jmlrac for
   * the whole escjava2 program, these new classes have been tested
   * apart from the rest of the program. Using jmlrac on a few test seems 
   * to show that these classes are respecting there specification.
   * Anyway nothing has been done with escjava2.
   */ 
  
  //@ public invariant started ==> serverInstance != null;
  
  private static /*@ spec_public @*/ XmlRpcClientLite serverInstance;
  
  /**
   * Vector representing the parameters that are given to sammy
   * Notice that it's cleared after each call to execute()
   */
  
  private /*@spec_public*/ static /*@non_null*/Vector parameters = new Vector();
  
  static boolean debug = false;
  
  // special constructor for debugging
  public Sammy(boolean debug){
    Sammy.debug = debug;
  }
  
  /*@
   @ also
   @ assignable serverInstance, parameters;
   @*/
  public /*@non_null*/ProverResponse start_prover() {
    
    //++
    if(debug) System.out.println("Sammy::start_prover");
    //++
    
    killAnySammyAndStartNewOne();
    
    try { serverInstance = new XmlRpcClientLite("localhost",8000); }
    catch (Exception e) {
      System.err.println("Sammy::Failed to init XmlRpc");
      System.exit(0);
    }
    
    // parameters can't be empty (rpc stuff)
    parameters.add(""); 
    
    ProverResponse res = execute("start_solver");
    
    //++
    if(debug)
      if( res != ProverResponse.OK )
        System.out.println("Failed to init sammy");
      //++
      
    started = true;
    
    return res;
    
  }
  
  public /*@non_null*/ProverResponse set_prover_resource_flags(/*@non_null*/Properties properties) {
    
    //++
    if(debug) System.out.println("Sammy::set_prover_resource_flags");
    //++
    
    /* we have to create a big string containing
     * all the flags ( = key in the hashtable of Properties)
     * and their values ( = value)
     */
    Enumeration flags = properties.keys();
    
    String value = null;
    String currentFlag = null;
    
    for (  ; flags.hasMoreElements() ;) {
      
      try {
        currentFlag = (String)flags.nextElement();
        value = properties.getProperty( currentFlag );
      }
      
      catch (Exception e) {
        System.err.println("Sammy::Failed to inspect properties");
        System.exit(0);
      }
      
      parameters.add(currentFlag);
      
      /* flags can be without value, I guess it means that
       * value = ""
       */
      if( value != null ) 
        parameters.add(value);
      
    }
    
    ProverResponse res = execute("set_flags");
    
    //++
    if(debug)
      if( res!= ProverResponse.OK )
        System.out.println("Failed to set flags");
      //++
      
    return res;
  }
  
  public /*@non_null*/ProverResponse signature(/*@non_null*/Signature signature) {
    
    //++
    if(debug)
      System.out.print("Sammy::signature");
    //++
    
    /*
     * Calls for every decomposition of the signature
     * Notice that it returns the last response code, not the 
     * worst one of all calls.
     */
    
    // 	/*
    // 	 * Types
    // 	 */
    // 	parameters.add(signature.type());
    
    // 	ProverResponse res = execute("type_declaration");
    
    // 	//++
    // 	if(debug)
    // 	    if( res!= ProverResponse.OK )
    // 		System.out.println("Failed to set types");
    // 	//++
    
    // 	/*
    // 	 * Variables = 0 unary function in smt lib so there is no
    // 	 * need to declare them apart from functions.
    // 	 */
    
    // 	/*
    // 	 * Functions
    // 	 */
    // 	parameters.add(signature.function());
    
    // 	res = execute("func_declaration");
    
    // 	if(debug)
    // 	    if( res!= ProverResponse.OK )
    // 		System.out.println("Sammy::Failed to set functions");
    
    // 	/*
    // 	 * Predicates
    // 	 */
    // 	parameters.add(signature.predicate());
    
    // 	res = execute("pred_declaration");
    
    // 	//++
    // 	if(debug)
    // 	    if( res!= ProverResponse.OK )
    // 		System.out.println("Failed to set predicates");
    // 	//++
    
    // 	signature_defined = true;
    
    return null;
  }
  
  public /*@non_null*/ProverResponse declare_axiom(/*@non_null*/Formula formula) {
    
    /*
     * Is an axiom a declaration of a predicate ?
     */ 
    
    return null;
  }
  
  public /*@non_null*/ProverResponse make_assumption(/*@non_null*/Formula formula) {
    
    /*
     * Does it correspond to 'add_assertion' in sammy's interface ?
     */
    
    
    return null;
  }
  
  public /*@non_null*/ProverResponse retract_assumption(int count) {
    
    /*
     * == 'backtrack' in sammy's interface ?
     */
    
    return null;
  }
  
  public /*@non_null*/ProverResponse is_valid(/*@non_null*/Formula formula,
      Properties properties) {
    
    /*
     * I guess it corresponds to the 'query' function of 
     * sammy's interface ?
     */
    
    return null;
  }
  
    //@ also
    //@ assignable parameters, signature_defined, started;
  public /*@non_null*/ProverResponse stop_prover() {
    
    //++
    if(debug) System.out.println("Sammy::stop_prover");
    //++
    
    int result;
    
    // parameters can't be empty (rpc stuff)
    parameters.add(""); 
    
    ProverResponse res = execute("stop_solver");
    
    if(debug)
      if( res != ProverResponse.OK )
        System.out.println("Sammy::Failed to stop");
      
    started = false;
    signature_defined = false;
    
    return res;
  }
  
  /**
   * call to sammy using xml rpc
   * commands available :
   * start_solver
   * set_flags
   * type_declaration
   * func_declaration
   * add_assertion
   * query
   * backtrack
   * stop_solver
   * See sammy's cvcl/smt/README.TXT for further informations
   */ 
  
  /*@   requires serverInstance != null;
   @   requires !parameters.isEmpty();
   @   ensures \result == ProverResponse.OK ||
   @   \result == ProverResponse.FAIL ||
   @   \result == ProverResponse.YES ||
   @   \result == ProverResponse.NO ||
   @   \result == ProverResponse.COUNTER_EXAMPLE ||
   @   \result == ProverResponse.SYNTAX_ERROR ||
   @   \result == ProverResponse.PROGRESS_INFORMATION ||
   @   \result == ProverResponse.TIMEOUT ||
   @   \result == ProverResponse.INCONSISTENCY_WARNING;
   @   assignable parameters;
   @*/
  private ProverResponse execute(/*@ non_null @*/ String cmd ){
    
    // TODO : improve that
    if( parameters.size() > 1)
      flatParameters();
    
    //++
    System.out.println("Sammy::execute   "+cmd+parameters.toString());
    //++
    
    Integer res = new Integer(-1); // negative attitude
    
    try { res = (Integer)serverInstance.execute("sammy."+cmd,parameters); }
    catch (Exception e) {
      System.err.println("Sammy::Error during rpc call\n"+e);
      System.exit(0);
    }
    
    //++
    if(debug)
      System.out.println("=> return value = "+res.intValue());
    //++
    
    // clean the parameters before next call
    parameters.clear();
    
    return SammyResponse.factory(res.intValue());
  }
  
  /*
   * Temporary ?
   */ 
  /*@
   @ assignable parameters;
   @*/
  private void flatParameters() {
    
    Iterator i = parameters.iterator();
    StringBuffer res = new StringBuffer() ;
    String temp = null ;
    
    for( ; i.hasNext();) {
      
      try { temp = (String)i.next(); }
      catch (Exception e) {
        System.err.println("Sammy::Error during flattening parameters\n"+e);
        System.exit(0);
      }
      
      // without the first space, it fails
      // (sammy parsing issue I guess)
      res.append(" ").append(temp);
      
    }
    
    parameters.clear();
    parameters.add(res.toString());
    
    //++
    if(debug)
      System.out.println("Sammy::flatParameters   "+res);
    //++
    
  }
  
  /*
   * Isn't it a good name ?
   */
  private /*@ pure @*/ int killAnySammyAndStartNewOne(){
    
    //++
    if(debug)
      System.out.println("Sammy::killAnySammyAndStartNewOne");
    //++
    
    /*
     * TODO : Is it the right way to do that, and what about Windows ?
     */
    
    if( System.getProperty("os.name").startsWith("Linux") ||  System.getProperty("os.name").startsWith("Mac") ){
      Runtime r = Runtime.getRuntime();
      
      /*
       * TODO : improve that
       * for the moment the sleep is required, otherwise there is some issues
       * I guess that sammy needs a little time to get ready to accept rpc calls
       */ 
      try{
        r.exec("killall sammy");
        r.exec("sammy");
        Thread.sleep(1000);
      }
      catch(Exception e){
        System.out.println(e);
        System.out.println("Sammy::Failed to kill existing sammy and starting a new one");
        System.exit(0);
      }
    }
    
    
    return 0;
    
  }
  
  /*
   * Test
   */
  
  static public void main(String[] argv){
    
    long startTime = java.lang.System.currentTimeMillis();
    Sammy sammy = new Sammy(true);
    
    /* Test 1 */
    // 	sammy.start_prover();
    
    // 	parameters.add("(exp real real real)");
    // 	parameters.add("(pc real)");
    // 	sammy.execute("func_declaration");
    
    // 	parameters.add("(greater_than real real)");
    // 	sammy.execute("pred_declaration");
    
    // 	parameters.add("(or (not true) (= (+ 1 1) 2))");
    // 	sammy.execute("add_assertion");
    
    // 	parameters.add("(= 1 2)");
    // 	sammy.execute("add_assertion");
    
    // 	sammy.stop_prover();
    
    /* Test 2, given by Cesare */
    sammy.start_prover();
    
    parameters.add("(x_0) (x_1)  (x_2)  (x_3)  (x_4)  (x_5)  (x_11)  (x_12)  (x_14)  (x_15)  (x_16)  (x_17)  (x_25)  (x_26)  (x_28)  (x_29)  (x_30)  (x_31)  (x_39)  (x_40)  (x_42)  (x_43)  (x_44)  (x_45)");
    sammy.execute("pred_declaration");
    
    parameters.add("(cvclZero Real)  (x_6 Real)  (x_7 Real)  (x_8 Real)  (x_9 Real)  (x_10 Real)  (x_13 Real)  (x_18 Real)  (x_19 Real)  (x_20 Real)  (x_21 Real)  (x_22 Real)  (x_23 Real)  (x_24 Real)  (x_27 Real)  (x_32 Real)  (x_33 Real)  (x_34 Real)  (x_35 Real)  (x_36 Real)  (x_37 Real)  (x_38 Real)  (x_41 Real)  (x_46 Real)  (x_47 Real)  (x_48 Real)  (x_49 Real)  (x_50 Real)  (x_51 Real)");
    sammy.execute("func_declaration");
    
    parameters.add("(flet ($cvcl_12 (not x_39)) (flet ($cvcl_13 (not x_40)) (flet ($cvcl_14 (and $cvcl_12 $cvcl_13)) (flet ($cvcl_45 (not x_42)) (flet ($cvcl_46 (not x_43)) (flet ($cvcl_47 (and $cvcl_45 $cvcl_46)) (flet ($cvcl_32 (not x_44)) (flet ($cvcl_33 (not x_45)) (flet ($cvcl_35 (and $cvcl_32 $cvcl_33)) (flet ($cvcl_17 (and (iff x_42 x_28) (iff x_43 x_29))) (flet ($cvcl_42 (not x_28)) (flet ($cvcl_41 (not x_29)) (flet ($cvcl_38 (and $cvcl_42 $cvcl_41)) (flet ($cvcl_7 (and (iff x_39 x_25) (iff x_40 x_26))) (flet ($cvcl_28 (not x_30)) (flet ($cvcl_26 (not x_31)) (flet ($cvcl_21 (and $cvcl_28 $cvcl_26)) (flet ($cvcl_43 (and $cvcl_42 x_29)) (flet ($cvcl_15 (and (iff x_44 x_30) (iff x_45 x_31))) (flet ($cvcl_30 (and $cvcl_28 x_31)) (flet ($cvcl_9 (not x_25)) (flet ($cvcl_8 (not x_26)) (flet ($cvcl_3 (and $cvcl_9 $cvcl_8)) (flet ($cvcl_10 (and $cvcl_9 x_26)) (flet ($cvcl_62 (and (iff x_28 x_14) (iff x_29 x_15))) (flet ($cvcl_83 (not x_14)) (flet ($cvcl_82 (not x_15)) (flet ($cvcl_79 (and $cvcl_83 $cvcl_82)) (flet ($cvcl_55 (and (iff x_25 x_11) (iff x_26 x_12))) (flet ($cvcl_73 (not x_16)) (flet ($cvcl_71 (not x_17)) (flet ($cvcl_66 (and $cvcl_73 $cvcl_71)) (flet ($cvcl_84 (and $cvcl_83 x_15)) (flet ($cvcl_60 (and (iff x_30 x_16) (iff x_31 x_17))) (flet ($cvcl_75 (and $cvcl_73 x_17)) (flet ($cvcl_57 (not x_11)) (flet ($cvcl_56 (not x_12)) (flet ($cvcl_51 (and $cvcl_57 $cvcl_56)) (flet ($cvcl_58 (and $cvcl_57 x_12)) (flet ($cvcl_103 (and (iff x_14 x_4) (iff x_15 x_5))) (flet ($cvcl_124 (not x_4)) (flet ($cvcl_123 (not x_5)) (flet ($cvcl_120 (and $cvcl_124 $cvcl_123)) (flet ($cvcl_96 (and (iff x_11 x_0) (iff x_12 x_1))) (flet ($cvcl_114 (not x_2)) (flet ($cvcl_112 (not x_3)) (flet ($cvcl_106 (and $cvcl_114 $cvcl_112)) (flet ($cvcl_125 (and $cvcl_124 x_5)) (flet ($cvcl_101 (and (iff x_16 x_2) (iff x_17 x_3))) (flet ($cvcl_116 (and $cvcl_114 x_3)) (flet ($cvcl_98 (not x_0)) (flet ($cvcl_97 (not x_1)) (flet ($cvcl_91 (and $cvcl_98 $cvcl_97)) (flet ($cvcl_99 (and $cvcl_98 x_1)) (flet ($cvcl_89 (< (- cvclZero x_6) 0)) (flet ($cvcl_88 (< (- cvclZero x_7) 0)) (flet ($cvcl_87 (< (- cvclZero x_8) 0)) (flet ($cvcl_92 (= (- x_9 cvclZero) 0)) (flet ($cvcl_0 (< (- x_32 x_33) 0)) (flet ($cvcl_1 (if_then_else $cvcl_0 (< (- x_32 x_34) 0) (< (- x_33 x_34) 0))) (flet ($cvcl_37 (= (- x_48 x_34) 0)) (flet ($cvcl_16 (= (- x_47 x_33) 0)) (flet ($cvcl_18 (= (- x_46 x_32) 0)) (flet ($cvcl_2 (= (- x_41 x_27) 0)) (flet ($cvcl_19 (= (- x_38 cvclZero) 0)) (flet ($cvcl_4 (= (- x_36 x_34) 0)) (flet ($cvcl_5 (= (- x_27 cvclZero) 0)) (flet ($cvcl_6 (< (- x_36 x_48) 0)) (flet ($cvcl_20 (= (- x_38 cvclZero) 1)) (flet ($cvcl_23 (not $cvcl_5)) (flet ($cvcl_25 (= (- x_38 cvclZero) 2)) (flet ($cvcl_127 (= (- x_41 cvclZero) 1)) (flet ($cvcl_27 (= (- x_38 cvclZero) 3)) (flet ($cvcl_11 (= (- x_27 cvclZero) 1)) (flet ($cvcl_29 (= (- x_38 cvclZero) 4)) (flet ($cvcl_130 (not $cvcl_11)) (flet ($cvcl_34 (= (- x_38 cvclZero) 5)) (flet ($cvcl_36 (= (- x_41 cvclZero) 0)) (flet ($cvcl_22 (= (- x_36 x_33) 0)) (flet ($cvcl_24 (< (- x_36 x_47) 0)) (flet ($cvcl_128 (= (- x_41 cvclZero) 2)) (flet ($cvcl_31 (= (- x_27 cvclZero) 2)) (flet ($cvcl_131 (not $cvcl_31)) (flet ($cvcl_39 (= (- x_36 x_32) 0)) (flet ($cvcl_40 (< (- x_36 x_46) 0)) (flet ($cvcl_129 (= (- x_41 cvclZero) 3)) (flet ($cvcl_44 (= (- x_27 cvclZero) 3)) (flet ($cvcl_132 (not $cvcl_44)) (flet ($cvcl_48 (< (- x_18 x_19) 0)) (flet ($cvcl_49 (if_then_else $cvcl_48 (< (- x_18 x_20) 0) (< (- x_19 x_20) 0))) (flet ($cvcl_78 (= (- x_34 x_20) 0)) (flet ($cvcl_61 (= (- x_33 x_19) 0)) (flet ($cvcl_63 (= (- x_32 x_18) 0)) (flet ($cvcl_50 (= (- x_27 x_13) 0)) (flet ($cvcl_64 (= (- x_24 cvclZero) 0)) (flet ($cvcl_52 (= (- x_22 x_20) 0)) (flet ($cvcl_53 (= (- x_13 cvclZero) 0)) (flet ($cvcl_54 (< (- x_22 x_34) 0)) (flet ($cvcl_65 (= (- x_24 cvclZero) 1)) (flet ($cvcl_68 (not $cvcl_53)) (flet ($cvcl_70 (= (- x_24 cvclZero) 2)) (flet ($cvcl_72 (= (- x_24 cvclZero) 3)) (flet ($cvcl_59 (= (- x_13 cvclZero) 1)) (flet ($cvcl_74 (= (- x_24 cvclZero) 4)) (flet ($cvcl_133 (not $cvcl_59)) (flet ($cvcl_77 (= (- x_24 cvclZero) 5)) (flet ($cvcl_67 (= (- x_22 x_19) 0)) (flet ($cvcl_69 (< (- x_22 x_33) 0)) (flet ($cvcl_76 (= (- x_13 cvclZero) 2)) (flet ($cvcl_134 (not $cvcl_76)) (flet ($cvcl_80 (= (- x_22 x_18) 0)) (flet ($cvcl_81 (< (- x_22 x_32) 0)) (flet ($cvcl_85 (= (- x_13 cvclZero) 3)) (flet ($cvcl_135 (not $cvcl_85)) (flet ($cvcl_86 (< (- x_8 x_7) 0)) (flet ($cvcl_90 (if_then_else $cvcl_86 (< (- x_8 x_6) 0) (< (- x_7 x_6) 0))) (flet ($cvcl_119 (= (- x_20 x_6) 0)) (flet ($cvcl_102 (= (- x_19 x_7) 0)) (flet ($cvcl_104 (= (- x_18 x_8) 0)) (flet ($cvcl_93 (= (- x_13 x_9) 0)) (flet ($cvcl_105 (= (- x_10 cvclZero) 0)) (flet ($cvcl_94 (= (- cvclZero x_6) 0)) (flet ($cvcl_95 (< (- cvclZero x_20) 0)) (flet ($cvcl_107 (= (- x_10 cvclZero) 1)) (flet ($cvcl_109 (not $cvcl_92)) (flet ($cvcl_111 (= (- x_10 cvclZero) 2)) (flet ($cvcl_113 (= (- x_10 cvclZero) 3)) (flet ($cvcl_100 (= (- x_9 cvclZero) 1)) (flet ($cvcl_115 (= (- x_10 cvclZero) 4)) (flet ($cvcl_136 (not $cvcl_100)) (flet ($cvcl_118 (= (- x_10 cvclZero) 5)) (flet ($cvcl_108 (= (- cvclZero x_7) 0)) (flet ($cvcl_110 (< (- cvclZero x_19) 0)) (flet ($cvcl_117 (= (- x_9 cvclZero) 2)) (flet ($cvcl_137 (not $cvcl_117)) (flet ($cvcl_121 (= (- cvclZero x_8) 0)) (flet ($cvcl_122 (< (- cvclZero x_18) 0)) (flet ($cvcl_126 (= (- x_9 cvclZero) 3)) (flet ($cvcl_138 (not $cvcl_126)) (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not (< (- x_9 cvclZero) 0)) (<= (- x_9 cvclZero) 3)) (not (< (- x_13 cvclZero) 0))) (<= (- x_13 cvclZero) 3)) (not (< (- x_27 cvclZero) 0))) (<= (- x_27 cvclZero) 3)) (not (< (- x_41 cvclZero) 0))) (<= (- x_41 cvclZero) 3)) $cvcl_91) $cvcl_106) $cvcl_120) $cvcl_89) $cvcl_88) $cvcl_87) $cvcl_92) (or (and (and (and (and (and (and (and (and (and (= (- x_49 cvclZero) 0) (if_then_else $cvcl_1 (if_then_else $cvcl_0 (< (- x_36 x_32) 0) (< (- x_36 x_33) 0)) (< (- x_36 x_34) 0))) (if_then_else $cvcl_1 (if_then_else $cvcl_0 (= (- x_50 x_32) 0) (= (- x_50 x_33) 0)) (= (- x_50 x_34) 0))) $cvcl_7) $cvcl_15) $cvcl_17) $cvcl_37) $cvcl_16) $cvcl_18) $cvcl_2)  (and (and (= (- x_49 cvclZero) 1) (or (or (and (and (and (and (and (= (- x_51 cvclZero) 1) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_19 $cvcl_3) $cvcl_4) $cvcl_5) x_39) $cvcl_13) $cvcl_6) (<= (- x_48 x_36) 2)) $cvcl_2)  (and (and (and (and (and (and $cvcl_20 $cvcl_3) $cvcl_4) $cvcl_23) $cvcl_6) $cvcl_2) $cvcl_7) )  (and (and (and (and (and (and (and $cvcl_25 x_25) $cvcl_8) $cvcl_4) $cvcl_12) x_40) $cvcl_127) (<= (- x_36 x_48) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_27 $cvcl_10) $cvcl_4) $cvcl_11) x_39) x_40) $cvcl_6) $cvcl_2) )  (and (and (and (and (and (and $cvcl_29 $cvcl_10) $cvcl_4) $cvcl_130) $cvcl_14) $cvcl_6) $cvcl_2) )  (and (and (and (and (and (and $cvcl_34 x_25) x_26) $cvcl_4) $cvcl_14) $cvcl_36) $cvcl_6) )) $cvcl_15) $cvcl_16) $cvcl_17) $cvcl_18)  (and (and (and (and (and (= (- x_51 cvclZero) 2) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_19 $cvcl_21) $cvcl_22) $cvcl_5) x_44) $cvcl_33) $cvcl_24) (<= (- x_47 x_36) 2)) $cvcl_2)  (and (and (and (and (and (and $cvcl_20 $cvcl_21) $cvcl_22) $cvcl_23) $cvcl_24) $cvcl_2) $cvcl_15) )  (and (and (and (and (and (and (and $cvcl_25 x_30) $cvcl_26) $cvcl_22) $cvcl_32) x_45) $cvcl_128) (<= (- x_36 x_47) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_27 $cvcl_30) $cvcl_22) $cvcl_31) x_44) x_45) $cvcl_24) $cvcl_2) )  (and (and (and (and (and (and $cvcl_29 $cvcl_30) $cvcl_22) $cvcl_131) $cvcl_35) $cvcl_24) $cvcl_2) )  (and (and (and (and (and (and $cvcl_34 x_30) x_31) $cvcl_22) $cvcl_35) $cvcl_36) $cvcl_24) )) $cvcl_7) $cvcl_37) $cvcl_17) $cvcl_18) )  (and (and (and (and (and (= (- x_51 cvclZero) 3) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_19 $cvcl_38) $cvcl_39) $cvcl_5) x_42) $cvcl_46) $cvcl_40) (<= (- x_46 x_36) 2)) $cvcl_2)  (and (and (and (and (and (and $cvcl_20 $cvcl_38) $cvcl_39) $cvcl_23) $cvcl_40) $cvcl_2) $cvcl_17) )  (and (and (and (and (and (and (and $cvcl_25 x_28) $cvcl_41) $cvcl_39) $cvcl_45) x_43) $cvcl_129) (<= (- x_36 x_46) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_27 $cvcl_43) $cvcl_39) $cvcl_44) x_42) x_43) $cvcl_40) $cvcl_2) )  (and (and (and (and (and (and $cvcl_29 $cvcl_43) $cvcl_39) $cvcl_132) $cvcl_47) $cvcl_40) $cvcl_2) )  (and (and (and (and (and (and $cvcl_34 x_28) x_29) $cvcl_39) $cvcl_47) $cvcl_36) $cvcl_40) )) $cvcl_7) $cvcl_37) $cvcl_15) $cvcl_16) )) (= (- x_50 x_36) 0)) )) (or (and (and (and (and (and (and (and (and (and (= (- x_35 cvclZero) 0) (if_then_else $cvcl_49 (if_then_else $cvcl_48 (< (- x_22 x_18) 0) (< (- x_22 x_19) 0)) (< (- x_22 x_20) 0))) (if_then_else $cvcl_49 (if_then_else $cvcl_48 (= (- x_36 x_18) 0) (= (- x_36 x_19) 0)) (= (- x_36 x_20) 0))) $cvcl_55) $cvcl_60) $cvcl_62) $cvcl_78) $cvcl_61) $cvcl_63) $cvcl_50)  (and (and (= (- x_35 cvclZero) 1) (or (or (and (and (and (and (and (= (- x_37 cvclZero) 1) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_64 $cvcl_51) $cvcl_52) $cvcl_53) x_25) $cvcl_8) $cvcl_54) (<= (- x_34 x_22) 2)) $cvcl_50)  (and (and (and (and (and (and $cvcl_65 $cvcl_51) $cvcl_52) $cvcl_68) $cvcl_54) $cvcl_50) $cvcl_55) )  (and (and (and (and (and (and (and $cvcl_70 x_11) $cvcl_56) $cvcl_52) $cvcl_9) x_26) $cvcl_11) (<= (- x_22 x_34) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_72 $cvcl_58) $cvcl_52) $cvcl_59) x_25) x_26) $cvcl_54) $cvcl_50) )  (and (and (and (and (and (and $cvcl_74 $cvcl_58) $cvcl_52) $cvcl_133) $cvcl_3) $cvcl_54) $cvcl_50) )  (and (and (and (and (and (and $cvcl_77 x_11) x_12) $cvcl_52) $cvcl_3) $cvcl_5) $cvcl_54) )) $cvcl_60) $cvcl_61) $cvcl_62) $cvcl_63)  (and (and (and (and (and (= (- x_37 cvclZero) 2) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_64 $cvcl_66) $cvcl_67) $cvcl_53) x_30) $cvcl_26) $cvcl_69) (<= (- x_33 x_22) 2)) $cvcl_50)  (and (and (and (and (and (and $cvcl_65 $cvcl_66) $cvcl_67) $cvcl_68) $cvcl_69) $cvcl_50) $cvcl_60) )  (and (and (and (and (and (and (and $cvcl_70 x_16) $cvcl_71) $cvcl_67) $cvcl_28) x_31) $cvcl_31) (<= (- x_22 x_33) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_72 $cvcl_75) $cvcl_67) $cvcl_76) x_30) x_31) $cvcl_69) $cvcl_50) )  (and (and (and (and (and (and $cvcl_74 $cvcl_75) $cvcl_67) $cvcl_134) $cvcl_21) $cvcl_69) $cvcl_50) )  (and (and (and (and (and (and $cvcl_77 x_16) x_17) $cvcl_67) $cvcl_21) $cvcl_5) $cvcl_69) )) $cvcl_55) $cvcl_78) $cvcl_62) $cvcl_63) )  (and (and (and (and (and (= (- x_37 cvclZero) 3) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_64 $cvcl_79) $cvcl_80) $cvcl_53) x_28) $cvcl_41) $cvcl_81) (<= (- x_32 x_22) 2)) $cvcl_50)  (and (and (and (and (and (and $cvcl_65 $cvcl_79) $cvcl_80) $cvcl_68) $cvcl_81) $cvcl_50) $cvcl_62) )  (and (and (and (and (and (and (and $cvcl_70 x_14) $cvcl_82) $cvcl_80) $cvcl_42) x_29) $cvcl_44) (<= (- x_22 x_32) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_72 $cvcl_84) $cvcl_80) $cvcl_85) x_28) x_29) $cvcl_81) $cvcl_50) )  (and (and (and (and (and (and $cvcl_74 $cvcl_84) $cvcl_80) $cvcl_135) $cvcl_38) $cvcl_81) $cvcl_50) )  (and (and (and (and (and (and $cvcl_77 x_14) x_15) $cvcl_80) $cvcl_38) $cvcl_5) $cvcl_81) )) $cvcl_55) $cvcl_78) $cvcl_60) $cvcl_61) )) (= (- x_36 x_22) 0)) )) (or (and (and (and (and (and (and (and (and (and (= (- x_21 cvclZero) 0) (if_then_else $cvcl_90 (if_then_else $cvcl_86 $cvcl_87 $cvcl_88) $cvcl_89)) (if_then_else $cvcl_90 (if_then_else $cvcl_86 (= (- x_22 x_8) 0) (= (- x_22 x_7) 0)) (= (- x_22 x_6) 0))) $cvcl_96) $cvcl_101) $cvcl_103) $cvcl_119) $cvcl_102) $cvcl_104) $cvcl_93)  (and (and (= (- x_21 cvclZero) 1) (or (or (and (and (and (and (and (= (- x_23 cvclZero) 1) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_105 $cvcl_91) $cvcl_94) $cvcl_92) x_11) $cvcl_56) $cvcl_95) (<= (- x_20 cvclZero) 2)) $cvcl_93)  (and (and (and (and (and (and $cvcl_107 $cvcl_91) $cvcl_94) $cvcl_109) $cvcl_95) $cvcl_93) $cvcl_96) )  (and (and (and (and (and (and (and $cvcl_111 x_0) $cvcl_97) $cvcl_94) $cvcl_57) x_12) $cvcl_59) (<= (- cvclZero x_20) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_113 $cvcl_99) $cvcl_94) $cvcl_100) x_11) x_12) $cvcl_95) $cvcl_93) )  (and (and (and (and (and (and $cvcl_115 $cvcl_99) $cvcl_94) $cvcl_136) $cvcl_51) $cvcl_95) $cvcl_93) )  (and (and (and (and (and (and $cvcl_118 x_0) x_1) $cvcl_94) $cvcl_51) $cvcl_53) $cvcl_95) )) $cvcl_101) $cvcl_102) $cvcl_103) $cvcl_104)  (and (and (and (and (and (= (- x_23 cvclZero) 2) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_105 $cvcl_106) $cvcl_108) $cvcl_92) x_16) $cvcl_71) $cvcl_110) (<= (- x_19 cvclZero) 2)) $cvcl_93)  (and (and (and (and (and (and $cvcl_107 $cvcl_106) $cvcl_108) $cvcl_109) $cvcl_110) $cvcl_93) $cvcl_101) )  (and (and (and (and (and (and (and $cvcl_111 x_2) $cvcl_112) $cvcl_108) $cvcl_73) x_17) $cvcl_76) (<= (- cvclZero x_19) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_113 $cvcl_116) $cvcl_108) $cvcl_117) x_16) x_17) $cvcl_110) $cvcl_93) )  (and (and (and (and (and (and $cvcl_115 $cvcl_116) $cvcl_108) $cvcl_137) $cvcl_66) $cvcl_110) $cvcl_93) )  (and (and (and (and (and (and $cvcl_118 x_2) x_3) $cvcl_108) $cvcl_66) $cvcl_53) $cvcl_110) )) $cvcl_96) $cvcl_119) $cvcl_103) $cvcl_104) )  (and (and (and (and (and (= (- x_23 cvclZero) 3) (or (or (or (or (or (and (and (and (and (and (and (and (and $cvcl_105 $cvcl_120) $cvcl_121) $cvcl_92) x_14) $cvcl_82) $cvcl_122) (<= (- x_18 cvclZero) 2)) $cvcl_93)  (and (and (and (and (and (and $cvcl_107 $cvcl_120) $cvcl_121) $cvcl_109) $cvcl_122) $cvcl_93) $cvcl_103) )  (and (and (and (and (and (and (and $cvcl_111 x_4) $cvcl_123) $cvcl_121) $cvcl_83) x_15) $cvcl_85) (<= (- cvclZero x_18) (~ 4))) )  (and (and (and (and (and (and (and $cvcl_113 $cvcl_125) $cvcl_121) $cvcl_126) x_14) x_15) $cvcl_122) $cvcl_93) )  (and (and (and (and (and (and $cvcl_115 $cvcl_125) $cvcl_121) $cvcl_138) $cvcl_79) $cvcl_122) $cvcl_93) )  (and (and (and (and (and (and $cvcl_118 x_4) x_5) $cvcl_121) $cvcl_79) $cvcl_53) $cvcl_122) )) $cvcl_96) $cvcl_119) $cvcl_101) $cvcl_102) )) (= (- x_22 cvclZero) 0)) )) (or (or (or (or (or (or (or (or (or (or (or (and (and x_39 x_40) (not $cvcl_127))  (and (and x_44 x_45) (not $cvcl_128)) )  (and (and x_42 x_43) (not $cvcl_129)) )  (and (and x_25 x_26) $cvcl_130) )  (and (and x_30 x_31) $cvcl_131) )  (and (and x_28 x_29) $cvcl_132) )  (and (and x_11 x_12) $cvcl_133) )  (and (and x_16 x_17) $cvcl_134) )  (and (and x_14 x_15) $cvcl_135) )  (and (and x_0 x_1) $cvcl_136) )  (and (and x_2 x_3) $cvcl_137) )  (and (and x_4 x_5) $cvcl_138) )))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))");
    sammy.execute("add_assertion");
    
    sammy.stop_prover();
    long endTime = java.lang.System.currentTimeMillis();
    
    long totalTime = endTime - startTime;
    
    System.out.println("Time spend : "+totalTime+" milliseconds");
    
  }
  
}
