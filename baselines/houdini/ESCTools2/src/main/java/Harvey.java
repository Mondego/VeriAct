package escjava.prover;

import java.util.Properties;
import java.util.Vector;
import java.util.Enumeration;

import java.io.*;

import java.lang.Process;

public class Harvey extends NewProver {
  
  SubProcess P; // connection to rv
  
  static boolean debug = false;
  
  // special constructor for debugging
  public Harvey(boolean debug){
    Harvey.debug = debug;
  }
  
  public /*@non_null*/ProverResponse start_prover() {
    
    //++
    if(debug) System.out.println("Harvey::start_prover");
    //++
    
    started = true;
    
    return HarveyResponse.factory(1);
    
  }
  
  public /*@non_null*/ProverResponse set_prover_resource_flags(/*@non_null*/Properties properties) {
    
    //++
    if(debug) System.out.println("Harvey::set_prover_resource_flags");
    //++
    
    return null;
  }
  
  public /*@non_null*/ProverResponse signature(/*@non_null*/Signature signature) {
    
    //++
    if(debug)
      System.out.print("Harvey::signature");
    //++
    
    return null;
  }
  
  public /*@non_null*/ProverResponse declare_axiom(/*@non_null*/Formula formula) {
    
    return null;
  }
  
  public /*@non_null*/ProverResponse make_assumption(/*@non_null*/Formula formula) {
    
    return null;
  }
  
  public /*@non_null*/ProverResponse retract_assumption(int count) {
    
    return null;
  }
  
  /*
   * Everything here is just for demo,
   * but it's ultra lame even compared to interacting with simplify
   */
  public /*@non_null*/ProverResponse is_valid(/*@non_null*/Formula formula,
      Properties properties) {
    
    PrintWriter file = null;
    
    Runtime r = Runtime.getRuntime();
    
    /* save formula on disk */
    try {
      
      file = new PrintWriter(new BufferedWriter
          (new FileWriter("temp-rv.rv")));
      
      file.println(formula.toString());
    }
    catch (Exception e) { System.out.println(e.toString()); }
    
    file.close();
    
    /* execute rvc */
    try {
      r.exec("rvc temp-rv.rv");
    }
    catch (Exception e) { System.out.println(e.toString());}
    
    /* result is written to "temp-rv.baf"
     * such interaction with the prover!!!
     */
    
    /* exec rv on the file generated */
    //P = new SubProcess("Harvey",
    //new String[] {"/usr/local/bin/rv","temp-rv.baf"},null);
    
    Process Q = null;
    
    try {	   
      Q = r.exec(new String[]{"/usr/local/bin/rv","temp-rv.baf"});
      
      Q.waitFor();
      
      /* check validity */
      InputStream i = Q.getInputStream();
      
      while( i.available() > 0 ){
        int next = i.read();
        char c = (char)next;
        System.out.print(c);
      }
      
    }
    catch (Exception e) { System.out.println(e.toString());}
    
    return null;
  }
  
  public /*@non_null*/ProverResponse stop_prover() {
    
    //++
    if(debug) System.out.println("Harvey::stop_prover");
    //++
    
    if(P!=null)
      P.close();
    
    return HarveyResponse.factory(1);
  }
  
  public static void main(String[] argv){
      
    
    Harvey harvey = new Harvey(true);
    
    harvey.start_prover();
    
    LineNumberReader lNR = null;
    String formulaString = null;
    StringBuffer formulaStringB = new StringBuffer();
    
    try {
      
      lNR = new LineNumberReader(new FileReader("test-rv.rv"));
      formulaString = new String();
      
      while( lNR.ready() ){
        
        formulaStringB.append("\n").append(lNR.readLine());
        
      }
      
      formulaString = formulaStringB.toString();
      
      harvey.is_valid(new Formula(formulaString),null); 
      
    }
    catch (Exception e) { 

	System.err.println(e);
	System.err.println("You seems to be trying to launch the harvey demo...");
	System.err.println("In order to be able to do that, you need to have harvey in /usr/local/bin/ under the name rv");
	System.err.println("And you need a proof example in a file called 'test-rv.rv' in ESCTools/Escjava/.");
	System.err.println("I know that's crappy..");
    }
    
    harvey.stop_prover();
    
    System.exit(0);
    
  }
  
}
