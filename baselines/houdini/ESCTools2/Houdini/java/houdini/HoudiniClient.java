/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import houdini.comsock.*;
import houdini.util.*;
import java.io.*;
import java.util.*;
import java.net.*;

import escjava.prover.*;
import javafe.util.FatalError;

// We do not extend from OptionHandler any more:
// that abstraction is already been taken care of
// by splitting off OptionHandlerOptions.
class HoudiniClient extends javafe.FrontEndTool {

    public javafe.Options makeOptions() {
        return new OptionHandlerOptions();
    }

    public static OptionHandlerOptions options() {
        return (OptionHandlerOptions) options;
    }

    public void logWithDate(String s) {
        options().logFile.println("["+ Utility.getDateString() + "] " + s);
    }

    public static String shortName(String s) {
	int lastSlash = s.lastIndexOf("/");
	if (lastSlash == -1) return s;
	return s.substring(lastSlash + 1);
    }
  
    SocketSender sender; 

    /**
     * the id assigned to this client by the server
     */
    int id = -1;

    /**
     * the boolean truth values for the Houdini guards. 
     */
    Hashtable hguards;

    /**
     * The number of Houdini guards that have been refuted.  This is used
     * to tell the server how many guards we know are false when asking
     * for more refuted guards. 
     */
    int numRefuted = 0;

    /**
     * the Simplify subprocess.
     */
    Simplify S;

    /**
     * the map from numbers to fileNames used by correlated readers when the
     * verification conditions were generated. 
     */
    String fileNames[];

    /**
     * the time the current Simplify and subprocess was started. 
     */
    long simplifyStartTime;

    /**
     * Set to true if the client should rerun the prover on the
     * current vc when finished with the current iteration.
     */
    boolean rerun;
    
    /**
     * number of times to try connecting before giving up. 
     */
    static final int connectRetryNum = 10;

    /**
     * number of times to try proving a vc before giving up. 
     */
    static final int simplifyRetryNum = 5;

    /**
     * the maximum amount of time between Simplify restarts. 
     */
    static final long simplifyRestart = 1000 * 60 * 5;
    
    public String name() { return "HoudiniClient"; }     
    
    private void log(String s) {
	logWithDate("[worker id="+id+"] " + s);
    }
    
    /**
     * connect the socket to this server. 
     */
    private void connect() {
	int retry = 0;
	while (retry < connectRetryNum) {
	    try {
		log("trying to connect (retry = " + retry++ + ")");
		sender.connect();
		return;
	    } catch (ConnectException e) {
		log("failed");
		try { Thread.sleep(1000); } catch (Exception exc) { Assert.fail(exc); }
	    }
	}
	Assert.fail("Could not connect");
    }
    
    //@ helper
    /**
     * load the Houdini guards from the hguards.txt file and to set them all
     * to TRUE initially. 
     */
    private void makeHGuards() {
	log("Loading guard files");
	hguards = new Hashtable(50000);
	try {
	    DataInputStream in = Utility.getInputStream(options().vcdir+"hguards.txt");
	    while (true) {
		String s = in.readLine(); 
		if (s==null) {
		    break;
		}
		hguards.put(s, SExpOptimizer.tBool);
	    }
	} catch (IOException e) {
	    Assert.fail(e);
	} 
    }   
    
    //@ helper
    /**
     * connect to the server, retrieve ID and verification condition directory,
     * load the file map, and create the guard table. 
     */
    private void init() {
	try {
	    connect();
	    String id = sender.sendMessage("hello", InetAddress.getLocalHost().getHostName(), true);
	    this.id = Integer.parseInt(id);
	    log("got id");
	    options().vcdir = sender.sendMessage("vcdir?", true);
	    log("got vcdir");
	    if (options().logToFile) {
		options().logFile = 
		    new PrintStream(new BufferedOutputStream(new FileOutputStream(new File(options().vcdir, "client." + this.id + ".log"))), 
				    true);
	    }
	    log("Recieved id " + id);
	    log("Java version is " +
		System.getProperty("java.vendor") + ":" +
		System.getProperty("java.version"));
	    log("Recieved vcdir " + options().vcdir);
	    this.fileNames = loadFiles();
	    makeHGuards();
	} catch (Exception e) {
	    Assert.fail(e);
	}
    }
    
    /**
     * create the Simplify sub process. 
     */
    private void prepareSimplify() {
	if (S == null) {
            log("Starting Simplify");
	    S = new Simplify();
	    simplifyStartTime = System.currentTimeMillis();
	}
    }
    

    /**
     * close the Simplify sub process, absorbing any errors
     * that occur.  This will always set S to null.
     */
    private void closeSimplify() {
	try {
            log("Closing Simplify");
	    if (S != null) S.close();
	} catch (Exception e) {
	    error(e.toString());
	} finally {
	    S = null;
	}
    }
    
    //@ requires this.S != null;
    /**
     * load the backGroundPredicate and push it on to the Simplify stream. 
     */
    private void backGroundPush() throws IOException {
	log("Loading background predicate from " + options().univBackPredFile);
	PrintStream ps = S.subProcessToStream();
	DataInputStream in = Utility.getInputStream(options().univBackPredFile);
	getFileContents(in, ps);
	in.close();
    }        
    
    //@ requires this.S != null;
    /**
     * load in the class predicate from fileName and pass it to Simplify. 
     */
    private void classPredPush(String fileName) throws IOException {
	log("Loading class predicates from " + shortName(fileName));
	PrintStream ps = S.subProcessToStream();
	DataInputStream in =  Utility.getInputStream(fileName);
	Assert.notFalse(in.readLine().equals("*Class*"));
	in.readLine();
	getFileContents(in, ps);
	in.close();
    }        
    
    // avoid allocating every time
    static private byte b[] = new byte[32676];
    /**
     * read in the stream in and print it to out until it is empty. 
     */
    private void getFileContents(DataInputStream in, PrintStream out) throws IOException {
	while (true) {
	    int n = in.read(b);
	    if (n == -1) break;
	    out.write(b, 0, n);
	}
	out.println("\n");
    }
    

    // timer code- avoid allocating these every time    
    long time[] = new long[8];
    String stime[] = new String[7];
    
    /**
     * Reads an sexp from in, modifies it with the current guard info, and
     * then prints it to out.
     */
    private void getUpdatedFileContents(DataInputStream in, PrintStream out) throws IOException {
	try {
	    SExp s = SExpOptimizer.readFromFile(in);
	    time[3] = System.currentTimeMillis();
	    SExpOptimizer.modifyGuardsInPlace(s, hguards);
	    time[4] = System.currentTimeMillis();
	    if (Debug.debug) Log.log("sexp", "orig" + s);
	    s = SExpOptimizer.optimize(s);
	    if (Debug.debug) Log.log("sexp", "opt " + s);
	    time[5] = System.currentTimeMillis();
	    s.print(out);
	    out.println("\n");
	    time[6] = System.currentTimeMillis();
	} catch (SExpTypeError e) {
	    Assert.fail(e);
	}
    }
    
    
    /**
     * Sends a message to the server asking for the new refuted guards.
     * The server replies with a list of the for g1;g2;g3;...;gn
     */
    private void updateRefutedList() throws IOException {
	String s = sender.sendMessage("refuted?",
				      Integer.toString(numRefuted),
				      true);
	String st[] = houdini.util.Utility.stringListToArray(s, ";");
	log("Adding to refuted list: " + s);
	int n = st.length;
	for (int i = 0; i < n; i++) {
	    hguards.put(st[i], SExpOptimizer.fBool);
	}       
	numRefuted += n;
    }
    
    /**
     * Reads the class name from in, then loads that class predicate
     * and pushes it in Simplify.
     */
    private void handleClassPred(DataInputStream in) throws IOException {
	String className = in.readLine();
	className = options().vcdir + className.substring(className.indexOf("@") + 1) + ".class.sx";
	classPredPush(className);
    }
    
    /**
     * Takes the name of a method's .sx file, and runs simplify on it.
     * This process is repeated until running simplify does not refute
     * any additinal guards.
     */
    //@ requires this.S != null;
    private void proveFile(String fileName) throws IOException, SExpTypeError {
	String shortFileName = shortName(fileName);
	boolean firstTime = true;
	rerun = true;
	boolean bgPushed = false;
	while (rerun) {
	    rerun = false;
	    time[0] = System.currentTimeMillis();
	    updateRefutedList();
	    time[1] = System.currentTimeMillis();
	    log("Proving file " + shortFileName);
	    DataInputStream in =  Utility.getInputStream(fileName);
	    Assert.notFalse(in.readLine().equals("*Method*"));
	    if (checkForRestart() || firstTime) {
		handleClassPred(in);
		firstTime = false;
	    } else {
		in.readLine();
		S.subProcessToStream().println("\n");
	    }
	    in.close();
	    time[2] = System.currentTimeMillis();
	    log("Done with class pred");
	    in =  Utility.getInputStream(fileName + "x");
	    getUpdatedFileContents(in, S.subProcessToStream());
	    S.startProve();
	    Enumeration results = S.streamProve();
	    exhaust(results);
	    time[7] = System.currentTimeMillis();
	    for (int i = 0; i < 7; i++) {
		stime[i] = Long.toString(time[i+1]-time[i]);
	    }
	    String reply = sender.sendMessage(rerun?"rerun":"done", stime, true);
	}
	S.subProcessToStream().println("(BG_POP)");
	log("Done with " + shortFileName);
    }

    /**
     * Sends errorLabel to the server if it is a houdini guard.
     * This also generates an escjava-like error message and
     * passes it to the server as well.
     */
    void sendPossibleLabel(String errorLabel) throws IOException {
	int k = errorLabel.indexOf('@');
	if (k == -1) {
	    // trace label...
	    return;
	}
	String s = errorLabel.substring(0, k);
	
	k = s.indexOf(':');
	if (k != -1) {
	    String label = "G_" + s.substring(k+1);
	    if (hguards.get(label)==null) {
		log("Refuted non-Houdini label " + label);
		return;
	    }
	    
	    log("Refuted label " + label);
	    ByteArrayOutputStream sw = new ByteArrayOutputStream();
	    PrintStream pw = new PrintStream(sw);
	    escjava.translate.ErrorMsg.houdiniPrint(errorLabel,
						    pw,
						    fileNames);
	    pw.close();
	    sw.close();
	    String msg = sw.toString();
	    String st[] = { label, msg };
	    String reply = sender.sendMessage("refuted",
					      st,
					      true);	    
	    if (reply.equals("rerun")) rerun = true;
	}
    }
    
    /**
     * exhaust the CounterExamples for one run of the prover,
     * sending each label in the CE's to the server.
     */
    void exhaust(Enumeration results) throws IOException {
	try {
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
		    break;
		}
		case SimplifyOutput.COUNTEREXAMPLE: {
		    SimplifyResult sr = (SimplifyResult)so;
		    SList labels = sr.getLabels();
		    if (labels==null) break;
		    int n = labels.length();
		    for (int i = 0; i < n; i++) {
			String s = labels.at(i).toString();
			sendPossibleLabel(s);
		    }
		    break;
		}
		case SimplifyOutput.EXCEEDED_PROVER_KILL_TIME: {
		    break;
		}
		case SimplifyOutput.EXCEEDED_PROVER_KILL_ITER: {
		    break;
		}
		case SimplifyOutput.REACHED_CC_LIMIT:
		    break;
		case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_TIME: {
		    break;
		}
		case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_ITER: {
		}
		case SimplifyOutput.WARNING_TRIGGERLESS_QUANT: {
		    break;
		}
		default:
		    Assert.fail("unexpected type of Simplify output");
		    break;
		}
	    }
	} catch (SExpTypeError e) {
	    Assert.fail(e);
	}
    }    
    
    private boolean checkForRestart() throws IOException {
	if (System.currentTimeMillis() - simplifyStartTime > simplifyRestart) {
	    closeSimplify();
	    prepareSimplify();
	    backGroundPush();
	    return true;
	}
	return false;
    }
    
    public void error(String s) {
	System.out.println ("HoudiniClient error: " + s);
	try {
	    sender.sendMessage("error", s, true);
	} catch (IOException e) {
	    Assert.fail(e);
	}
    }
    
    public HoudiniClient() {
	//options.processOptions(st);
	//sender = new SocketSender(options().host, options().port);
	//init();  
    }
    
    public final void frontEndToolProcessing (ArrayList args) {
	sender = new SocketSender(options().host, options().port);
	init();  
	String vcFile = "";
	int retry = 0;
	while (sender.isConnected()) {
	    if (retry == simplifyRetryNum) {
		log("FAILED: Too many retries on " + vcFile);
		error("FAILED: Too many errors: " + vcFile);	
		Assert.fail("[id = " + id + "] Too many retries on " + vcFile);
	    }
	    try {
		prepareSimplify();
		backGroundPush();
		while (true) {
		    if (retry == 0 || vcFile == null) {
			vcFile = sender.sendMessage("vc", true);
		    } else {
			log("Retry " + retry + " on " + vcFile);
		    }
		    proveFile(vcFile);
		    retry = 0;
		}
	    } catch (IOException e) {
		retry++;
		log("Failure: " + e);
		error(e.toString());		
	    } catch (FatalError e) {
		retry++;
		log("Failure: " + e);
		System.out.println ("Fatal error in client " + id);
		error(e.toString());		
	    } catch (Exception e) {
		retry++;
		log("Failure: " + e);
		error(e.toString());		
	    } finally {
		closeSimplify();
	    }
	}
	log("Connection lost");
    }
    
    
    public static void main(String st[]) {
	HoudiniClient s = new HoudiniClient();
	int exitcode = s.run(st);

        if (exitcode != 0)
	    System.exit(exitcode);
    }
    
    
    private String[] loadFiles() throws IOException {
	DataInputStream in =  Utility.getInputStream(options().vcdir+"files.txt");
	Vector v = new Vector();
	log("Loading file map");
	while (true) {
	    String s = in.readLine();
	    if (s == null) {
		break;
	    }
	    String name = s.substring(s.indexOf(" ") + 1);
	    v.addElement(name);
	}
	String st[] = new String[v.size()];
	v.copyInto(st);
	return st;
    }
    
    
}

//	HoudiniClient s = new HoudiniClient();
/*	try {
	int offset = s.processOptions(st);            
	s.guards = new GuardManager(s.vcdir);
	s.prepareSimplify();
	System.out.println("[simp started]");
	s.backGroundPush();
	s.proveFile("gvc/0.148.17.method.sx");
	System.out.println("[filed proved]");
	} catch (Exception e) {
	Assert.fail(e);
	}*/

