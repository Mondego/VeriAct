/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import houdini.comsock.*;
import houdini.util.*;
import java.util.*;
import java.io.*;

import javafe.Options;

import escjava.prover.*;

class HoudiniServer extends javafe.FrontEndTool {

    GuardManager guards;

    /** vector of WorkerState */
    final Vector workers = new Vector();  

    /** count of how many workers have been created.  Use this for unique ids */
    int workerCounter = 0;

    /** signal object for when worlist is empty */
    final Object signal = new Object();

    final WorkList list = new WorkList(signal);

    /** time we started */
    final Date startTime = new Date();

    /** stream for all error messages */
    PrintWriter coordWriter;

    /** check pointer */
    CheckPoint cp;

    /** file name for guard info */
    static final String guardRestartFile = "GUARDS";

    /** file name of universal background pred */
    String backGroundPredicate = "UnivBackPred.ax";

    public void logWithDate(String s) {
        options().logFile.println("["+ Utility.getDateString() + "] " + s);
    }

    public static String shortName(String s) {
	int lastSlash = s.lastIndexOf("/");
	if (lastSlash == -1) return s;
	return s.substring(lastSlash + 1);
    }
 
    private void log(String s) {
        logWithDate(s);
    }

    /** return all .sx files in the vc directory */
    private String[] getVCFiles() {
        File f = new File(options().vcdir);
	houdini.util.Assert.notFalse(f.exists(), "gvc dir does not exist");
        return f.list(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return (name.endsWith("method.sx"));
            }
        });
    }

    /**
     * Return a job to process from the worklist
     */
    public Job getJob(int worker) {
        return this.list.extractMax(worker);
    }

    /** 
     * Abort a job.
     */
    public void abortJob(Job j) {
        if (this.list.add(j.getFile(), true)) {
	    log(j + " preempted");
	}
    }
    public Options makeOptions() {
        return new HoudiniServerOptions();
    }

    public static HoudiniServerOptions options() {
        return (HoudiniServerOptions) options;
    }
        
    /**
     * Process options, set up threads, and then wait for clients to connect.
     */
    public final void frontEndToolProcessing (ArrayList args) {
	try {
	    if (options().logToFile) {
		options().logFile = 
		    Utility.getPrintStream(new File(options().vcdir, "server.log"));
	    }
	} catch (IOException e) {
	    Assert.fail(e);
	}
	cp = new CheckPoint(this);
	Thread t = new Thread(new Main());
	t.start();
        while (true) {
            WorkerState ws = new WorkerState(this, options().port, workerCounter++);
            workers.addElement(ws);
        }
    }

    /**
     * make cononical sexp for the method file fileName.  This also sets all
     * non-houdini guards to true.
     */
    SExp makeCanonForm(String fileName) {
	try {
	    String shortFileName = shortName(fileName);
	    log("Computing canon form for " + fileName);
	    DataInputStream in = Utility.getInputStream(fileName);
	    Assert.notFalse(in.readLine().equals("*Method*"));
	    String className = in.readLine();
	    SExpParser parser = new SExpParser(in);
	    SExp s = parser.parse();
	    Assert.notFalse(s != null);
	    s = SExpOptimizer.canon(s);
	    SExpOptimizer.modifyGuardsInPlace(s, guards.getGuards());
	    DataOutputStream out = Utility.getOutputStream(fileName+"x");
	    SExpOptimizer.writeToFile(s, out);
	    out.close();
	    in.close();
	    log("Done with canon form for " + shortFileName);
	    return s;
	} catch (Exception e) {
	    Assert.fail(e);
	}
	return null; // no reached
    }

    /**
     * Compute deps for fileName.  If the sexp has been loaded pass in as s, otherwise
     * pass in null.
     */
    void makeDeps(String fileName, SExp s) {
	try {
	    String shortFileName = shortName(fileName);
	    log("Computing deps for " + shortFileName);
	    guards.computeDependencies(fileName, s);
	    log("Done with deps for " + shortFileName);
	} catch (Exception e) {
	    Assert.fail(e);
	}
    }
 

    public String toString() {
	String s = "Workers\n";
	for (int i = 0; i < workers.size(); i++) {
	    s += ((WorkerState)workers.elementAt(i)).toString() + "\n";
	}
	return s;
    }

    private void createCoordFile(boolean restart) throws IOException {
	String coord = (new File(options().vcdir, "coordinator.out")).getAbsolutePath();
	coordWriter = new PrintWriter(new FileWriter(coord, restart));
    }

    /** 
     * Main thread that creates all data structures and handles termination.
     */
    class Main implements Runnable {

	private void report() {
	    log("Running for " + (System.currentTimeMillis() - startTime.getTime()) / 1000 + " sec.");
	}

	//@ helper
	private void setup() {
	    try { 
		String[] f = getVCFiles();

		// try to load deps from file.  if it fails, stop the restart and do it 
		// all from scratch.
		if (options().restart) {
		    log("Loading deps from file " + options().vcdir + guardRestartFile);
		    try {
			guards = new GuardManager(new ObjectInputStream(new FileInputStream(options().vcdir + guardRestartFile)));
		    } catch (Exception e) {
			log("failed on reload.  starting from scratch");
			guards = new GuardManager(options().vcdir);
			options().restart = false;
		    }
		} else {
		    guards = new GuardManager(options().vcdir);
		}
	
		// make cononical file forms as needed (ie, if restarting don't redo work)
		for (int i = 0; i < f.length; i++) {
		    String name = (new File(options().vcdir, f[i])).getAbsolutePath();
		    if ((new File(options().vcdir, f[i]+"x")).exists() && options().restart) {
			log("Skipping prep of " + name);
		    } else {
			SExp s = makeCanonForm(name);
			if (!options().restart) makeDeps(name, s);
		    }
		    f[i] = name;
		}
		
		// write out guard info if not restarting
		if (!options().restart) {
		    log("Writing deps to file"); 
		    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(options().vcdir + guardRestartFile));
		    guards.writeToFile(out);
		    out.close();
		    log("Done");
		}

		log(guards.depsToString());
		createCoordFile(options().restart);
		
		// add every job to the work list
		for (int i = 0; i < f.length; i++) {
		    list.add(f[i], false);
		}
	    } catch (Exception e) {
		Assert.fail(e);
	    }
	}

	/**
	 * setup and then wait for everything to finish.
	 */
	public void run() {
	    long lastMajorReport;
	    log("Java version is " +
		System.getProperty("java.vendor") + ":" +
		System.getProperty("java.version"));
	    setup();
	    report();
	    synchronized(signal) {
		while (list.numOutstanding() != 0) {
		    try { signal.wait(1000 * 30); } catch (Exception e) {}
		    report();
		}
		Date endTime = new Date();
		log("End time is " + endTime);
		log("Elapsed time is "  
		    + (endTime.getTime() - startTime.getTime()) / (60 * 60 * 1000) + " hour. "
		    + ((endTime.getTime() - startTime.getTime()) / (60 * 1000)) % 60 + " min. "
		    + ((endTime.getTime() - startTime.getTime()) / 1000) % 60 + " sec.");
		ShutDown.exit(0);
	    }
	}
    }
    
    public static void main(String st[]) {
	HoudiniServer s = new HoudiniServer();
	s.run(st);
    }
    
    public String name() { return "HoudiniServer"; }

    public String showOptions(boolean all) {
	return options.showOptions(all);
    }
    
}

class HoudiniServerOptions extends OptionHandlerOptions
{
    /** true if we are trying to restart */
    boolean restart;

    public int processOption(String option, String[] args, int offset) throws javafe.util.UsageError {
	if (option.equals("-restart")) {
	    restart = true;
	    return offset;
	}
        return super.processOption(option, args, offset);
    }


    public String showOptions(boolean all) {
        StringBuffer sb = new StringBuffer(super.showOptions(all));
        String[][] data1 = {{"-restart", "TODO"}};
        sb.append(data1);
        return sb.toString();
    }
}
