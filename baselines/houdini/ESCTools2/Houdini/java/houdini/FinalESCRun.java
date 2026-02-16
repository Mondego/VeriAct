/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import houdini.util.*;
import java.net.*;
import java.util.*; 
import java.io.*;
import javafe.*;

public  class FinalESCRun extends javafe.FrontEndTool {
    
    public String name() { return "FinalESCRun"; }

    public Options makeOptions() {
        return new FinalESCRunOptions();
    }

    private static FinalESCRunOptions options() {
        return (FinalESCRunOptions) options;
    }
   
    /**
     ** A tool's main entry point; <code>args</code> are the
     ** command-line arguments we have been invoked with. <p> 
     **/
    public void frontEndToolProcessing (ArrayList args) {
	Assert.notFalse(options().hostsFile!=null, "No host file specified");
	Assert.notFalse(options().filesFile!=null, "No files file specified");
	Assert.notFalse(options().command.indexOf("%1")!=-1, "command does not have %1");
	Assert.notFalse(options().command.indexOf("%2")!=-1, "command does not have %2");
	makeFileLists();
	makeCommands();
    }

    public static void main(String st[]) {
	FinalESCRun s = new FinalESCRun();
	s.run(st);
    }

    /** Vector of host names */
    Vector hosts = new Vector();

    /** Array of file lists, one for each host */
    String files[];
    
    void makeFileLists() {
	try {
	    log("reading hosts file");
	    DataInputStream in = Utility.getInputStream(options().hostsFile);
	    while (true) {
		String s = in.readLine();
		if (s == null) break;
		if (s.equals("")) continue;
		hosts.addElement(s);
	    }
	    in.close();
	} catch (IOException e) {
	    Assert.fail(e);
	}	
	log("making file lists");
	files = new String[hosts.size()];
	for (int i = 0; i < files.length; i++) {
	    files[i] = "";
	}
	
	try {
	    int count = 0;
	    DataInputStream in = Utility.getInputStream(options().filesFile);
	    while (true) {
		String s = in.readLine();
		if (s == null) break;
		String files_count = files[count] + " " + s;
		files[count] = files_count; // work around += bug
		count = (count + 1) % files.length;
	    }
	    in.close();
	} catch (IOException e) {
	    Assert.fail(e);
	}	
    }

    //@ requires files != null;
    void makeCommands() {
	try {
	    log("generating script");
	    PrintStream out = 
		Utility.getPrintStream("esc.final.script");
	    for (int i = 0; i < hosts.size(); i++) {
		String cmd = options().command;
		if (files[i].length() == 0) continue;
		cmd = houdini.util.Utility.replaceString(cmd, "%1", (String)hosts.elementAt(i), true);
		cmd = houdini.util.Utility.replaceString(cmd, "%2", files[i], true);
		cmd = houdini.util.Utility.replaceString(cmd, "%3", ""+i, true);
		out.println(cmd);
	    }
	    out.close();
	} catch (IOException e) {
	    Assert.fail(e);
	    return;
	}
    }
	

    public void log(String s) {
        System.out.println("["+ new Date() + "] " + s);
    }
}

class FinalESCRunOptions extends Options
{
    /** Name of file with hosts in it */
    String hostsFile = null;

    /** Name of file with files in it */
    String filesFile = null;

    /** command to use for each client */
    String command = "echo %1 %2";

    final public int processOption(String option, String[] args, int offset) throws javafe.util.UsageError {
	if (option.equals("-debug")) {
	    Debug.debug = true;
	    return offset;
	} else if (option.equals("-nodebug")) {
	    Debug.debug = false;
	    return offset;
	} else if (option.equals("-hosts")) {
	    if (offset>=args.length) {
	        usage("FinalESCRun");
	        throw (new javafe.util.UsageError("-hosts: offset>=args.length"));
	    }
	    hostsFile =args[offset];
	    return offset+1;
	} else if (option.equals("-command")) {
	    if (offset>=args.length) {
	        usage("FinalESCRun");
	        throw (new javafe.util.UsageError("-command: offset>=args.length"));
	    }
	    command =args[offset];
	    return offset+1;
	} else if (option.equals("-files")) {
	    if (offset>=args.length) {
	        usage("FinalESCRun");
	        throw (new javafe.util.UsageError("-files: offset>=args.length"));
	    }
	    filesFile=args[offset];
	    return offset+1;
	} else if (option.equals("-log")) {
	    if (offset>=args.length) {
	        usage("FinalESCRun");
	        throw (new javafe.util.UsageError("-log: offset>=args.length"));
	    }
	    Log.logToStdout(args[offset], Log.LL_HIGH);
	    return offset+1;
	} 
        return super.processOption(option, args, offset);
    }

    public String showOptions(boolean all) {
        StringBuffer sb = new StringBuffer(super.showOptions(all));
        String[][] data = {{"-debug","TODO"},
            {"-nodebug","TODO"},
            {"-log <key>","TODO"},
            {"-files <files file>","TODO"},
            {"-hosts <host file>","TODO"},
            {"-command <escjava command>","TODO"}};
        sb.append (data);
        return sb.toString();
    }
 }
