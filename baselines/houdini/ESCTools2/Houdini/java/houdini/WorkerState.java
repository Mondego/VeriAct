/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import houdini.util.*;
import houdini.comsock.*;
import java.util.*;
import java.io.*;

class WorkerState {

    /** server owning worker */
    HoudiniServer parent;

    /** command socket */
    SocketCommand command; 
    
    /** worker I.D. */
    int id;

    /** current status of the worker */
    int status;

    /** current job being executed if status is busy */
    Job currentJob = null;

    /** location of client */
    String clientComputer = "???";

    /** time that lastMessage was received from client */
    long lastMessage = System.currentTimeMillis(); // race on this is ok- only for reporting
    
    public static final int ALLOCATED = 0;
    public static final int CONNECTED = ALLOCATED + 1;
    public static final int BUSY = CONNECTED + 1;
    public static final int DEAD = BUSY + 1;
    
    /** String representation of the different status flags */
    private String rep[] = { "allocated", "connected", "busy", "dead" };
    
    public int getStatus() {
	return status;
    } 
    
    public Job getJob() {
	return currentJob;
    }
    
    private void touchDate() {
	lastMessage = System.currentTimeMillis();
    }
    
    public static String shortName(String s) {
	int lastSlash = s.lastIndexOf("/");
	if (lastSlash == -1) return s;
	return s.substring(lastSlash + 1);
    }

    /** connect to client and set up command handlers */
    SocketCommand makeConnection(int port) {
	final SocketCommand command = new SocketCommand(port);
	command.addCommand("hello", new HelloCommand());
	command.addCommand("vcdir?", new VCDirCommand());
	command.addCommand("vc", new VcCommand());
	command.addCommand("refuted?", new RefuteListCommand());
	command.addCommand("rerun", new DoneCommand());
	command.addCommand("done", new DoneCommand());
	command.addCommand("error", new ErrorCommand());
	command.addCommand("refuted", new RefutedCommand());
	command.setExitCommand(new ExitCommand());
	ShutDown.runOnExit(new Runnable() {
		public void run() {
		    try {
			if (command.isConnected()) command.close();
		    } catch (IOException e) {
			Assert.notify(e);
		    }
		}
	    });
	return command;
    }
    
    private void log(String s) {
	parent.logWithDate("[worker id="+id+"] " + s);
    }
    
    /**
     * Format:  "Hello <host>" -->  "<id>"
     */
    class HelloCommand extends Command {
	public String doIt(String args[]) {
	    touchDate();
	    Assert.notFalse(args.length == 2);
	    clientComputer = args[1];
	    log("Creating worker from computer " + clientComputer);
	    status = CONNECTED;
	    parent.cp.makeRoot();
	    return Integer.toString(id);
	}
    }
    
    /**
     * Format:  "vcdir?" -->  "<dir>"
     */    
    class VCDirCommand extends Command {
	public String doIt(String args[]) {
	    touchDate();
	    return parent.options().vcdir;
	}
    };
    
    /**
     * Format:  "refuted?" -->  "g1:g2:g3:..."
     */    
    class RefuteListCommand extends Command {
	public String doIt(String args[]) {
	    touchDate();
	    Assert.notFalse(args.length == 2);
	    Assert.notFalse(status == BUSY || status == CONNECTED);
	    StringBuffer b = new StringBuffer();
	    int index = Integer.parseInt(args[1]);
	    Vector list = parent.guards.getRefutedList();
	    int n = list.size();
	    for (int i = index; i < n; i++) {
		b.append(list.elementAt(i)).append(i != n-1?";":"");
	    }
	    return b.toString();
	}
    }
    
    /**
     * Format:  "vc?" -->  "fileName"
     */    
    class VcCommand extends Command {
	public String doIt(String args[]) {
	    touchDate();
	    Assert.notFalse(status == CONNECTED);
	    currentJob = parent.getJob(id);
	    log("Farming out " + currentJob);
	    status = BUSY;
	    return currentJob.getFile();
	}
    }
    
    /**
     * Format:  "error <str>" -->  "ack"
     */    
    class ErrorCommand extends Command {
	public String doIt(String args[]) {
	    touchDate();
	    Assert.notFalse(args.length == 2);
	    parent.cp.reportError(id, currentJob, args[1]);
	    return "ack";
	}
    }
    
    /**
     * Format:  "done t1 t2 t3 t4 t5 t6 t7" -->  "ack"
     */    
    class DoneCommand extends Command {
	public String doIt(String args[]) {
	    touchDate();
	    Assert.notFalse(args.length == 8);
	    StringBuffer b = new StringBuffer("proved ");
	    Assert.notFalse(currentJob != null);
	    b.append(currentJob).append(" in ");
	    b.append(args[1]).append(", ");
	    b.append(args[2]).append(", ");
	    b.append(args[3]).append(", ");
	    b.append(args[4]).append(", ");
	    b.append(args[5]).append(", ");
	    b.append(args[6]).append(", ");
	    b.append(args[7]);
	    log(b.toString());
	    parent.cp.reportTime(id, b.toString());
	    if (Debug.debug) Log.log("job", currentJob.historyToString());
	    if (args[0].equals("rerun")) {
		currentJob.updateHistory();
		log("rerun " + currentJob);
	    } else {  
		long t = currentJob.complete();
		log("completed " + currentJob + " in " + t + " ms.");
		currentJob = null;
		status = CONNECTED;
	    }
	    return "ack";
	}
    }
    
    /**
     * Format:  "refuted <guard>" -->  "rerun/ack"
     */        
    class RefutedCommand extends Command {
	public String doIt(String args[]) {
	    touchDate();
	    Assert.notFalse(args.length == 3);
	    String reply = "ack";
	    log("Refuted " + args[1]);
	    Vector deps = parent.guards.refute(args[1]);
	    if (deps != null) {
		if (args[2].length() > 0) {
		    parent.coordWriter.println(args[2]);
		    parent.coordWriter.flush();
		}
		for (int i = 0; i < deps.size(); i++) {
		    String s = (String)deps.elementAt(i);
		    if (parent.guards.isPositiveDependency(s, args[1])) {
			parent.list.add(s, false);
			log("Refuted " + args[1] + " --> " + shortName(s));
		    }
		}
		reply = "rerun";
	    }
	    return reply;
	}
    }
    
    /**
     * Format:  "exit" -->  ""
     */    
    class ExitCommand extends Command {
	public String doIt(String args[]) {
	    status = DEAD;
	    if (currentJob != null) {
		parent.abortJob(currentJob);
	    }
	    log("Died");
	    return "";
	}
    }
    
    public WorkerState(HoudiniServer parent, int port, int id) {
	this.parent = parent;
	this.command = this.makeConnection(port);
	this.id = id;
	this.status = 0;
	this.command.start();
    }
    
    public String toString() {
	return "[worker " + id + " --- job= " + currentJob + " --- status=" + rep[this.status] + " --- last message = " + 
	    (System.currentTimeMillis() - lastMessage) + "]";
    }
}
