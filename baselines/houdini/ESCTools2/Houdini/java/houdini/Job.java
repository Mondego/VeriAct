/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini;

import java.util.*;
import houdini.util.*;

/**
 * This represents a job that the server can dispatch to a client.
 * It contains the name of the vc file, and a bunch of code for
 * monitoring/profiling.
 * the main entry points to this class are assign, complete, and 
 * update history.  these are the operations that assign the job
 * to a process and notify the job when the process is done.
 * These methods interact with the work list so that a client really 
 * only needs to ask the work list for a job and then interact with 
 * the job itself and not the work list.
 */
class Job {
    /** list to which the job belongs. */
    final WorkList owner;

    /** verification file to check. */
    /*# readonly */
    final String vcFile;

    /** printable name of the file. */
    /*# readonly */
    final String shortName;

    /** profile history of the job. This is a list of JobHistoryElem */
    final Vector history = new Vector();

    /** current profile information. */
    JobHistoryElem current = null;

    /** time at which the job was assigned. */
    long assignTime;
    
    public static String shortName(String s) {
	int lastSlash = s.lastIndexOf("/");
	if (lastSlash == -1) return s;
	return s.substring(lastSlash + 1);
    }

    public Job(WorkList owner, String vcFile) {
	this.owner = owner;
	this.vcFile = vcFile;
	String s = shortName(vcFile);
	shortName = s.substring(0, s.indexOf(".method.sx"));
    }


    /**
     * only the worklist should call this.
     * This is not synchronized to avoid deadlock with complete.
     */
    public void assign(int worker) {
	Assert.notFalse(current == null, this.historyToString());
	current = new JobHistoryElem(-1, -1, -1);
	history.addElement(current);
	current.currentWorker = worker;
	assignTime = current.timeStart = System.currentTimeMillis();
	if (Debug.debug) Log.log("job", historyToString());
    }
    

    /**
     * Call this when finished processing the job.  It
     * informs the work list of the completion.
     */
    synchronized public long complete() {
	Assert.notFalse(current != null, this.historyToString());
	long t = System.currentTimeMillis() - assignTime;
	current.timeToProve = (int)(System.currentTimeMillis() - current.timeStart);
	current = null;
	owner.complete(this);
	if (Debug.debug) Log.log("job", historyToString());
	return t;
    }


    /**
     * Inform the job that one proof iteration was completed,
     * but that the job was not fully completed. 
     */
    synchronized public void updateHistory() {
	Assert.notFalse(current != null, this.historyToString());
	current.timeToProve = (int)(System.currentTimeMillis() - current.timeStart);
	current = new JobHistoryElem(current.currentWorker, System.currentTimeMillis(), -1);
	history.addElement(current);
	owner.partialComplete(this);
    }

    public boolean inProcess() {
	return current != null;
    }
    
    public String getFile() {
	return vcFile;
    }
    
    public String toShortString() {
	return "[" + shortName + "]";
    }

    public String toString() {
	return "[Job " + toShortString() + "]";
    }

    public String historyToString() {
	String s = "[Job " + shortName;
	for (int i = 0; i < history.size(); i++) {
	    s += ((JobHistoryElem)history.elementAt(i));
	}
	s += "]";
	return s;
    }

   public String historyToShortString() {
       StringBuffer sb = new StringBuffer();
       sb.append("[").append(shortName).append(" ").
	   append(averageTime()).append(" (");
	for (int i = 0; i < history.size(); i++) {
	    JobHistoryElem j = (JobHistoryElem)history.elementAt(i);
	    if (j.timeToProve >= 0 && !j.aborted) {
		sb.append(j.timeToProve).append(" ");
	    }
	}	
	sb.append(")]");
	return sb.toString();
    }

    /**
     * Use this to notify the Job that it is being put on the work list,
     * regardless of whether or not it is currently assigned.  This
     * gives the job a chance to abort gracefully.
     */
    synchronized void puttingOnWorkList() {
	if (current != null) {
	    owner.complete(this);
	    current.aborted = true;
	}
	current = null;
	if (Debug.debug) Log.log("job", historyToString());
    }

    /**
     * returns the average time taken for each prove iteration.
     */
    public int averageTime() {
	int n = 0;
	int t = 0;
	for (int i = 0; i < history.size(); i++) {
	    JobHistoryElem j = (JobHistoryElem)history.elementAt(i);
	    if (j.timeToProve >= 0 && !j.aborted) {
		n++;
		t+=j.timeToProve;
	    }
	}
	return n > 0 ? t / n : -1;
    }
}

/**
 * profile information about 1 proof iteration.
 */
class JobHistoryElem {
    int currentWorker;
    int timeToProve;
    long timeStart;
    boolean aborted;
    
    JobHistoryElem(int worker, long timeStart, int timeToProve) {
	this.currentWorker = worker;
	this.timeToProve = timeToProve;
	this.timeStart = timeStart;
    }
    public String toString() {
	return "(worker = " + currentWorker + " " +
	    (aborted ? " aborted)" : "time = " + timeToProve + ")");
    }
}
