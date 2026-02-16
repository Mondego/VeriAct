/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import houdini.comsock.*;
import houdini.util.*;
import java.util.*;
import java.io.*;

/**
 * Implements the work list for the server.
 */
class WorkList {

    /** list of jobs that are currently assigned but which should be re-done. */
    private Vector pending = new Vector() /*# guarded_by list */ ;

    /** list of jobs currently assigned to clients. */
    private Vector assigned = new Vector() /*# guarded_by list */;

    /** hash table of all jobs, maps FileName -> Job */
    private Hashtable jobs = new Hashtable(1000);

    /** list of jobs that need to be processed. */
    private houdini.util.PriorityQueue list = new houdini.util.PriorityQueue() {
	public int compare(Object o, Object p) {
	    return ((Job)o).averageTime() - ((Job)p).averageTime();
	}
    } /*# guarded_by list */;

    /** jobs in a sorted order for printing */
    private OrderedVector sortedJobs = new OrderedVector() {
	public int compare(Object o, Object p) {
	    return ((Job)o).shortName.compareTo(((Job)p).shortName);
	}
    };

    /** object used to notify the server when the list is empty */
    private Object signalObject;
    
    /**
     * Pass in the signal object and a list of vc files. 
     */
    public WorkList(Object signalObject, String st[]) {
	this(signalObject);
        for (int i = 0; i < st.length; i++) {
            add(st[i], false);
        }
    }

    public WorkList(Object signalObject) {
	this.signalObject = signalObject;
    }
    
    public OrderedVector getSortedJobs() {
	return sortedJobs;
    }    
    
    /**
     * Add the job for file s to the worklist.  If this addition should
     * preempt any client working on the job (ie, assign it even if it 
     * is currently assigned to someone else), set preEmptive to true.
     * returns true if the job was preempted.
     */
    boolean add(String s, boolean preEmptive) {
	boolean preempted = false;
	synchronized(list) {
	    Job j = (Job)jobs.get(s);
	    if (j == null) {
		// new job
		jobs.put(s, j = new Job(this, s));
		sortedJobs.addOrderedElement(j);
	    } else {
		// old job
		if (j.inProcess()) {
		    // assigned to client right now
		    if (preEmptive) {
			preempted = true;
			j.puttingOnWorkList();
		    } else {
			if (!pending.contains(j)) pending.addElement(j);
			return false;
		    }
		}
	    } 
	    if (!list.contains(j)) {
		list.enqueue(j);
	    }
	}
	return preempted;
    }
    
    /**
     * Get element from the priority queue.
     */
    Job extractMax(int worker) {
	synchronized(list) {
	    Job j = (Job)list.dequeue(true);
	    j.assign(worker);
	    assigned.addElement(j);
	    return j;
	}
    }


    /**
     * Call this on job only AFTER EVERYTHING ELSE is done on the job.
     * Only the job itself should call this.
     */
    void complete(Job j) {
	synchronized(list) {
	    assigned.remove(j);
	    // re-add if pending.
	    if (pending.contains(j)) {
		add(j.getFile(), false);
		pending.remove(j);
	    }
	    // signal if list is empty.
	    if (numOutstanding() == 0) {
		synchronized(signalObject) {
		    signalObject.notifyAll();
		}
	    }
	}
    }

    /**
     * Call this on job only AFTER EVERYTHING ELSE is done.
     * Only the job itself should call this.
     */
    void partialComplete(Job j) {
      synchronized(list) {
	if (pending.contains(j)) {
	  pending.remove(j);
	}
      }
    }

    boolean isPending(Job j) {
      return pending.contains(j);
    }

    /** 
     * Return the number of jobs that are assigned, pending, or need
     * to be assigned.
     */
    int numOutstanding() {
	synchronized(list) {
	    return list.size() + pending.size() + assigned.size();
	}
    }
    
    public String toString() {
	String s = "Queue \n" + list + "\n"
	    + "Assigned \n" + assigned + "\n"
	    + "Pending \n" + pending + "\n";
	s += "Jobs \n";
	for (int i = 0; i < sortedJobs.size(); i++) {
	    s += ((Job)sortedJobs.elementAt(i)).historyToString() + "\n";
	}
	return s;
    }

    public String toHTML() {
	String s = "Queue \n" + list + "<BR>\n"
	    + "Assigned \n" + assigned + "<BR>\n"
	    + "Pending \n" + pending + "<BR>\n";
	return s;
    }

}


