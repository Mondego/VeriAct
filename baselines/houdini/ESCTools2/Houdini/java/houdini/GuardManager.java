/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini;

import houdini.comsock.*;
import houdini.util.*;
import java.util.*;
import java.io.*;

import escjava.prover.*;

class GuardManager {
    
    /** dir where vc files live. */
    String dir;
    
    /** guards and their truth values */
    Hashtable guards;

    /** houdini guards and the truth values */
    Hashtable hguards;

    /** list of refuted guards */
    Vector refutedList;

    /** print stream where refuted guards will be printed */
    PrintStream refutedStream;

    /** file name -> vector guards */
    Hashtable assumes;

    /** file name -> vector of guards */
    Hashtable asserts;

    /** hguard -> vector of file names */
    Hashtable deps;
   
    public static String shortName(String s) {
	int lastSlash = s.lastIndexOf("/");
	if (lastSlash == -1) return s;
	return s.substring(lastSlash + 1);
    }

    /**
     * load guards from fileName, putting them in g.  Will not put a guard
     * into g if it already is contained in hguards.
     */
    private void loadGuards(String fileName, Hashtable g) {
        try {
	    if (Debug.debug) Log.log("guard", "loading guards from " + fileName);
	    DataInputStream in = Utility.getInputStream(fileName);
	    while (true) {
	        String s = in.readLine(); 
	        if (s==null) {
		    break;
	        }
		if (Debug.debug) Log.log("guard", "found guard " + s);
		// only put in guards/hguards -- not both
	        if (hguards.get(s) == null) {
		    g.put(s, SExpOptimizer.tBool);
		}
		deps.put(s, new Vector());
	    }
	} catch (IOException e) {
	    Assert.fail(e);
	}
    }
    
    void addHGuard(String s) {
        hguards.put(s, SExpOptimizer.tBool);
    }

    /**
     * Set the value of a guard
     */
    public void set(String guard, boolean value) {
	if (hguards.get(guard)!=null) {
	    hguards.put(guard, value?SExpOptimizer.tBool:SExpOptimizer.fBool);
	} else if (guards.get(guard)!=null) {
	    guards.put(guard, value?SExpOptimizer.tBool:SExpOptimizer.fBool);
	} else {
	    Assert.fail("no such guard:" + guard);
	}
    }

    /**
     * refute a guard.  This will negate it in the guard tables, print it
     * to the refuted stream.  This will return a vector of files that
     * depend on the guard if it is the first time it has been refuted.
     * It will return null otherwise.
     */
    public Vector refute(String guard) {
      synchronized (refutedList) {
        Boolean s = (Boolean)hguards.get(guard);
	if (s != null) {
	  if (s != SExpOptimizer.fBool) {
	    hguards.put(guard, SExpOptimizer.fBool);
	    refutedList.addElement(guard);
	    refutedStream.println(guard);
	    refutedStream.flush();
	    return (Vector)deps.get(guard);
	  }
	} else if (guards.get(guard)!=null) {
	    Assert.fail("no such hguard:" + guard);
	}
	return null; 
      }
    }
 
    public Vector getRefutedList() {
        return this.refutedList;
    }

    public Hashtable getHGuards() {
        return this.hguards;
    }

    public Hashtable getGuards() {
        return this.guards;
    }
   
    /**
     * pass in true if we are attempting a restart.  This will set
     * up the data structures and load what it can from files.
     */
    //@ helper
    final private void init(boolean restart) {
	try {
	    // remove old backup file 
	    (new File(dir+"refuted.txt.old")).delete();

	    // rename refuted file to backup file name
	    File f = new File(dir+"refuted.txt");
	    if (f.exists()) {
		f.renameTo(new File(dir+"refuted.txt.old"));
	    }

	    // open refuted stream
	    this.refutedStream = Utility.getPrintStream(dir+"refuted.txt");	    

	    // if we are restarting, reload the backup file.
	    if (restart) {
		try {
		    DataInputStream in = Utility.getInputStream(dir+"refuted.txt.old");
		    while (true) {
			String s = in.readLine();
			if (s == null) break;
			refute(s);
		    }
		    in.close();
		} catch (IOException e) {
		}
	    }
	    System.out.println(refutedList);
	} catch (Exception e) {
	    Assert.fail(e);
	}
    }

    /**
     * Construct a new Guard Manager.  Use this when not restarting.
     */
    GuardManager(String dir) {
	this();
	this.dir = dir;
	loadGuards(dir+"hguards.txt", hguards);
	loadGuards(dir+"guards.txt", guards);
	init(false);
    }    
    
    private GuardManager() {	
	guards = new Hashtable(50000);
	hguards = new Hashtable(50000);

	assumes = new Hashtable();
	asserts = new Hashtable();
	deps = new Hashtable();
	refutedList = new Vector();
    }    

    /**
     * Construct a new guard manager for restarting.  ios should
     * be the stream where the old data is kept.
     */
    GuardManager(ObjectInputStream ios) {	
	this.readFromFile(ios);
	init(true);
    }    

    
    public String toString() {
	return "Refuted\n" + refutedList + "\nsize = " + refutedList.size();
    }

    public String depsToString() {
	StringBuffer s = new StringBuffer();
	s.append("Dependencies by HGuard\n");
	for (Enumeration e = hguards.keys(); e.hasMoreElements(); ) {
	    String o = (String)e.nextElement();
	    s.append(" " + o + ": [");
	    Vector v = (Vector)deps.get(o);
	    for (int i = 0; i < v.size(); i++) {
		String name = (String)v.elementAt(i);
		s.append(shortName(name)).append(i < v.size()-1?", " : "");
	    }
	    s.append("]\n");
	}
        s.append("\nDependencies by File\n");
	for (Enumeration e = assumes.keys(); e.hasMoreElements(); ) {
	    String name = (String)e.nextElement();
	    s.append(" ").append(shortName(name)).append(": ").append(depsToString(name)).append("\n");
	}
	return s.toString();
    }

    public String depsToString(String fileName) {
	return (((Vector)assumes.get(fileName)).size() + ((Vector)asserts.get(fileName)).size()) + "    assumes=" + assumes.get(fileName) + " asserts=" + asserts.get(fileName);
    }    

    /**
     * return true if fileName depends on guard positively. 
     */
    public boolean isPositiveDependency(String fileName, String guard) {
	Vector v = (Vector)asserts.get(fileName);
	Assert.notFalse(v != null);
	return v.contains(guard);
    }
    
    static public void main(String st[]) throws SExpTypeError {
        GuardManager g = new GuardManager();
        g.addHGuard("|G_1.1.1|");
        SExp s = SList.make("foo", "|G_1.1.1|", new Integer(2));
        SExp.display(s);
        s = SExpOptimizer.modifyGuards(s, g.getHGuards());
        SExp.display(s);
        SExpOptimizer.modifyGuardsInPlace(s, g.getHGuards());
	SExp.display(s);
    }

    /**
     * Compute all dependencies for an sexp in file fileName.  Pass
     * in s if it happends to have already been read in.  Otherwise,
     * pass in null, and the method will read it.
     */
    public void computeDependencies(String fileName, SExp s) {
	try {
	    String shortFileName = shortName(fileName);
	    if (Debug.debug) Log.log("guard", "Computing deps for " + shortFileName);
	    if (s == null) {
		DataInputStream in = Utility.getInputStream(fileName + "x");
		s = SExpOptimizer.readFromFile(in);
		in.close();
	    }
	    asserts.put(fileName, new Vector());
	    assumes.put(fileName, new Vector());
	    computeDependencies(fileName, s, true);
	} catch (Exception e) {
	    Assert.fail(e);
	}
    }

    /**
     * Add dep to appropriate table, depending on polarity.
     */
    private void addToPolList(String fileName, String guard, boolean posPol) {
	if (Debug.debug) Log.log("guard", guard + " found in " + fileName);
	Vector v = (Vector)deps.get(guard);
	Assert.notFalse(v != null);
 	if (!v.contains(fileName)) {
	    v.addElement(fileName);
	}
	v = (Vector)(posPol?asserts:assumes).get(fileName);
	if (!v.contains(guard)) {
	    v.addElement(guard);
	}
    }
    
    /**
     * find all guard deps in sexp s.  fileName is the name of the file where
     * s was read from, and posPol is true if the polarity of s is positive.
     */
    void computeDependencies(String fileName, SExp s, boolean posPol) throws SExpTypeError {
	if (s.isAtom()) {
	    String a = s.getAtom().toString();
	    if (hguards.get(a) != null) {
		addToPolList(fileName, a, posPol);
	    }
	} else if (s.isInteger()) {
	    // nothing
	} else {
	    SList list = s.getList();
	    if (list.isEmpty()) {
		return;
	    }
	    
	    SExp head = list.at(0);
	    if (head == SExpOptimizer.orSexp ||
		head == SExpOptimizer.andSexp) {
		int n = list.length();
		for (int i = 1; i < n; i++) {
		    computeDependencies(fileName, list.at(i), posPol);
		}
	    } else if (head == SExpOptimizer.notSexp) {
		Assert.notFalse(list.length() == 2, "length " + s);
		computeDependencies(fileName, list.at(1), !posPol);
	    } else if (head == SExpOptimizer.forAllSexp) {
		Assert.notFalse(list.length() == 3 || list.length() == 4, "length " + s);
		computeDependencies(fileName, list.at(list.length()-1), posPol);
	    } else if (head == SExpOptimizer.labelPosSexp ||
		       head == SExpOptimizer.labelNegSexp) {
		Assert.notFalse(list.length() == 3, "length " + s);
		computeDependencies(fileName, list.at(2), posPol);
	    } else {
		if (Debug.debug) Log.log("guard", "Unknown form: " + s);
	    }
	}
    }


    /**
     * Write guard structures to stream.
     */
    void writeToFile(ObjectOutputStream oos) {
	try {
	    oos.writeObject(dir);
	    oos.writeObject(guards);
	    oos.writeObject(hguards);
	    
	    oos.writeObject(assumes);
	    oos.writeObject(asserts);
	    oos.writeObject(deps);
	} catch (Exception e) {
	    Assert.fail(e);
	}
    }

    /**
     * Read guard structures from stream.
     */
    void readFromFile(ObjectInputStream ios) {
	try {
	    dir = (String)ios.readObject();
	    guards = (Hashtable)ios.readObject();
	    hguards = (Hashtable)ios.readObject();
	    
	    assumes = (Hashtable)ios.readObject();
	    asserts = (Hashtable)ios.readObject();
	    deps = (Hashtable)ios.readObject();
	    refutedList = new Vector();
	} catch (Exception e) {
	    Assert.fail(e);
	}
    }
}
