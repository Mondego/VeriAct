/* Copyright Hewlett-Packard, 2002 */


package escjava.ast;

import java.util.Hashtable;
import java.util.Vector;
import java.util.Enumeration;
import javafe.util.Set;
import javafe.util.ErrorSet;

public class LabelRelation {
    Hashtable domain = new Hashtable();
    Vector mapToInts = new Vector();
    Hashtable actions = new Hashtable();

    public LabelRelation(PerformsStmt p) {
	Set frontier = new Set();
	addLabel("begin");
	addLabel("end");
	frontier.add("begin");
	frontier = processStmt(frontier, p);
	Enumeration e = frontier.elements();
	while (e.hasMoreElements()) {
	    String s = (String)e.nextElement();
	    addEdge(s, "end");
	}
    }


    public PerformsAction getAction(String ident) {
	return (PerformsAction)actions.get(ident);
    }

    public Enumeration domain() {
	return domain.keys();
    }

    public Enumeration range(String ident) {
	return ((Vector)domain.get(ident)).elements();
    }

    public int stringToInt(String ident) {
	return mapToInts.indexOf(ident);
    }

    public String intToString(int index) {
	return (String)mapToInts.elementAt(index);
    }

    void addLabel(String ident) {
	if (!domain.contains(ident)) {
	    domain.put(ident, new Vector());
	    mapToInts.addElement(ident);
	}
    }

    void addEdge(String src, String dst) {
	Vector t = (Vector)domain.get(src);
	if (t == null) {
	    addLabel(src);
	    t = (Vector)domain.get(src);
	}
	if (!t.contains(dst)) {
	    t.addElement(dst);
	}
    }

    boolean inDomain(String src) {
	return domain.get(src) != null;
    }

    public boolean inRelation(String src, String dst) {
	Vector t = (Vector)domain.get(src);
	if (t == null) {
	    return false;
	}
	return t.contains(dst);
    }

    private Set processStmt(Set frontier, PerformsStmt p) {
	switch (p.getTag()) {
	  case TagConstants.CHOICE: {
	      PerformsChoice samp = (PerformsChoice)p;
	      Set s = processStmt(frontier, samp.left);
	      s.union(processStmt(frontier, samp.right));
	      return s;
	  }
	  case TagConstants.SEMICOLON: {
	      PerformsSequence samp = (PerformsSequence)p;
	      int i;
	      for (i = 0; i < samp.seq.size(); i++) {
		  PerformsStmt s = samp.seq.elementAt(i);
		  frontier = processStmt(frontier, s);
	      }
	      return frontier;
	  }

	  case TagConstants.ACTION: {
	      PerformsAction samp = (PerformsAction)p;
	      Enumeration e = frontier.elements();
	      if (inDomain(samp.label)) {
		  ErrorSet.fatal(samp.getStartLoc(), "duplicate action label: " + samp.label);
	      }
	      while (e.hasMoreElements()) {
		  String s = (String)e.nextElement();
		  addEdge(s, samp.label);
	      }
	      Set s = new Set();
	      s.add(samp.label);
	      actions.put(samp.label, p);
	      return s;
	  }
	  default: {
	      return null;
	  }
	}
    }

    public String toString() {
	Enumeration e = domain.keys();
	String res = "";
	while (e.hasMoreElements()) {
	    int i;
	    String s = (String)e.nextElement();
	    res += s + ":";
	    Vector v = (Vector)domain.get(s);
	    for (i = 0; i < v.size(); i++) {
		res += " " + v.elementAt(i);
	    }
	    res += "\n";
	}	
	return res;
    }
    
}
