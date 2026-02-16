/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.pa.generic;

import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;

import javafe.util.Set;
import javafe.util.Location;
import javafe.util.Assert;
import javafe.util.StackVector;

import mocha.wrappers.jbdd.*;

// General interface for a predicate abstraction implementation

public interface Abstractor {

    // Returns true if fixpoint
    public abstract boolean union(/*@ non_null @*/ Prover p);

    public abstract /*@ non_null @*/ jbdd get();
    public abstract /*@ non_null @*/ Vector getClauses();

}
