/* Copyright Hewlett-Packard, 2002 */

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
    public abstract boolean union(Prover p);

    public abstract jbdd get();
    public abstract Vector getClauses();

}
