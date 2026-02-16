/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.pa.generic;

import mocha.wrappers.jbdd.*;

/* General prover abstraction.
 */
public interface Prover {

    public static final int VALID = 0;
    public static final int INVALID = 1;
    public static final int UNKNOWN = 2;
    
    // Universally conjunctive and universally disjunctive
    public boolean check(jbdd b);
    public int quickCheck(jbdd b);

    public String printClause(jbdd b);

    public String report(); // Reports num queries, etc
}
