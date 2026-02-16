/* Copyright Hewlett-Packard, 2002 */

package mercator.util;

/** <code>Assert.P</code> performs a run-time check of a 
    design-time assumption. */

public class Assert {
    /** Throw <code>java.lang.Error</code> iff <code>b</code> is false. */
    static final public void P(boolean b) {
        if (!b) throw new Error("*** Assertion failure ***");
    }
}
