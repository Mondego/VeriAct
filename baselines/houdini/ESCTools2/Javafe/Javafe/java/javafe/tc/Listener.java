/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;


/**
 * The interface for listeners of <code>CompilationUnit</code>-loading
 * notification events (sent by <code>OutsideEnv</code>). <p>
 *
 * @see CompilationUnit
 * @see OutsideEnv
 */

public interface Listener {
    /**
     * Each time a <code>CompilationUnit</code> is loaded by
     * <code>OutsideEnv</code>, this routine in the current
     * <code>Listener</code> (see <code>OutsideEnv.setListener</code>)
     * is called with the newly-loaded
     * <code>CompilationUnit</code>. <p>
     *
     * The passed <code>CompilationUnit</code> will already have
     * the <code>sig</code> fields of its direct <code>TypeDecl</code>s
     * adjusted.  (See the class comments for
     * <code>OutsideEnv</code>).<p>
     */
    public void notify(/*@non_null*/CompilationUnit justLoaded);
}
