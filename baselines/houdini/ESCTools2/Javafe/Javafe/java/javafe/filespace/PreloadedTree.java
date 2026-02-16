/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;


/**
 * A PreloadedTree is a HashTree whose edges map is loaded exactly once
 * before any children-fetching queries complete; the loading is lazy,
 * however, and occurs when the first children-fetching method is
 * called.<p>
 *
 * If a subclass adds additional methods that refer to edges, it should
 * make sure ensureEdgesLoaded is called before edges is used.<p>
 */

abstract class PreloadedTree extends HashTree {

    /***************************************************
     *                                                 *
     * Loading the edges map:			       *
     *                                                 *
     **************************************************/

    /** Have we loaded the edges map yet? */
    private boolean loaded = false;

    /** Ensure that the edges map is ready for use */
    public final void ensureEdgesLoaded() {
	if (!loaded) {
	    loadEdges();
	    loaded = true;
        }
    }

    /** Load the edges map for use.  */
    protected abstract void loadEdges();


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** Create a root node: */
    public PreloadedTree(Object data) {
	super(data);
    }

    /** Create a non-root node: */
    //@ requires parent != null && label != null;
    protected PreloadedTree(Tree parent, String label, Object data) {
	super(parent, label, data);
    }


    /***************************************************
     *                                                 *
     * Fetching and counting children:		       *
     *                                                 *
     **************************************************/

    public final Enumeration children() {
	/* Ensure edges is loaded before we use it: */
	ensureEdgesLoaded();

	return super.children();
    }

    public final Tree getChild(String label) {
	/* Ensure edges is loaded before we use it: */
	ensureEdgesLoaded();

	return super.getChild(label);
    }

    /*
     * isLeaf and getChildrenCount should be considered as made final
     * here as well.
     */
}
