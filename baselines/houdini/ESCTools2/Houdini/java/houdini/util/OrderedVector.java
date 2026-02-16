/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini.util;

import java.util.*;

/**
 * This class extends Vector with an ordering.
 * Clients should subclass it and provide a compare
 * operation.
 * By default, duplicate values can be stored, but
 * you can set it to treat the Vector more like
 * a set.  Only use addOrderedElement to insert
 * elements (can't override Vector methods to throw
 * an exception so be careful!).
 */
abstract public class OrderedVector extends Vector {

    /** override this.   
     ** <pre><blockquote>
     **                 o < p --> -1
     **                  o > p --> 1
     **                  o == p -> 0
     ** </blockquite></pre>
     **/
    abstract public int compare(Object o, Object p);
    
    private boolean duplicatesOK = true;
    
    /** 
     * Set whether or not we have a Set or not.
     * Pass false to make into a Set.  
     */
    public void setDuplicatesOK(boolean b) {
        this.duplicatesOK = b;
    } 
    
    public boolean getDuplicatesOK() {
        return this.duplicatesOK;
    }
    
    public OrderedVector(int initialCapacity, int capacityIncrement) {
	super(initialCapacity, capacityIncrement);
    }

    public OrderedVector(int initialCapacity) {
        this(initialCapacity, 0);
    }

    public OrderedVector() {
	this(10);
    }

    public synchronized void addOrderedElement(Object c) {
        int n = this.size();
        int i;
        for (i = 0; i < n; i++) { 
            Object e = this.elementAt(i);
            int x = compare(e,c);
            if (x == 0 && !this.duplicatesOK) return;
            if (x >= 0) {
                break;
            }
        }
        this.insertElementAt(c, i);
    }
}
