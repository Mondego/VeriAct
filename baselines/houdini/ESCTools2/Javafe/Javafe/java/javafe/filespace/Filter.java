/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


/**
 * A simple filter interface for use in filtering out values.
 */


public interface Filter {

    /**
     * The actual "static" type of objects we filter
     */
    //@ ghost public \TYPE acceptedType;


    /** Should our client accept a given value? */
    //@ requires value != null;
    //@ requires \typeof(value) <: acceptedType;
    boolean accept(Object value);
}
