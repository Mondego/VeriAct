/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


/**
 * The normal result type for Resolve.lookup:
 */

public class Resolve_Result {

    public Tree     myPackage	= null;
    public String   myTypeName	= null;

    //@ invariant \nonnullelements(remainder);
    public String[] remainder	= new String[0];
}
