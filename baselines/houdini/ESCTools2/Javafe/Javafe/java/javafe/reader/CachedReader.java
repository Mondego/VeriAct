/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.reader;

import java.util.Hashtable;

import javafe.ast.CompilationUnit;
import javafe.genericfile.*;

/**
 * CachedReader takes a uncached Reader and produces a cached version
 * of it using a simple implementation of caching via a HashTable.
 *
 * <p> Reads from GenericFiles with null canonicalIDs are not cached.
 */

public class CachedReader extends Reader
{
    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * The underlying Reader whose results we are caching.
     */
    //@ invariant underlyingReader != null;
    protected /*@ non_null */ Reader underlyingReader;

    /**
     * Creating a cached version of a Reader:
     */
    //@ requires reader != null;
    public CachedReader(/*@ non_null */ Reader reader) {
	underlyingReader = reader;

	//+@ set cache.keyType = \type(String);
	//+@ set cache.elementType = \type(Object);
    }


    /***************************************************
     *                                                 *
     * The Cache itself:			       *
     *                                                 *
     **************************************************/

    /**
     * Our cache; maps the non-null canonicalID (if it has one) of a
     * GenericFile (see GenericFile.getCanonicalID) to either a
     * CompilationUnit or a CachedReader_Null.
     */
    //@ invariant cache != null;
    //+@ invariant cache.keyType == \type(String);
    //+@ invariant cache.elementType == \type(Object);
    protected /*@ non_null */ Hashtable cache = new Hashtable();


    /***************************************************
     *                                                 *
     * Caching methods:				       *
     *                                                 *
     **************************************************/

    /**
     * Lookup a non-null GenericFile in the cache.
     */
    //@ requires target != null;
    final protected /*@ nullable */ Object get(/*@ non_null */ GenericFile target) {
	String canonicalID = target.getCanonicalID();
	if (canonicalID==null)
	    return null;

	return cache.get(canonicalID);
    }


    /**
     * Store information about a non-null GenericFile in the cache;
     * this has no effect if the GenericFile has a null canonicalID.
     */
    //@ requires target != null;
    final protected void put(/*@ non_null */ GenericFile target, /*@ nullable */ CompilationUnit value) {
	String canonicalID = target.getCanonicalID();
	if (canonicalID==null)
	    return;

	if (value != null)
	    cache.put(canonicalID, value);
	else
	    cache.put(canonicalID, new CachedReader_Null());
    }


    /**
     * Is the result of read on target cached for this Reader?<p>
     *
     * Target must be non-null.<p> 
     */
    //@ requires target != null;
    public boolean isCached(/*@ non_null */ GenericFile target) {
	String canonicalID = target.getCanonicalID();
	if (canonicalID==null)
	    return false;

	return cache.containsKey(canonicalID);
    }

    /**
     * Flush the saved info (if any) for target for this Reader.
     *
     * Target must be non-null.<p>
     */
    //@ requires target != null;
    public void flushTarget(/*@ non_null */ GenericFile target) {
	String canonicalID = target.getCanonicalID();
	if (canonicalID==null)
	    return;

	cache.remove(canonicalID);
    }

    /**
     * Flush all cached information for this Reader.
     */
    public void flushAll() {
	cache = new Hashtable();

	//+@ set cache.keyType = \type(String);
	//+@ set cache.elementType = \type(Object);
    }


    /***************************************************
     *                                                 *
     * Reading:					       *
     *                                                 *
     **************************************************/

    /**
     * Attempt to read and parse a CompilationUnit from target.
     * Any errors encountered are reported via javafe.util.ErrorSet.
     * Null is returned iff an error was encountered.<p>
     *
     *
     * By default, we attempt to read only a spec (e.g., specOnly is set
     * in the resulting CompilationUnit) to save time.  If avoidSpec is
     * true, we attempt to return a non-spec, although this may not
     * always be possible.<p>
     *
     *
     * The results of this function (including null results, but not
     * the action of reporting error messages) are cached.
     * Only the value of avoidSpec used the first time a given file is
     * read is used.  This may result in a spec being returned
     * unnecessarily when avoidSpec is true.<p>
     *
     * Target must be non-null.<p>
     */
    public /*@ nullable */ CompilationUnit read(/*@non_null*/GenericFile target, boolean avoidSpec) {
	Object result = get(target);
	if (result != null) {
	    if (result instanceof CompilationUnit)
		return (CompilationUnit)result;
	    else if (result instanceof CachedReader_Null)
		return null;
	    else
		javafe.util.Assert.fail("impossible");
	}

	CompilationUnit newResult = underlyingReader.read(target, avoidSpec);
	put(target, newResult);
	return newResult;
    }
}


/**
 * Instances of this class are used privately by CachedReader to
 * represent null values in Hashtables.
 */
/*package*/ class CachedReader_Null {
}
