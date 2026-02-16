/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.reader;

import javafe.ast.CompilationUnit;

import javafe.genericfile.*;
import java.util.ArrayList;
import java.io.FilenameFilter;
import java.io.File;

/**
 * A TypeReader is an extended {@link Reader} that understands how to
 * read in Java reference types given either a fully-qualified name or
 * a source file (in the form of a {@link GenericFile}).  A TypeReader
 * can also determine cheaply if a Java reference type exists or if a
 * Java package is accessible.
 *
 * <p> TypeReaders encapsulate how to map from fully-qualified names
 * to the data for the Java reference types.
 *
 * <p> TypeReaders are responsible for ensuring that all reads from a
 * given source yield the same {@link CompilationUnit}, regardless of
 * whether the duplicate reads occur through <code>read()</code> or
 * <code>readType()</code>.
 */

abstract public class TypeReader extends Reader
{
    /***************************************************
     *                                                 *
     * Existance/Accessibility:			       *
     *                                                 *
     **************************************************/

    /**
     * Return true iff the package P is "accessible".<p>
     *
     * Warning: the definition of accessible is host system dependent
     * and may in fact be defined as always true.<p>
     */
    //@ requires \nonnullelements(P);
    abstract public boolean accessable(/*@ non_null @*/ String[] P);


    /**
     * Return true iff the fully-qualified outside type P.T exists.
     */
    //@ requires \nonnullelements(P);
    abstract public boolean exists(/*@ non_null @*/ String[] P, 
				   /*@ non_null @*/ String T);


    /**
     *  Returns a (source or binary) file for the given type.
     */
    abstract public GenericFile findType(/*@ non_null @*/ String[] P, 
					 /*@ non_null @*/ String T);

    /***************************************************
     *                                                 *
     * Reading:					       *
     *                                                 *
     **************************************************/

    /**
     * Attempt to read and parse a CompilationUnit from *source file*
     * target.  Any errors encountered are reported via
     * javafe.util.ErrorSet.  Null is returned iff an error was
     * encountered.<p>
     *
     *
     * By default, we attempt to read only a spec (e.g., specOnly is set
     * in the resulting CompilationUnit) to save time.  If avoidSpec is
     * true, we return a non-spec, except in the case where we have
     * previously read in the same source file with avoidSpec false.
     * (See notes on caching below.)<p>
     *
     * There are 2 safe ways to ensure source files yield
     * non-spec files: (1) always use avoidSpec, or (2) read all
     * desired non-spec's at the beginning with avoidSpec set.
     * [these instructions apply to both versions of read.]<p>
     *
     *
     * The result of this function is cached.  Note that read(String[],
     * ...) may implicitly call this function, resulting in caching of
     * source files. <p>
     *
     * Only the value of avoidSpec used the first time a given file is
     * read is used (including implicit calls).  This may result in a
     * spec being returned unnecessarily when avoidSpec is true.<p>
     *
     * Target must be non-null.<p>
     */
    abstract public CompilationUnit read(/*@ non_null @*/ GenericFile target,
					 boolean avoidSpec);


    /**
     * Attempt to read and parse a CompilationUnit for the
     * fully-qualified outside type P.T.  Any errors encountered are
     * reported via javafe.util.ErrorSet.  Null is returned if no data
     * for P.T exists or if an error occurs. <p>
     *
     *
     * By default, we attempt to read only a spec (e.g., specOnly is set
     * in the resulting CompilationUnit) to save time.  If avoidSpec is
     * true, a spec is only returned if no source is available or if the
     * source file containing the type was read in earlier with
     * avoidSpec false.<p>
     *
     * Thus, there are 2 safe ways to ensure source files yield
     * non-spec files: (1) always use avoidSpec, or (2) read all
     * desired non-spec's at the beginning with avoidSpec set.<p>
     * [these instructions apply to both versions of read.]<p>
     *
     *
     * This routine is partially cached, in that it uses
     * read(GenericFile,...) (implicitly) to read source files.<p>
     *
     * If the resulting CompilationUnit is non-null, then it is always
     * complete, having no stubs.<p>
     *
     * This routine is responsible for such issues as out-of-date
     * binaries.<p>
     */
    //@ requires \nonnullelements(P);
    abstract public CompilationUnit read(/*@ non_null @*/ String[] P, 
					 /*@ non_null @*/ String T,
                                         boolean avoidSpec);

    /** Returns an enumeration of the GenericFile objects in the given 
	package P.
    */
    abstract public ArrayList findFiles(/*@ non_null @*/ String[] P);

    public /*@non_null*/ FilenameFilter filter() { 
	return new FilenameFilter() {
		public boolean accept(/*@non_null*/ File f, /*@non_null*/ String s) { return true; }
	    }; 
    }
}
