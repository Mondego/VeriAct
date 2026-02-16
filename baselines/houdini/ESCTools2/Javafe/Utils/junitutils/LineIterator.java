/*
 * Copyright (C) 2000-2006 Iowa State University
 *
 * This file is part of mjc, the MultiJava Compiler.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Id$
 */
package junitutils;

import java.io.*;
import java.util.Iterator;

/** This is a utility class that is an iterator over the lines produced
    by reading a file.  It is used in particular by various
    JUnit tests.

    @author David R. Cok
*/
public class LineIterator implements Iterator {

    //@ public invariant elementType == \type(String);
    //@ public invariant !returnsNull;

    //@ public represents moreElements <- nextLine != null;

    /** A reader that reads lines from the file. */
    private /*@ non_null */ BufferedReader r;
    
    /** The next value to be returned by the iterator.  We read ahead one so
	that we know the value of hasNext() when asked.
    */
    /*@ spec_public */ private /*@ nullable */ String nextLine; //@ in objectState;

    /** Starts an iterator reading from the given external process. 
	@param filename The name of the file to be read
     */
    /*@ public behavior
      @   modifies nextLine, elementType, returnsNull;
      @   ensures (* file is readable *);
      @   signals (java.io.IOException) (* file is not readable *);
      @ also private behavior
      @   modifies r;
      @*/
    public LineIterator(/*@ non_null */ String filename) throws java.io.IOException  {
	r = new BufferedReader(new FileReader(filename)); 
	nextLine = r.readLine();
	//System.out.println("READ " + nextLine);
	//@ set elementType = \type(String);
	//@ set returnsNull = false;
    }

    /** Per a standard iterator, returns true if there is another value waiting. */
    public boolean hasNext() throws Error {
	if (nextLine == null) {
	    try {
		r.close();
	    } catch (java.io.IOException e) {
		throw new Error("EXCEPTION in hasNext - " + e); 
	    }
	}
	return nextLine != null;
    }

    /** Per a standard iterator, returns the next value - and throws 
	java.util.NoSuchElementException if the list has been exhausted 
	(hasNext() returns false).
    */
    public Object next() throws java.util.NoSuchElementException, Error {
	//@ set remove_called_since = false;
	if (nextLine == null) 
	    throw new java.util.NoSuchElementException();

	try {
	    String n = nextLine;
	    nextLine = r.readLine();
	    //System.out.println("READ " + nextLine);
	    return n;
	} catch (java.io.IOException e) {
	    throw new Error("EXCEPTION in next - " + e); 
	}
    } //@ nowarn Exception, Post;

    /** This operation will throw an exception, as there is no need for 
	remove in this context. */
    //@ also
    //@ requires \typeof(this)==\type(LineIterator);
    //@ ensures false;
    //@ signals (UnsupportedOperationException) true;
    public void remove() throws UnsupportedOperationException {
	throw new java.lang.UnsupportedOperationException();
    }

}



