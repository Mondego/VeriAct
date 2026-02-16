/*
 * Copyright (C) 2000-2001 Iowa State University
 *
 * This file is part of mjc, the MultiJava Compiler.
 *
 * $Id$
 */
package junitutils;

/**
 * Some utility methods for test cases */
public class TestCase extends junit.framework.TestCase {

    //---------------------------------------------------------------------
    // CONSTRUCTORS
    //---------------------------------------------------------------------

    public TestCase( String name ) {
	super( name );
    }

    //---------------------------------------------------------------------
    // PROTECTED MEMBERS
    //---------------------------------------------------------------------

    /**
     * Compare Strings for equality with better difference reporting. 
     * @param expected	the expected string
     * @param actual	the actual string
     * @param detailed if true then report first position of
     *			difference along with Unicode values of the
     *			two characters, otherwise just compare strings
     *			for equality */
    protected void assertEquals( /*@ non_null */ String expected, 
				 /*@ non_null */ String actual, 
				 boolean detailed )
    {
	if( detailed ) {
	    assertDiff( expected, actual );
	} else {
	    assertEquals( expected, actual );
	}
    }

    protected void assertDiff(/*@ non_null */ String expected, /*@ non_null */ String actual ) {
	Diff diff = new Diff( "expected", expected, "actual", actual );
	if (diff.areDifferent()) {
	    fail( diff.result() );
	}
    }
    
    protected static final String NEWLINE = 
	System.getProperty( "line.separator" );
}
