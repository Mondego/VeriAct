/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.pa.generic;

import javafe.util.*;

/* Represents a clause (disjunction of literals) over up to 64 variables.
   Clause represented by low bits in "stars" and "bits".
   "stars" has a 1 in bit n if variable n is not mentioned in clause.
   If "stars" has a 0 in bit n then variable n is mentioned in clause, and
   its sign is given by bit n of "bits".   
 */   
public class Disjunction {
    public long stars;
    public long bits;

    public  Disjunction(long stars, long bits) {
	Assert.notFalse( (stars & bits) == 0 );
	this.stars = stars;
	this.bits = bits;
    }
    public Disjunction(/*@ non_null @*/ Disjunction d) {
	this( d.stars, d.bits);
    }
    // Yields Disjunction for false
    public Disjunction() {
	this( -1L, 0L);
    }
    public /*@ non_null @*/ String toString() {
	return "("+Long.toBinaryString(stars)+","+Long.toBinaryString(bits)+")";
    }
    public boolean equals(Object o) {
	if( o instanceof Disjunction ) {
	    Disjunction d = (Disjunction)o;
	    return d.stars == stars && d.bits == bits;
	} else {
	    return false;
	}
    }
}
