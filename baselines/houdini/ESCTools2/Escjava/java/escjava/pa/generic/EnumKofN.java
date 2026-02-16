/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.pa.generic;

import javafe.util.*;

/* Enumerates all clauses of size k, where there are n variables.
 */

class EnumKofN extends Disjunction {

    int n,k;
    
    public  EnumKofN(int k, int n) {
	super(-1L, 0);
	Assert.notFalse( k <= n );
	this.k = k;
	this.n = n;
    }

    boolean getNext() {

	if( stars == -1 ) {
	    // intitialize
	    stars = 0;
	    for(int i=k; i<n; i++) {
		stars |= 1<<i;
	    }
	    return true;
	}

	// get next

	int nb=0, ns=0, i=0;

	for(i=0; i<n; i++) {
	    Assert.notFalse( nb + ns == i );

	    long b = 1<<i;
	    if( (stars & b) != 0 ) {
		// was a star
		ns++;
		if (nb>0) {
		    // remove star
		    stars &= ~b;
		    // clear bit
		    bits &= ~b;
		    nb--;
		    break; // 0.enum(nb-1,s)
		} else {
		    continue; // ret, done 0, nb=0
		}
	    } else {
		// was a bit
		if( (bits & b) == 0 ) {
		    // was a 0, set to 1
		    bits |= b;
		    break; // 1.enum(nb-1,s)
		} else {
		    nb++;
		    // remove bit
		    bits &= ~b;
		    // put star in
		    stars |= b; 
		    continue; // ret, done bits
		}
	    }
	}
	if( i==n ) return false;

	// fill from i-1 to 0
	while( ns > 0 ) {
	    ns --;
	    i--;
	    long b = 1<<i;
	    Assert.notFalse( (stars & b) != 0 );
	}
	Assert.notFalse( i == nb );

	while( i > 0 ) {
	    i--;
	    long b = 1<<i;
	    stars &= ~b;
	    Assert.notFalse( (stars & b) == 0 );
	    Assert.notFalse( (bits & b) == 0 );
	}

	return true;
    }

    public static void main(String argv[]) {
	
	EnumKofN e = new EnumKofN(2,3);

	while( e.getNext() ) {
	    System.out.println("Bits = "+Long.toString(e.bits,2)+
			       " stars = "+Long.toString(e.stars,2));
	}
    }

}
	
    
