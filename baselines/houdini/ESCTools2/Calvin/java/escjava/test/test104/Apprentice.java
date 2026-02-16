/* Copyright Hewlett-Packard, 2002 */

class Container {
    /*@ guarded_by holder == \tid */
    public int counter; 
    
    /*@ oti counter >= \old(counter) */
}

class Job extends Thread {
    Container objref; 

    /*@ oti (\old(objref) == null && \tid != \main) || (\old(objref) == objref) */

    /* 
       Create a dummy helper and inline it to ensure that objref is 
       initialized to null.
    */
    //@ helper
    Job() {
    }

    //@ helper
    public final void incr () {
	synchronized(objref) {
	    objref.counter = objref.counter + 1;
	}
    }

    //@ helper
    public final void setref(Container o) {
        objref = o;
    }

    public void run() {
	/* 
	   Assume that \tid is not the main thread and that objref is 
	   nonnull at the beginning.  The second condition is asserted
	   when this thread is forked in the main method of Apprentice.
	*/
	//@ assume \tid != \main
	//@ assume objref != null

	//@ loop_invariant objref != null
        for (;;) {
	    incr();
        }
    }
}

class Apprentice {
    public static void main(String[] args) {
	/* assume that \tid is the main thread */
	//@ assume \tid == \main

        Container container = new Container();
	Container bogus = new Container();

	for (;;) {
	    Job job = new Job();

	    job.setref(container);
	    
	    //@ assert job.objref != null

	    job.start();

	    job.setref(bogus);
	}	
    }

}
