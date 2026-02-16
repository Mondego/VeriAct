class Container {
    /*@ unwritable_by_env_if holder == \tid */
    public int counter; 
    
    /*@ env_assumption counter >= \old(counter) */
}

class Job extends Thread {
    /*@ unwritable_by_env_if (objref != null || \tid == \main) */
    Container objref; 

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

    //@ helper
    final public void run() {
	//@ loop_invariant objref != null
        for (;;) {
	    incr();
        }
    }    

    //  Esc/Java does not allow inlining of a method override.
    //  This is a hack to get around this limitation.
    //@ helper
    final public void start() {
	run();
    }

}

class Apprentice {
    /* assume that \tid is the main thread */
    //@ requires \tid == \main
    public static void main(String[] args) {
        Container container = new Container();
	//Container bogus = new Container();
	
	//@ loop_invariant \tid == \main
	for (;;) {
	    Job job = new Job();

	    job.setref(container);

	    /*
	    synchronized(job.objref) {
		job.objref.counter++;
	    }
	    */

	    job.start();
	    //	    job.setref(bogus);
	}	
    }

}
