/* Copyright Hewlett-Packard, 2002 */

package mercator.util;

public final class CalvinRWTest {

    public static void main(/*@ non_null */ String[] args) {
        // parse command-line
        int numThreads = 2;
        if (args.length > 0) {
            try {
                numThreads = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Illegal number: " + args[0]);
                e.printStackTrace();
                System.exit(1);
            }
        }

        // create workers
        ReadersWriterLock rwl = new ReadersWriterLock();
        ThreadsState ts = new ThreadsState(rwl);
        for (int i = 0; i < numThreads; i++) {
	    System.out.println("Starting Thread " + Integer.toString(i+1));
            MyWorker w = new MyWorker(i+1, ts);
            w.start();
        }
    }


    private static final class ThreadsState {
	/*@ writable_if 
	    lock.holder == \tid || lock.holder == 0 */ 
	/* non_null */ private ReadersWriterLock lock;

	/*@ writable_if lock.currentWriter == \tid */ 
	/*@ readable_if lock.readerSet[\tid] */
	private int data;
	/*@ global_invariant data >= 0 */

	ThreadsState(/*@ non_null */ ReadersWriterLock rwl) {
	    this.data = 0;
	    this.lock = rwl;
	}
    }

    private static final class MyWorker extends Thread {

	private int tid;

	/*@ spec_public */ 
	/* non_null */ private ThreadsState ts;
	/*@ oti ts == \old(ts) */  
	/*@ oti ts.lock == \old(ts.lock) */  
	/*@ global_invariant ts != null */  
	/*@ global_invariant ts.lock != null */  

	private int myData;
	/*@ global_invariant myData >= 0 */

	MyWorker(int id, /*@ non_null */ ThreadsState st) {
	    this.tid = id;
	    this.ts = st;
	    this.myData = 0;
	}

	public void run() {
	//@ assume !(ts.lock.readerSet[\tid])
	    while(true) {
		// read ts.data into myData
	    System.out.println("Thread " + Integer.toString(tid) + " attempting read");
		synchronized(ts.lock) { 
		    try {
			ts.lock.beginRead();
		    }
		    catch(StopException e) {
			Assert.P(false);
			//@ assume false
			//@ unreachable
		    }
		}
		myData = ts.data;

		//@ assert ts.lock.readerSet[\tid]
		// assert false
		synchronized(ts.lock) {
		    ts.lock.endRead();
		}

		// increment myData
		myData++;
		
		// assert false
		// write myData back to ts.data
		System.out.println("Thread " + Integer.toString(tid) + " attempting write");

	        // assert false
		// assert ts.lock.readerSet[\tid]
	        // assert ts.lock.currentWriter == 0 
		/*
		synchronized(ts.lock) {
		// assert ts.lock.readerSet[\tid]
		    ts.lock.beginWrite();
		// assert ts.lock.readerSet[\tid]
		// assert false
		}
		*/
		// assert false
		// assert ts.lock.currentWriter == \tid
		// assert ts.lock.currentWriter != 0
		ts.data = myData;
		synchronized(ts.lock) {
		    ts.lock.endWrite();
		}
	    }
	}
    }

}
