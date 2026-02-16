/* Copyright Hewlett-Packard, 2002 */

package mercator.util;

/** A <code>ReadersWriterLock</code> is a class that can be
    used to synchronize multiple readers and a single writer.
    There are <code>BeginRead</code> and <code>EndRead</code>
    methods that should be called before and after a read-only
    critical section, and <code>BeginWrite</code> and <code>EndWrite</code>
    methods that should be called before and after a critical
    section in which writes are performed. The <code>ReadersWriterLock</code>
    guarantees that no writers will be in the critical section
    while at least one reader is, and vice-versa. It also guarantees
    that at most one writer will be in the critical section at
    a time.<p>
    
    This implementation gives priority to writers. That is, if
    both writers and readers are blocked waiting to enter the
    critical section, then the writer will be given preference.<p>
    
    A <code>ReadersWriterLock</code> also provides a method for
    aborting readers via the <code>abortRead</code> method.<p>
    
    IMPORTANT NOTE: The methods of a <code>ReadersWriterLock</code>
    are unsynchronized. It is the caller's responsibility to ensure
    that the mutex associated with a <code>ReadersWriterLock</code>
    object <code>rwl</code> is held when any of <code>rwl</code>'s
    methods are called. This requirement is denoted in this module
    by "REQUIRES LL = SELF". */

public class ReadersWriterLock {
    
    /** Acquire a share of the object's read lock. This call will
        block until no writer holds the lock and there are no writers
        waiting to acquire the lock. 

        @exception StopException if abortRead has been called
     */
    //@ requires this.holder == \tid
    //@ requires !readerSet[\tid]
    //@ modifies numReaders, readerSet[\tid]
    /*@ performs (readerSet[\tid])
      { (!\old(readerSet[\tid]) && readerSet[\tid]) } */
    public void beginRead() throws StopException {
        while (!this.abortMode &&
               (this.hasWriter || this.waitingWriters > 0)) {
            try {
                this.wait();
            } catch (InterruptedException e) {
                Assert.P(false);
            }
        }
        if (this.abortMode) throw new StopException();
	this.numReaders++;
	//@ set readerSet[\tid] = true;
    }
    
    //@ requires this.holder == \tid
    // modifies this.abortMode
    /* ensures this.abortMode == abortMode */
    // NEED TO HANDLE readerSet here?
    public void abortRead(boolean abortMode) {
        this.abortMode = abortMode;
        if (abortMode) this.notifyAll();
    }
    
    /** Release the calling thread's portion of this object's
        read lock. */
    //@ requires this.holder == \tid
    //@ requires readerSet[\tid]
    //@ modifies numReaders, readerSet
    /*@ ensures !readerSet[\tid] */
    public void endRead() {
        this.numReaders--;
	//@ set readerSet[\tid] = false;
        if (this.numReaders == 0) {
            this.notifyAll();
        }
    }
    
    /** Acquire this object's write lock. This call will block
        until all readers release their shares of this object's
        read lock. */
    //@ requires this.holder == \tid
    //@ modifies hasWriter, currentWriter
    /*@ performs (currentWriter) 
      { (\old(currentWriter == 0 && !readerSet[\tid])) 
        && (currentWriter == \tid) } */
    /* performs (currentWriter) 
      { (\old(currentWriter) == 0) 
        && (currentWriter == \tid) } */
    public void beginWrite() {
        while (this.numReaders > 0 || this.hasWriter) {
            this.waitingWriters++;
            try {
                this.wait();
            } catch (InterruptedException e) {
                Assert.P(false);
            }
            this.waitingWriters--;
        }
        this.hasWriter = true;
	//@ set currentWriter = \tid;
    }
    
    /** Release this object's write lock. */
    //@ requires this.holder == \tid
    //@ requires currentWriter == \tid
    //@ modifies hasWriter, currentWriter
    /*@ ensures currentWriter == 0 */
    public void endWrite() {
        this.hasWriter = false;
	//@ set currentWriter = 0;
        this.notifyAll();
    }
    
    /* The following fields are protected by SELF. */
    /*@ spec_public */
    /*@ writable_if this.holder == \tid */ 
    private int numReaders = 0;        // # of threads that hold a share of the read lock

    /*@ spec_public */
    /*@ writable_if this.holder == \tid */ 
    private boolean hasWriter = false; // a thread holds the write lock

    /*@ spec_public */
    /*@ writable_if this.holder == \tid */ 
    private int waitingWriters = 0;    // # of threads waiting to acquire write lock

    /*@ spec_public */
    /*@ writable_if this.holder == \tid */ 
    private boolean abortMode = false; // mode for aborting readers


    //@ ghost public int currentWriter;
    /*@ oti this.holder == \tid ==> currentWriter == \old(currentWriter) */

    /*@ oti (\old(hasWriter) && \old(currentWriter) == \tid) 
      ==> (currentWriter == \tid && hasWriter) */ 


    //@ ghost public int -> boolean readerSet;
    /*@ oti \old(readerSet[\tid]) == readerSet[\tid] */

    /*@ global_invariant (\forall int i; (currentWriter != 0)
      ==> !readerSet[i]) */

    // This GI needed to prove the performs pred for beginWrite
    /*@ global_invariant !this.hasWriter ==> (currentWriter == 0) */


    /** OTIs/GIs that might not be needed */
    /* global_invariant numReaders >= 0 */
    /* global_invariant hasWriter ==> numReaders == 0 */
    /* oti (\old(currentWriter) == \tid) 
       == (currentWriter == \tid) */ 
    /* oti (\forall int i; 
      (i != \tid && this.holder == \tid) 
      ==> readerSet[i] == \old(readerSet)[i]) */

}

