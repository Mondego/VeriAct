/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.util;
    
import java.util.Stack;
import java.util.Vector;

/**
 * A class that implements a queue (either bounded or unbounded).
 */
abstract public class PriorityQueue extends OrderedVector {

    /**
     * Create an unbounded queue.
     */
    public PriorityQueue() {
        super(10); 
    }
    
    /*
     * Add an object to the queue.
     * @returns the booted element if the queue was already full
     * (it boots the oldest elem).
     */
    public synchronized void enqueue(Object o) {
        Object booted = null;      
	addOrderedElement(o);
        this.notify();
    }
    
    /**
     * Removes an element from the queue and returns it.
     * @return elem from queue or <code>null</code> if empty.
     */
    public synchronized Object dequeue() {            
        return this.dequeue(false);
    }
            
    /**
     * Removes an element from the queue and returns it.
     * @param block if told to block, the call blocks on an empty queue
     * until something is added.
     * @return elem from queue or <code>null</code> if empty and not blocking.
     * 
     */
    public synchronized Object dequeue(boolean block) {
        while (size() == 0) {
            if (!block) return null;
            try {
                this.wait();
            } catch (InterruptedException e) {
                return null;
            }
        }
        Object booted;
        booted = elementAt(0);       
        removeElementAt(0);   
        return booted;
    }
    
}
