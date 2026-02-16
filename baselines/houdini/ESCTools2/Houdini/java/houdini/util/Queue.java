/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.util;
    
import java.util.Stack;
import java.util.Vector;

/**
 * A class that implements a queue (either bounded or unbounded).
 */
public class Queue extends Stack {
    static private final int UNBOUNDED = -1;
    protected int bound = UNBOUNDED;
    
    /**
     * Create queue with size bound.
     */
    public Queue(int bound) {
        super();
        this.bound = bound;
    }
    
    /**
     * Create an unbounded queue.
     */
    public Queue() {
        this(10); 
        this.bound = UNBOUNDED;
    }
    
    /**
     * Same as enqueue.
     */
    public synchronized Object push(Object item) {
	return enqueue(item);  
    }    
    
    /**
     * Add an object to the queue.
     * @returns the booted element if the queue was already full
     * (it boots the oldest elem).
     */
    public synchronized Object enqueue(Object o) {
        Object booted = null;      
        if (bound != UNBOUNDED && size() == bound) {
            booted = elementAt(0);
            removeElementAt(0);          
        }
        addElement(o);
        this.notify();
        return booted;
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
    
    public synchronized Object[] toArray(Object o[]) {
        for (int i = 0; i < o.length; i++) {
            o[i] = elementAt(o.length - (i + 1));
        }
        return o;
    }
    
    public synchronized Object clone() {
	Queue v = (Queue)super.clone();
        v.bound = this.bound;
	return v;
    }    
    
}
