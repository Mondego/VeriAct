class Mutex {
    //@ ghost /*@ unwritable_by_env_if m == \tid */ public static Thread m

    //@ non_null
    private static final Object lock;

    //@ unwritable_by_env_if lock.holder == \tid
    private static boolean b;

    /*@ global_invariant b || m == null */


    /*@ 
        performs 
	action "acquire" (m) {
	  \old(m) == null && m == \tid
	}
    */
    public static void acquire() { 
	while (true) {
	    synchronized (lock) {
		if (!b) {
		    {{
			b = true;
			//@ set m = \tid
			//@ set \witness = "acquire"
		    }}
		    return;
		}
	    }
	}
    }

    /*@ 
        requires m == \tid;
        performs 
	action "release" (m) {
	  m == null
	}
    */
    public static void release() { 
	synchronized (lock) {
	    {{
		b = false;
		//@ set m = null
		//@ set \witness = "release"
	    }}
	}
    }
}

class Queue {
    //@ non_null
    //@ elems_unwritable_by_env_if[i] Mutex.m == \tid 
    private static final int[] data = new int[10];

    //@ unwritable_by_env_if Mutex.m == \tid
    private static int head;

    //@ unwritable_by_env_if Mutex.m == \tid
    private static int tail;

    //@ ghost public static int -> int elements

    //@ ghost public static int front 
    
    //@ ghost public static int back

    /*@
      global_invariant 
      Mutex.m == null ==> 0 <= head && head < data.length;

      global_invariant 
      Mutex.m == null ==> 0 <= tail && tail < data.length;

      global_invariant 
      (Mutex.m == null && head <= tail) ==> (tail - head == back - front);

      global_invariant 
      Mutex.m == null ==> 
      (\forall int i; head <= tail && head <= i && i < tail ==> data[i] == elements[front+i-head]);
      
      global_invariant 
      (Mutex.m == null && head > tail) ==> (data.length - head + tail == back - front);

      global_invariant 
      (Mutex.m == null && head > tail)  ==> 
      (\forall int i; head <= i && i < data.length ==> data[i] == elements[front+i-head]);

      global_invariant 
      (Mutex.m == null && head > tail) ==> 
      (\forall int i; 0 <= i && i < tail ==> data[i] == elements[front+i+data.length-head])
    */

    /* 
        modifies data[*], tail;
        performs 

	action "act1" (elements[\old(back)], back) {
	  (elements[\old(back)] == x) &&
          (back == \old(back) + 1)

        action "act1" { elements[back] = x; back++; }

	action "act1" (elements[*], back) {
	  (\forall int i;
              (i == \old(back) && elements[i] == x) ||
              (i != \old(back) && elements[i] == \old(elements)[i])) && 
          (back == \old(back) + 1)
        }
    */
    /*@ 
        modifies data[*], tail;
        performs 
	action "act1" (elements[back], back) {
	  (elements[\old(back)] == x) &&
          (back == \old(back) + 1)
        }
    */
    public static void put (int x) {
	// loop_invariant Mutex.m != \tid
	while (true) {
	    Mutex.acquire();
	    if ( (head == tail + 1) || 
		 (head == 0 && tail == data.length - 1) ) {
		// the queue is full
		Mutex.release();
	    } else {
		// the queue has space
		{{
		    //@ set \witness = "act1"
		    //@ set elements[back] = x
		    //@ set back = back + 1
		}}
		data[tail] = x;
		int temp = tail + 1;
		if (temp == data.length) {
		    tail = 0;
		} else {
		    tail = temp;
		}
		Mutex.release();
		return;
	    }	    
	}
    }

    /*@
        modifies head;
        performs 
	action "act1" (front) {
	  \old(front) < \old(back) &&
	  front == \old(front) + 1 &&
	  \result == elements[\old(front)]
        }
    */
    public static int get () {
	while (true) {
	    Mutex.acquire();
	    if (head == tail) {
		// the queue has no data 
		Mutex.release();
	    } else {
		// the queue has data 
		int result = data[head];
		int temp = head + 1;
		if (temp == data.length) {
		    head = 0;
		} else {
		    head = temp;
		}
		{{
		    //@ set \witness = "act1"
		    //@ set front = front + 1
		}}
		Mutex.release();
		return result;
	    }
	}
    }

}


class ProducerConsumer {
    //@ non_null
    public final static Producer producer = new Producer();
    //@ non_null
    public final static Consumer consumer = new Consumer();

    /*@
      env_assumption (\tid == producer && \old(producer.running)) ==> 
                       (producer.running && \old(Queue.back) == Queue.back);
      env_assumption (\tid == consumer && \old(consumer.running)) ==> 
                       (consumer.running && \old(Queue.front) == Queue.front);

      env_assumption (\tid == \main && !\old(producer.running)) ==>
                       (!producer.running && \old(Queue.back) == Queue.back);
      env_assumption (\tid == \main && !\old(consumer.running)) ==>
                       (!consumer.running && \old(Queue.front) == Queue.front);

      global_invariant (\forall int x; 0 <= x && x < Queue.back ==> 
                                           Queue.elements[x] == x);
    */

    //@ requires \tid == \main
    //@ requires Queue.back == 0 && Queue.front == 0
    //@ requires !producer.running && !consumer.running
    public static void main(String[] args) {
	producer.start();
	consumer.start();

	//Producer producer2 = new Producer();
	//Consumer consumer2 = new Consumer();
	//consumer2.start();
    }

}

class Producer extends Thread {

    //@ helper
    final public void run() {
	int i = 0;

	//@ loop_invariant i == Queue.back
	while (true) {
	    Queue.put(i);
	    i++;
	}
    }

    //  Esc/Java does not allow inlining of a method override.
    //  This is a hack to get around this limitation.
    //@ helper
    final public void start() {
	run();
    }
}

class Consumer extends Thread {

    //@ helper
    final public void run() {
	int i = 0;

	//@ loop_invariant i == Queue.front
	while (true) {
	    int j = Queue.get();
	    //@ assert i == j
	    i++;
	}
    }

    //  Esc/Java does not allow inlining of a method override.
    //  This is a hack to get around this limitation.
    //@ helper
    final public void start() {
	run();
    }

}
