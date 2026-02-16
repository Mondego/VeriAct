package decsrc.container;

/** A queue of integers. */
final public class IntQueue {
    private /*@non_null*/int[] buf;			// Queue contents
    private int   first;		// Index of first entry
    private int   num;			// Number of entries in queue

    /** Create an empty queue of integers. */
    public IntQueue() {
	this(10);
    }

    /** Create an empty queue that is appropriately sized to
	hold "predict" elements.  If the queue ends up requiring
	more than "predict" elements, it will be resized
	automatically. */
    public IntQueue(int predict) {
	// Find power of 2 >= predict
	int x = 1;
	while (x < predict) {
	    x <<= 1;
	}

	buf = new int[x];
	first = 0;
	num = 0;
    }

    /** Removes and returns the oldest element in the queue.  Throws
	"IndexOutOfBoundsException" if the queue is empty. */
    public int dequeue() throws IndexOutOfBoundsException {
	if (num > 0) {
	    int result = buf[first];
	    first = (first + 1) & (buf.length - 1);
	    num--;
	    return result;
	} else {
	    throw new IndexOutOfBoundsException("queue is empty");
	}
    }

    /** Appends "x" to the queue. */
    public void enqueue(int x) {
	if (num == buf.length) {
	    enlarge();
	}
	buf[(first + num) & (buf.length - 1)] = x;
	num++;
    }

    /** Returns true iff the queue is empty. */
    public boolean empty() {
	return (num == 0);
    }

    /** Enlarge the queue to allow it to hold at least one more element. */
    private void enlarge() {
	int[] old = buf;
	int old_size = old.length;
	int n = num;
	int f = first;
	int[] copy = new int[old_size << 1];

	/* chunk1 is size of old queue contents from "first" onwards.
	   chunk2 is size of any queue contents wrapped around past
	   "chunk1" */
	int chunk1, chunk2;
	if ((f + n) > old_size) {
	    // Wrapped...
	    chunk1 = old_size - f;
	    chunk2 = n - chunk1;
	} else {
	    chunk1 = n;
	    chunk2 = 0;
	}

	for (int i = 0; i < chunk1; i++) {
	    copy[i] = old[f + i];
	}
	for (int i = 0; i < chunk2; i++) {
	    copy[chunk1 + i] = old[i];
	}
	buf = copy;
	first = 0;
    }
}
