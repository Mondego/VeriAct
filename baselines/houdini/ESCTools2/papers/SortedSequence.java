public class SortedSequence  {
	private /*@ spec_public */ int[] seq = new int[9];
	private /*@ spec_public */ int size = 0;
	//@ invariant seq != null;
	//@ invariant size >= 0 && size <= seq.length;

	//@ assignable seq, seq[*],size;
	public void push(int i) {
	    if (size == seq.length) {
	        int[] newseq = new int[2 * seq.length + 1];
	        System.arraycopy(seq,0,newseq,0,size);
	        seq = newseq;
	    }
	    seq[size++] = i;
	}

	//@ requires size > 0;
	//@ assignable seq,seq[*],size;
	//@ ensures size == \old(size-1);
	//@ ensures \result == sort()[size];
	public int pop() {
	    seq = sort();
	    return seq[--size];
	}

	//@ ensures \result != null;
	//@ ensures \result.length >= size;
	//@ ensures (* result is sorted *);
	//@ ensures (* result has same elements as seq *);
	/*@ pure */ private int[] sort() { 
	    int[] result = new int[seq.length];
	    /*...*/ 
	    return result;
	}
}
