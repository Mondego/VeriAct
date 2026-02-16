class List
{
    static List freelist;

    int head;
    List tail;

    //@ public normal_behavior
    //@   modifies this.head, this.tail;
    //@   ensures this.head == head;
    //@   ensures this.tail == tail;
    List(int head, List tail) {
	this.head = head;
	this.tail = tail;
    }

    //@ public normal_behavior
    //@   requires n >= 1;
    //@   modifies freelist;
    //@   ensures freelist != null;
    //@   ensures (* the length of the freelist is n *);
    //@   ensures (* each element in the freelist has head == 0 *);
    public static void init(int n) {
        // initialise freelist
	for(int i = 1; i <= n; i++) {
	    freelist = new List(0, (List)freelist);
	}
    }

    //@ public normal_behavior
    //@   requires l != null;
    //@   modifies l.tail, freelist;
    //@   ensures l.tail == \old(freelist);
    //@   ensures freelist == l;
    public static void free(List l) {
	l.tail = freelist;
	freelist = l;
    }

    //@ public normal_behavior
    //@   requires freelist != null;
    //@   modifies freelist;
    //@   ensures \result != null;
    //@   ensures \result == \old(freelist);
    //@   ensures freelist == \old(freelist.tail);
    //@ also
    //@ public normal_behavior
    //@   requires freelist == null;
    //@   ensures false;
    public static List make() {
	if (freelist != null) {
	    List res = freelist;
	    freelist = freelist.tail;
	    return res;
	} else {
            //@ assert false;
	    System.out.println("Called new within make");
	    return new List(0, (List)null);
	}
    }

    //@ public normal_behavior
    //@   ensures \result == null;
    public static /*@ pure @*/ List nil() {
	return null;
    }

    //@ public normal_behavior
    //@   requires freelist != null;
    //@   modifies freelist;
    //@   ensures \result != null;
    //@   ensures \result.head == i;
    //@   ensures \result.tail == l;
    //@   ensures_redundantly head(\result) == i;
    //@   ensures_redundantly tail(\result) == l;
    public static List cons(int i, List l) {
	List res = make();
	res.head = i;
	res.tail = l;
	return res;
    }

    //@ public normal_behavior
    //@   ensures \result <==> (l == null);
    public static /*@ pure @*/ boolean isnil(List l) {
	return (l == null);
    }

    //@ public normal_behavior
    //@   requires l != null;
    //@   ensures \result == l.head;
    public static /*@ pure @*/ int head(List l) {
	return l.head;
    }

    //@ public normal_behavior
    //@   requires l != null;
    //@   ensures \result == l.tail;
    public static /*@ pure @*/ List tail(List l) {
	return l.tail;
    }
}


























