package escjava.dfa.cfd;

/**
 * This class represent a list of nodes.
 */
public class NodeList {
    protected/*@ non_null @*/Node[] nodes;

    //@ invariant count >= 0;
    //@ invariant nodes.length >= count;
    protected int count;

    //@ invariant (\forall int i; i >= 0 & i < count; nodes[i] != null);
    //@ invariant \typeof(nodes) == \type(Node[]);

    /**
     * Initializes a new instance of NodeList class. The node list is
     * initialized with an empty list.
     */
    /*@ public normal_behavior
      @   ensures getCount() == 0;*/
    /*@ also private normal_behavior
      @    ensures count == 0; */
    public NodeList() {
        this.nodes = new Node[1];
        this.count = 0;
    }
    
    
    /**
     * Add a node to this list
     * 
     * @param node
     *            the node to be added
     */
    //@ public normal_behavior
    //@   ensures getCount() == \old(getCount()) + 1;
    //@ also protected behavior
    //@   ensures count == \old(count) + 1;
    //@   assignable nodes, nodes[*];
    //@   assignable count;
    public void addNode(/*@ non_null @*/Node node) {
        resize(count + 1);
        nodes[count] = node;
        ++count;
    }

    //@ protected behavior
    //@    ensures \result == count;
    /*@ pure @*/ public int getCount() {
        return count;
    }

    /*@ public normal_behavior
      @    ensures \result <==> getCount() == 0; */
    /*@ also protected behavior
      @    ensures \result <==> count == 0; */
    /*@ pure @*/ public boolean isEmpty() {
        return count <= 0;
    }

    /**
     * Returns the enumeration that enumerates the nodes
     */
    /*@ pure @*/ public /*@ non_null @*/Enumeration elements() {
        return new Enumeration();
    }

    /**
     * Appends the given list to this one.
     */
    public void append(/*@ non_null @*/NodeList ns) {
        resize(count + ns.count);
        for (int i = 0; i < ns.count; i++)
            nodes[i + count] = ns.nodes[i];
        this.count = count + ns.count;
    }

    /**
     * Clears this list.
     */
    //@ protected behavior
    //@   ensures count == 0;
    public void clear() {
        this.count = 0;
    }

    /**
     * Returns true iff this list is empty
     */
    //@ protected behavior
    //@    ensures \result <==> this.count <= 0;
    public/*@ pure @*/boolean empty() {
        return this.count <= 0;
    }

    /**
     * Returns true iff this list contains the given node <code>n</node>, does not use equals.
     */
    //@ protected behavior
    //@    ensures \result <==> (\exists int j; j >= 0 & j < count; nodes[j] == n);
    /*@ pure @*/ public boolean member(Node n) {
        //@ loop_invariant i >= 0 & i <= count;
        //@ loop_invariant (\forall int j; j >= 0 & j < i; nodes[j] != n);
        for (int i = 0; i < this.count; i++)
            if (n == this.nodes[i])
                return true;

        return false;
    }

    /**
     * Resizes the <code>nodes</code> fields so it is at least
     * <code>newSize</code> long.
     * 
     * @param newSize
     *            desired minimal size of the <code>nodes</code> field
     */
    //@ assignable nodes;
    //@ ensures nodes.length >= newSize;
    private void resize(int newSize) {
        if (this.nodes.length < newSize) {
            int newLength = Math.max(this.nodes.length * 2, newSize);
            Node[] newNodes = new Node[newLength];
            for (int i = 0; i < this.count; i++)
                newNodes[i] = this.nodes[i];
            this.nodes = newNodes;
        }
    }


    //@ requires index >= 0 & index < getCount();    
    /*@ pure @*/ public Node getAt(int index) {
        return nodes[index];
    }
    
    public class Enumeration {
        private int currentItem = 0;

        //@ private invariant currentItem >= 0;

        /*
         * Tests if this enumeration contains more elements.
         */
        //@ private behavior
        //@   ensures \result <==> currentItem < count;
        public/*@ pure @*/boolean hasMoreElements() {
            return currentItem < count;
        }

        /*
         * Returns the next element of this enumeration, if this enumeration
         * object has at least one more element to provide.
         */
        //@ requires hasMoreElements();
        public/*@ non_null @*/Node nextElement() {
            Node retv = nodes[currentItem];
            currentItem++;
            return retv;
        }
    }

}
