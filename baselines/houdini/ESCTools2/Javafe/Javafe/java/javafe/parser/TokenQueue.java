/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

import javafe.util.Assert;

class TokenQueue
{
    //// Instance variables

    /**
     * Contents of queue tokens.  Used as circular buffer with
     * <code>start</code> and <code>end</code> being output and input
     * pointers, respectively.  <code>toks[start]</code> is first
     * element; <code>toks[(end + toks.len - 1) % toks.len]</code> is
     * last element.
     */

    //@ invariant \nonnullelements(toks);
    //@ spec_public
    private ParserToken[] toks;

    //@ invariant 0 <= start && start < toks.length;
    //@ spec_public
    private int start;

    //@ invariant 0 <= end && end < toks.length;
    //@ spec_public
    private int end;


    //@ ensures !notempty;
    public TokenQueue() {
        toks = new ParserToken[4];
        for(int i = 0; i < toks.length; i++)
            toks[i] = new ParserToken();
        start = end = 0;
    }


    /** Do not write.  True iff queue is not empty. */  //
    //@ invariant notempty == (end != start );
    public boolean notempty = false;

    // Invariants (size == ((end + toks.length) - start) % toks.length):


    //// Methods

    /** Returns number of items in token queue. */

    //@ ensures \result>=0;
    //@ ensures \result>0 == notempty;
    public int size() {
        int sa = (start <= end ? 0 : toks.length);
        return (end + sa) - start;
    }

    /** Returns <code>n</code>th element in token queue.
     <pre><esc>
     requires 0 <= n;
     </esc></pre>    
     */

    //@ ensures \result != null;
    public ParserToken elementAt(int n) {
        int sa = (start <= end ? 0 : toks.length);
        int size = (end + sa) - start;
        if (n < 0 || size <= n)
            throw new IndexOutOfBoundsException();
        int ndx = start + n;
        if (toks.length <= ndx) ndx -= toks.length;
        return toks[ndx];
    }   //@ nowarn Exception;

    public void setElementAt(int n,ParserToken t) {
        int sa = (start <= end ? 0 : toks.length);
        int size = (end + sa) - start;
        if (n < 0 || size <= n)
            throw new IndexOutOfBoundsException();
        int ndx = start + n;
        if (toks.length <= ndx) ndx -= toks.length;
        toks[ndx] = t;
    }   //@ nowarn Exception;


    /** Empties lookahead queue. */

    //@ modifies end, notempty, start;
    //@ ensures !notempty;
    public void clear() {
        end = start = 0;
        notempty = false;
        for(int i = 0; i < toks.length; i++) {
            ParserToken t = toks[i];
            //@ assert 0 <= i && i < toks.length;
            //@ assert (\forall int j; 0 <= j && j < toks.length ==> toks[j] != null );
            //@ assert toks[0] != null;
            //@ assert t != null;
            t.clear();
        }
    }

    /** Removes head of token queue.  Requires: notempty
     <pre><esc>
     requires dst != null;
     </esc></pre>
     */

    //@ requires notempty;
    //@ modifies start;
    //@ modifies notempty;
    public void dequeue(ParserToken dst) {
        if (start != end) {
            toks[start].copyInto(dst);

            // Clear token to allow GC:
            toks[start].clear();

            start = start+1;
            if (start == toks.length) start = 0; 
            notempty = (start != end);
        } else Assert.precondition(false);
    }

    /** Pushes a token onto the lookahead queue.
     <pre><esc>
     requires td != null;
     </esc></pre>
     */

    public void enqueue(ParserToken td) {
        Assert.notNull(td);

        // We always have space at end
        td.copyInto(toks[end]);
        end=end+1;
        if (end == toks.length) end = 0; 

        if (start == end) {
            // Out of space, need to extend array, double it
            int len = toks.length;
            ParserToken[] _new = new ParserToken[2*len];
            for(int i = 0; i < len; i++) _new[i] = toks[(i + start) % len];
            for(int i = _new.length-1; len <= i; i--) _new[i] = new ParserToken();
            start = 0;
            end = toks.length;
            toks = _new;
        }
        notempty = true;
    }

    //@ ensures \result != null;
    private String stateToString() {
        return ("start: " + start
                + " end: " + end
                + " length: " + toks.length);
    }

    public void zzz(String prefix) {
        Assert.notFalse(notempty == (end != start), prefix+"notempty not correct");
		    
        int len = toks.length;
        int size = size();

        Assert.notFalse(0 <= start && start < len, prefix + "start out of bounds");
        Assert.notFalse(0 <= end && end < len, prefix + "end out of bounds");
        for(int i = 0; i < len; i++) {
            Assert.notNull(toks[i], (prefix + "bad lookahead queue: "
                                     + stateToString() + " at: " + i));
            int ndx = (i + start) % len;
            if (i < size) toks[ndx].zzz();
            else Assert.notFalse(toks[ndx].auxVal == null,   //@ nowarn Pre;
                                 prefix + "bad lookahead queue: "
                                 + stateToString() + " at: " + ndx);
        }
    }
}
