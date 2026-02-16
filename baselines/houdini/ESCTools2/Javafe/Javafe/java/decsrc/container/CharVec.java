package decsrc.container;

/**
 * Expandable vectors of _TYPE_ (automatically generated via m4
 * from "decsrc/container/Vec.j")
 *
 * <p> This class is thread safe, but instances of this class are not.
 * That is, two different threads may safely access two different
 * instances of this class concurrently, but they may not safely
 * access the same instance of this class without external
 * synchronization. </p>
 */

public final class CharVec
{
    //@ private invariant count <= elements.length;
    private int count;
    private /*@non_null*/char[] elements;

    public CharVec () { this(1); }

    /*@ private normal_behavior
      @   ensures elements != null;
      @   ensures elements.length == (predict < 1 ? 1 : predict);
      @*/
    public CharVec (int predict) {
        if (predict < 1) predict = 1;
        this.elements = new char[predict];
    }

    public int length() { return count; }

    public boolean empty() { return count == 0; }

    public char elementAt(int index) {
        if (count <= index) throw new ArrayIndexOutOfBoundsException(index);
        return elements[index];
    }

    public void setElementAt(char x, int index) {
        if (count <= index) throw new ArrayIndexOutOfBoundsException(index);
        elements[index] = x;
    }

    public char[] toArray() {
        char[] result = new char[count];
        System.arraycopy(elements, 0, result, 0, count);
        return result;
    }

    // XXX Exposes the rep.  Only valid until next mutation.
    public char[] contents() {
        return elements;
    }

    public boolean contains(char x) {
        return (find(x) >= 0);
    }

    /** If <code>this</code> contains <code>x</code>, return an index
     <code>i</code> such that <code>this.elementAt(i) == x</code>.
     Else return -1. */
    public int find(char x) {
        for(int i = 0; i < count; i++)
            if (elements[i] == x) return i;
        return -1;
    }

    public void append(char x) {
        if (count == elements.length) expand();
        elements[count++] = x;
    }

    public void appendArray(/*@non_null*/char[] list) {
        appendArray(list, 0, list.length);
    }

    public void appendArray(/*@non_null*/char[] list, int start, int num) {
        while ((count + num) > elements.length) {
            expand();
        }
        System.arraycopy(list, start, elements, count, num);
        count += num;
    }    

    public void appendVector(/*@non_null*/CharVec list) {
        appendArray(list.elements, 0, list.count);
    }

    public void appendVector(/*@non_null*/CharVec list, int start, int n) {
        if (start < 0) throw new ArrayIndexOutOfBoundsException(start);
        if ((start + n - 1) >= list.count)
            throw new ArrayIndexOutOfBoundsException(start + n - 1);
        appendArray(list.elements, start, n);
    }


    // Operations specific to vectors of characters
    public /*@non_null*/String toString() {
        return new String(elements, 0, count);
    }

    public void appendString(/*@non_null*/String s) {
        appendString(s, 0, s.length());
    }

    public void appendString(/*@non_null*/String s, int start, int num) {
        while ((count + num) > elements.length) {
            expand();
        }
        s.getChars(start, start + num, elements, count);
        count += num;
    }


    /** Insert "x" just before the element at "index".  After this
     routine returns, "x" is at "index", and succeeding elements
     have all been shifted to one higher index. */
    public final void insertElementAt(char x, int index) {
        if (count < index) throw new ArrayIndexOutOfBoundsException(index);
        if (count == elements.length) expand();
        System.arraycopy(elements, index, elements, index+1, count - index);
        elements[index] = x;
        count++;
    }

    public char remove() {
        char result = elements[count-1];
        count--;
        return result;
    }

    /** Remove last "n" elements from the vector. */
    public void remove(int n) {
        if (n < 0) throw new ArrayIndexOutOfBoundsException(count);
        if (n > count) throw new ArrayIndexOutOfBoundsException(-1);
        count -= n;
    }

    /** Remove first occurrence (if any) of "x".  Returns true iff an
     occurrence was found. */
    public boolean removeElement(char x) {
        int index = find(x);
        if (index >= 0) {
            System.arraycopy(elements, index+1, elements, index, count - index - 1);
            count--;
            return true;
        } else {
            return false;
        }
    }

    public char removeElementAt(int index) {
        if (count <= index) throw new ArrayIndexOutOfBoundsException(index);
        char result = elements[index];
        System.arraycopy(elements, index+1, elements, index, count - index - 1);
        count--;
        return result;
    }

    public final void removeAllElements() {
        count = 0;
    }

    /** Replace all occurrences of "x" with "y" in "this".  Returns the
     number of replaced occurrences. */
    public int replaceElement(char x, char y) {
	int num = 0;
	for (int i = 0; i < count; i++) {
	    if (elements[i] == x) {
		elements[i] = y;
		num++;
	    }
	}
	return num;
    }

    /** Copies this vector.  The elements are <strong>not</strong> copied. */
    public CharVec copy() {
        CharVec v = new CharVec (count);
        v.count = count;
        System.arraycopy(elements, 0, v.elements, 0, count);
        return v;
    }

    private void expand() {
        char[] newElements = new char[2*elements.length];
        System.arraycopy(elements, 0, newElements, 0, elements.length);
        elements = newElements;
    }
}
