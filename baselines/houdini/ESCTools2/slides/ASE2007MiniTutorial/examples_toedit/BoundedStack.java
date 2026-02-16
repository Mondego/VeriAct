public class BoundedStack {

    private Object[] elems;
    private int size = 0;

    public BoundedStack(int n) {
        elems = new Object[n];
    }

    public void push(Object x) {
        elems[size] = x;
        size++;
    }

    public void pop() {
        size--;
        elems[size] = null;
    }

    public Object top() {
        return elems[size-1];
    }
}
