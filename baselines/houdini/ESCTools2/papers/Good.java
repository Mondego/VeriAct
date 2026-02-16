class Example {
    public int i;
}
public class Good {
    /*@ ensures o!=null ==> \result == o.i;
        diverges false;
        signals (Exception e) o == null; */
    //@ pure
    static public int valueOfI(Example o);

    //@ ensures valueOfI(o) > 0;
    public void init(Example o);
}
