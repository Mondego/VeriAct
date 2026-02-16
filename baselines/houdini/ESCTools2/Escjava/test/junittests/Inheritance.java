class Super {
    //@ ensures \result >= 0;
    /*@ pure */ public int m();
}

class Sub extends Super {
    //@ also ensures \result <= 0;
    /*@ pure */ public int m();
}

public class Inheritance {
    Inheritance();

    public void mm(/*@ non_null */ Super s) {
        //@ assert s.m() >= 0;              // Line A
        if (s instanceof Sub) {
            Sub ss = (Sub)s;
            //@ assert ss.m() == 0;         // Line B
            //@ assert ((Sub)s).m() == 0;   // Line C
            //@ assert s.m() == 0;          // Line D 
        }
    }

    public void m3(/*@ non_null */ Super s) {
        if (s instanceof Sub) {
            //@ assert s.m() == 0;         // FIXME should be ok
        }
    }

    public void m2(/*@ non_null*/ Super s) {
        //@ assert s.m() == 0; // ERROR
    }
}
