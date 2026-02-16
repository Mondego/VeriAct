import java.util.Vector;

public class ClassEquiv {

    public void m(Object o) {
        //@ assume o instanceof Vector;
        //@ assert \typeof(o) <: \type(Vector);
        //@ assert o.getClass() <: \type(Vector);
    }

    public void m1(Object o) {
        //@ assume \typeof(o) <: \type(Vector);
        //@ assert o instanceof Vector;
    }

    public void m1a(Object o) {
        //@ assume o.getClass() <: \type(Vector);
        //@ assert o instanceof Vector;
    }

    public void m2(Object o) {
        //@ assume o instanceof Vector;
        //@ assert \typeof(o) <: Vector.class;
        //@ assert o.getClass() <: Vector.class;
    }

    public void m3(Object o) {
        //@ assume \typeof(o) <: Vector.class;
        //@ assert o instanceof Vector;
    }

    public void m3a(Object o) {
        //@ assume o.getClass() <: Vector.class;
        //@ assert o instanceof Vector;
    }

    public void m4(Object o) {
        //@ assume \typeof(o) == \type(Vector);
        //@ assert o instanceof Vector;
    }

    //@ requires o != null;
    public void m5(Object o) {
        //@ assume \typeof(o) == Vector.class;
        //@ assert o instanceof Vector;
    }

    public void mq(Vector o) {
        //@ assert \typeof(o) == o.getClass();
        //@ assert \type(Vector) == Vector.class;
    }


}
