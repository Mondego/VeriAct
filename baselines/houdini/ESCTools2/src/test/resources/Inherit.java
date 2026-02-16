public class Inherit extends InheritS {

   //@ also
   //@ requires k >= -1;
   //@ ensures \result == 1;
   public int m(int k);

}

class InheritB extends InheritS {

   //@ also
   //@ requires k >= -2;
   //@ ensures \result == 2;
   public int m(int k);

}

class InheritS {

   //@ requires k >= 0;
   //@ ensures \result >=0;
   //@ pure
   public int m(int k);

}

class M {

    public void mm() {
       InheritS i = new InheritS();
       int j = i.m(0);
       //@ assert j >= 0; 
    }
    public void mm1() {
       InheritS i = new Inherit();
       int j = i.m(0);
       //@ assert \typeof(i) <: Inherit.class;
       //@ assert j >= 0; 
       //@ assert j == 1;  // OK - if we had dynamically type specs FIXME
    }
    public void mm1a() {
       InheritS ii = new Inherit();
       //@ assert \typeof(ii) <: Inherit.class;
       Inherit i = (Inherit)ii;
       int j = i.m(0);
       //@ assert j >= 0; 
       //@ assert j == 1; 
    }
    public void mm2() {
       InheritS i = new InheritB();
       int j = i.m(0);
       //@ assert \typeof(i) <: InheritB.class;
       //@ assert j >= 0; 
       //@ assert j == 2;  // OK - if we had dynamically type specs FIXME
    }
    public void mm2a() {
       InheritS ii = new InheritB();
       //@ assert \typeof(ii) <: InheritB.class;
       InheritB i = (InheritB)ii;
       int j = i.m(0);
       //@ assert j >= 0; 
       //@ assert j == 2; 
    }
    public void mm3() {
       Inherit i = new Inherit();
       int j = i.m(0);
       //@ assert j >= 0; 
       //@ assert j == 1; 
    }
    public void mm4() {
       InheritB i = new InheritB();
       int j = i.m(0);
       //@ assert j >= 0; 
       //@ assert j == 2; 
    }

    //@ requires i != null;
    public void p(InheritS i) {
       //@ assume i instanceof Inherit;
       //@ assert i.m(0) == 1;  // OK - if we had dynamically typed specs FIXME
    }

    //@ requires i != null;
    public void q0(InheritS i) {
       i.m(0);  // OK
    }
    //@ requires i != null;
    public void q1(InheritS i) {
       i.m(-1);  // ERROR
    }
    //@ requires i != null;
    public void q2(InheritS i) {
       i.m(-3);  // ERROR
    }
    //@ requires i != null;
    public void q3(InheritS i) {
       if (i instanceof Inherit) i.m(-1); // OK - if we had dynamically typed specs FIXME
    }
    //@ requires i != null;
    public void q4(InheritS i) {
       if (i instanceof Inherit) i.m(-2); // ERROR
    }
}
