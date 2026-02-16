public class MartijnsPoint{

     /*@ normal_behavior
       @  requires true;
       @ assignable \nothing;
       @ ensures  (\typeof(this) == \type(MartijnsPoint)) ==> \result == 1;
       @*/
    //@ pure
     int equals(MartijnsPoint x){return 1;}
}

class ColorPoint extends MartijnsPoint{

     /*@ also
       @ normal_behavior
       @ requires \typeof(this) <: \type(ColorPoint);
       @ assignable \nothing;
       @ ensures  \result == 2;
       @*/
    //@ pure
    int equals(MartijnsPoint x){return 2;}
}

  class Inheritance{

      public int r1,r2,r3,r4,r5,r6,r7,r8,r9;

      /*@ normal_behavior
        @  requires true;
        @   assignable r1,r2,r3,r4,r5,r6,r7,r8,r9;
        @  ensures  true;
        @*/
      void m(){
        MartijnsPoint p1 = new MartijnsPoint();
        MartijnsPoint p2 = new ColorPoint();
        ColorPoint cp = new ColorPoint();
        r1 = p1.equals(p1);
        //@ assert r1 == 1;
        r2 = p1.equals(p2);
        //@ assert r2 == 1;
        r3 = p2.equals(p1);
        //@ assert r3 == 2;  //ESC/Java2 gives a warning here.
        r4 = p2.equals(p2);
        // @ assert r4 == 2;  //ESC/Java2 gives a warning here.
        r5 = cp.equals(p1);
        //@ assert r5 == 2;
        r6 = cp.equals(p2);
        //@ assert r6 == 2;
        r7 = p1.equals(cp);
        //@ assert r7 == 1;
        r8 = p2.equals(cp);
        // @ assert r8 == 2;  //ESC/Java2 gives a warning here.
        r9 = cp.equals(cp);
        //@ assert r9 == 2;
      }

      /*@ normal_behavior
        @  requires true;
        @   assignable r3;
        @  ensures  true;
        @*/
      void mm(MartijnsPoint p1, MartijnsPoint p2){
	//@ assume p1 != null && \typeof(p1) == \type(MartijnsPoint);
	//@ assume p2 != null && \typeof(p2) == \type(ColorPoint);
        r3 = p2.equals(p1);
        //@ assert r3 == 2;  //ESC/Java2 gives a warning here.
      }

      /*@ normal_behavior
        @  requires true;
        @   assignable r3;
        @  ensures  true;
        @*/
      void mmm(MartijnsPoint p1, ColorPoint p2){
	//@ assume p1 != null && \typeof(p1) == \type(MartijnsPoint);
	//@ assume p2 != null && \typeof(p2) == \type(ColorPoint);
        r3 = p2.equals(p1);
        //@ assert r3 == 2;  //ESC/Java2 gives a warning here.
      }
  }


