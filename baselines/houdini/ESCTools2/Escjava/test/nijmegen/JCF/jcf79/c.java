/*
 Adaptation of Bergstra & Loots's Java Class Family number 79:
  
 The results are stored in booleans of a special class, and not printed.
  
 Thursday 24 June 99 10:00:47 bart@cicero.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf79;

public class c {

    static c2 x = new c2();
    public static Return m(){ return x.m();}
}  

class c1 {
    boolean q = true;
}

class c11 extends c1 {
    private boolean q = false;
    boolean r = q;
    boolean s = ((c1) this).q;
}

class c12 extends c11 {
    boolean q = true;
}

class c2 extends c12 {
    public Return m(){
	Return z = new Return();
	z.return1 = q;
	z.return2 = r;
	z.return3 = s;
	return z;
    }
}


class Return {

    boolean return1, return2, return3;

}
