/*
 Adaptation of Bergstra & Loots's Java Class Family number 86:
  
 The result is returned as boolean, not printed.
  
 Thursday 24 June 99 16:42:29 bart@cicero.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf86;

public class c {

    static c2 x = new c2();

    public static void m(){
	//cco.bw();
	Result.r[Result.pos] = 2;
	Result.pos++;
	x.m();
	//bco.f();
	Result.r[Result.pos] = -1;
	Result.pos++;
	c2 y = new c2(c2.q);
	//cco.e();
	Result.r[Result.pos] = 5;
	Result.pos++;
    }
}  

class c1 {

    static boolean q = true;

    c1() {
	q = false;
	//bco.t();
	Result.r[Result.pos] = -2;
	Result.pos++;
    }
    
    c1(boolean b) {
	//cco.cw();
	Result.r[Result.pos] = 3;
	Result.pos++;
	q = false;
	//bcop.pw(!b);
	Result.r[Result.pos] = !b?-2:-1;
	Result.pos++;
    }
}
 
class c2 extends c1 {

    c2() {
	super(false);
	//cco.aw();
	Result.r[Result.pos] = 1;
	Result.pos++;
    }

    c2(boolean b) {
	super(!b);
	//cco.dw();
	Result.r[Result.pos] = 4;
	Result.pos++;
	q = !q;
    }

    public void m(){
	//bcop.p(c1.q);
	Result.r[Result.pos] = c1.q?-2:-1;
	Result.pos++;
    }
}


class Result {

    static int[] r = new int[10];
    static int pos = 0;

}
