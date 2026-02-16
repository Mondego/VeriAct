/*
 Adaptation of Bergstra & Loots's Java Class Family number 93:
      
 Booleans have been changed into integers (true = -2, false = -1),
 and results are written in a static array of integers, not printed.
      
 Monday 28 June 99 15:35:10 bart@irritatie.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */
    
package jcf93;

class c {
    static void m() {
	c1 y1;
	c2 y2;
	c3 y3 = new c3(); // only this one matters, BJ.
	c4 y4;
	c5 y5;
	c6 y6;
	c7 y7;
	U y;
	/*
         c1 y1 = new c1();
         c2 y2 = new c2();
         c3 y3 = new c3(); // only this one matters, BJ.
         c4 y4 = new c4();
         c5 y5 = new c5();
         c6 y6 = new c6();
         c7 y7 = new c7();
         U y = new U();
         */
	
	// implicit casting
	//bcop.pw(y3.b);
	R.store(y3.b);
	//bcop.pw(y3.f());
	/*
         NB. It is subtle to take the side effect into account:
         The array index is evaluated first. And since R.pos is
         incremented once inside f(), the result should be stored
         at R.pos + 1, to prevent unwanted overriding
         */
	R.store(y3.f()); 
	
	//cco.g();
	y4 = y3;
	//bcop.pw(y4.b);
	R.store(y4.b);
	//bcop.pw(y4.f());
	R.store(y4.f());
	//cco.g();
	//cco.a();

	// explicit casting
	//bcop.pw(((c4) y3).b);    
	R.store(((c4) y3).b);
	//cco.gw();
	//bcop.pw(((c4) y3).f());  
	R.store(((c4) y3).f());
	//cco.gw();
	//bcop.pw(((c5) y3).b);   
	R.store(((c5) y3).b); 
	//cco.gw();
	//bcop.pw(((c5) y3).f());  
	R.store(((c5) y3).f());
	//cco.g();
	//cco.b();
	
	// repeated explicit casting
	//bcop.pw(((c4)(c7) y3).b);    
	R.store(((c4)(c7) y3).b);  
	//cco.gw(); 
	//bcop.pw(((c4)(c7) y3).f());    
	R.store(((c4)(c7) y3).f());
	//cco.gw();
	//bcop.pw(((c4)(c7)(c5) y3).b);  
	R.store(((c4)(c7)(c5) y3).b);
	//cco.g();
	//bcop.pw(((c4)(c7)(c5) y3).f());
	R.store(((c4)(c7)(c5) y3).f());	
	//cco.gw();
	//bcop.pw(((c5) (U) y3).b);      
	R.store(((c5) (U) y3).b);
	//cco.gw();
	//bcop.pw(((c5) (U) y3).f());  
	R.store(((c5) (U) y3).f());  
	//cco.g();
	//cco.c();
	
	// overloaded method application - explicit casting
	//bcop.pw(F.f(y3));  
	R.store(F.f(y3));               
	//cco.g();
	//bcop.pw(F.f((c4) y3));   
	R.store(F.f((c4) y3));            
	//cco.g();
	//bcop.pw(F.f((c5) y3));  
	R.store(F.f((c5) y3));             
	//cco.g();
	//bcop.pw(F.f((c4) (c7) (c5) y3));  
	R.store(F.f((c4) (c7) (c5) y3));   
	//cco.g();
	//cco.d();
	
	// overloaded method application -implicit casting
	y4 = y3;
	//bcop.pw(F.f(y4));  
	R.store(F.f(y4));
	//cco.g();
	y5 = y3;
	//bcop.pw(F.f(y5));  
	R.store(F.f(y5));
	//cco.g();
	y6 = y3;
	//bcop.pw(F.f(y6)); 
	R.store(F.f(y6));
	//cco.g();
	y5 = (c4) y3;
	//bcop.pw(F.f(y5));
	R.store(F.f(y5));	   
	//cco.g();
	y6 = (c5) y3;
	//bcop.pw(F.f(y6));
	R.store(F.f(y6));
	//cco.g();
	y6 = (c5)(c7) y3;
	//bcop.pw(F.f(y6));
	R.store(F.f(y6));	   
	//cco.g();
	//cco.e();
    }
}


class c1 extends c2 {
    int b = -2,c = -1;
    int f() {
	//cco.aw();
	R.store(1);	
	return -2;}}

class c2 extends c3 {
    int b = -1,c = -1;
    int f() {
	//cco.bw();
	R.store(2);	
	return -1;}}

class c3 extends c4 {
    int b = -2,c = -2;
    int f() {
	//cco.cw();
	R.store(3);	
	return -2;}}

class c4 extends c5 {
    int b = -1,c = -2;
    int f() {
	//cco.dw();
	R.store(4);	
	return -1;}}

class c5 extends c6 {
    int b = -2,c = -1;
    int f() {
	//cco.ew();
	R.store(5);	
	return -2;}}

class c6 extends c7 {
    int b = -1,c = -1;
    int f() {
	//cco.fw();
	R.store(6);	
	return -1;}}

class c7 extends U {
    int b = -2,c = -2;
    int f() {
	//cco.gw();
	R.store(7);	
	return -2;}}

class U {
    int b = -1,c = -2;
    int f() {
	//cco.gw();
	R.store(8);	
	return -1;}}


class F {
    static int f(c1 x) {
	//bcop.pw(x.b);
	R.store(x.b);	
	//bcop.pw(x.f());
	R.store(x.f());	
	//cco.aw();
	R.store(1);	
	return x.b;}
    
    static int f(c2 x) {
	//bcop.pw(x.b);
	R.store(x.b);	
	//bcop.pw(x.f());
	R.store(x.f());	
	//cco.bw();
	R.store(2);	
	return x.b;}

    static int f(c3 x) {
        //bcop.pw(x.b);
        R.store(x.b);    
        //bcop.pw(x.f());
        R.store(x.f());    
        //cco.cw();
        R.store(3);    
        return x.b;}

    static int f(c4 x) {
        //bcop.pw(x.b);
        R.store(x.b);    
        //bcop.pw(x.f());
        R.store(x.f());    
        //cco.dw();
        R.store(4);    
        return x.b;}

    static int f(c5 x) {
        //bcop.pw(x.b);
        R.store(x.b);    
        //bcop.pw(x.f());
        R.store(x.f());    
        //cco.ew();
        R.store(5);    
        return x.b;}

    static int f(c6 x) {
        //bcop.pw(x.b);
        R.store(x.b);    
        //bcop.pw(x.f());
        R.store(x.f());    
        //cco.fw();
        R.store(6);    
        return x.b;}

    static int f(c7 x) {
        //bcop.pw(x.b);
        R.store(x.b);    
        //bcop.pw(x.f());
        R.store(x.f());    
        //cco.gw();
        R.store(7);    
        return x.b;}

    static int f(U x) {
        //bcop.pw(x.b);
        R.store(x.b);    
        //bcop.pw(x.f());
        R.store(x.f());    
        //cco.hw();
        R.store(9);    
        return x.b;}
}


class R {
  
    static int[] r = new int[100];
    static int pos = 0;

    static void store(int result) {
	/* Note that the result gets evaluated first */
	r[pos++] = result;
    }
}

class Test {

    public static void main(String[] args) {
	c.m();
	for(int i = 0; i < 100; i++) {
	    System.out.println(R.r[i]);
	}
    }
}
