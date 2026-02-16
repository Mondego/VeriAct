/*
 Adaptation of Bergstra & Loots's Java Class Family number 97:
    
 Monday 28 June 99 17:01:06 bart@irritatie.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */  

class jcf97 {
  
    static c1 x1 = new c1();

    static void m(){
        c2 x2 = new c2();
        x1.m1();
        x2.m1();
        R.store(2);
        //cco.b();
        i ivar = (i) new  c2();
        R.store(2);
        //cco.b();
        ivar.m1();
        R.store(2);
        //cco.b();
        ivar = (i) x1;
        ivar.m1();
    }
}

class c1 {
    c1() {
        R.store(1);
        //cco.aw();
    }
    public void m1(){
        R.store(5);
        //cco.ew(); 
    }
}

class c2 extends c1 implements i {
    c2() {super();}
}


interface i {void m1();}


class R {
  
    static int[] r = new int[100];
    static int pos = 0;

    static void store(int result) {
	/* Note that the result gets evaluated first */
	r[pos++] = result;
    }
}
