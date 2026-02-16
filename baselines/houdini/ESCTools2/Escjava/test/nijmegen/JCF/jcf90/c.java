/*
 Adaptation of Bergstra & Loots's Java Class Family number 90:
  
 The results are stored in a static array of booleans, not printed.
  
 Monday 28 June 99 13:40:45 bart@irritatie.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf90;

public class c {

    static void m(){
        c1 x = new c1();
        c2 y = new c2();
        C z = new C();
        //bcop.pw(x.b);
        R.r[R.pos] = x.b;
        R.pos++;
        x.m1();
        x.m2();
        x.m3();
        //cco.d();
        //bcop.pw(y.b);
        R.r[R.pos] = y.b;
        R.pos++;
        y.m1();
        y.m2();
        y.m4();
        //cco.d();
        //bcop.pw(z.b);
        R.r[R.pos] = z.b;
        R.pos++;
        z.m1();
        z.m2();
        //cco.d();cco.d();
     
        i p,q,r,s;
        C u,v,w;
        p = x;q = y;
        u = x;v = y;
        w = x;r = w;
        w = y;s = w;
        //bcop.pw(x.b);
        R.r[R.pos] = x.b;
        R.pos++;
        //bcop.pw(x.c);
        R.r[R.pos] = x.c;
        R.pos++;
        x.m1();
        //bcop.p(x.m2());
        R.r[R.pos] = x.m2();
        R.pos++;

        //bcop.pw(p.b);
        R.r[R.pos] = p.b;
        R.pos++;
        //bcop.pw(p.c);
        R.r[R.pos] = p.c;
        R.pos++;
        p.m1();
        //bcop.p(p.m2());
        R.r[R.pos] = p.m2();
        R.pos++;

        //bcop.pw(u.b);
        R.r[R.pos] = u.b;
        R.pos++;
        //bcop.pw(u.c);
        R.r[R.pos] = u.c;
        R.pos++;
        u.m1();
        //bcop.p(u.m2());
        R.r[R.pos] = u.m2();
        R.pos++;
     
        //bcop.pw(r.b);
        R.r[R.pos] = r.b;
        R.pos++;
        //bcop.pw(r.c);
        R.r[R.pos] = r.c;
        R.pos++;
        r.m1();
        //bcop.p(r.m2());
        R.r[R.pos] = r.m2();
        R.pos++;
        //cco.e();
     
        //bcop.pw(y.b);
        R.r[R.pos] = y.b;
        R.pos++;
        //bcop.pw(y.c);
        R.r[R.pos] = y.c;
        R.pos++;
        y.m1();
        //bcop.p(y.m2());
        R.r[R.pos] = y.m2();
        R.pos++;

        //bcop.pw(q.b);
        R.r[R.pos] = q.b;
        R.pos++;
        //bcop.pw(q.c);
        R.r[R.pos] = q.c;
        R.pos++;
        q.m1();
        //bcop.p(q.m2());
        R.r[R.pos] = q.m2();
        R.pos++;

        //bcop.pw(v.b);
        R.r[R.pos] = v.b;
        R.pos++;
        //bcop.pw(v.c);
        R.r[R.pos] = v.c;
        R.pos++;
        v.m1();
        //bcop.p(v.m2());
        R.r[R.pos] = v.m2();
        R.pos++;

        //bcop.pw(s.b);
        R.r[R.pos] = s.b;
        R.pos++;
        //bcop.pw(s.c);
        R.r[R.pos] = s.c;
        R.pos++;
        s.m1();
        //bcop.p(s.m2());
        R.r[R.pos] = s.m2();
        R.pos++;
        //cco.f();
    }
}


class c1 extends C implements i {
    public boolean b = true;
    public boolean c = true;
    public void m1(){
        //bcop.pw(true);
        R.r[R.pos] = true;
        R.pos++;
    }
    public boolean m2(){
        //bcop.pw(true);
        R.r[R.pos] = true;
        R.pos++;
        return true;
    }
    public void m3(){
        //cco.aw();
    }
}

class c2 extends C implements i {
    public boolean b = false;
    public boolean c = false;
    public void m1(){
        //bcop.pw(false);
        R.r[R.pos] = false;
        R.pos++;
    }
    public boolean m2(){
        //bcop.pw(false);
        R.r[R.pos] = false;
        R.pos++;
        return false;
    }
    public void m4(){
        //cco.bw();
    }
}

class C implements i {
    public boolean b = true;
    public boolean c = true;
    public void m1(){
        //bcop.pw(true);
        R.r[R.pos] = true;
        R.pos++;
        //bcop.pw(false);
        R.r[R.pos] = false;
        R.pos++;
    }
    public boolean m2(){
        //bcop.pw(true);
        R.r[R.pos] = true;
        R.pos++;
        //bcop.pw(false);
        R.r[R.pos] = false;
        R.pos++;
        return true;}
}

interface i {
    boolean b = true;
    boolean c = false;
    void m1();
    boolean m2();}


class R {
  
    static boolean[] r = new boolean[51];
    static int pos = 0;
  
}
