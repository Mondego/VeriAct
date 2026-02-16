
public class C {
  int f;
  static int gg;

  //@ ensures f == fx;
  C(int fx) {
    f = fx;
  }
  
  // test relative assignments to local variables
  void test0() {
    int loc= 0;
    int g= loc++;
    //@ assert loc == 1; 
    //@ assert g == 0;
    g= ++loc;
    //@ assert loc == 2; 
    //@ assert g == 2;
    loc+= 1;
    //@ assert loc == 3;
    loc--;
    //@ assert loc == 2;
    loc*= 2;
    //@ assert loc == 2 * 2;
  }

  // test relative assignments to fields
  void test1() {
    f= 0;
    int g= f++;
    //@ assert this.f == 1; 
    //@ assert g == 0;
    g= ++f;
    //@ assert this.f == 2; 
    //@ assert g == 2;
    f+= 1;
    //@ assert this.f == 3;
    f--;
    //@ assert f == 2;
    f*= 2;
    //@ assert f == 2 * 2;
  }

  // test relative assignments to static fields
  void test2() {
    gg= 0;
    int g= gg++;
    //@ assert this.gg == 1; 
    //@ assert g == 0;
    g= ++gg;
    //@ assert C.gg == 2 ;
    //@ assert g == 2;
    gg+= 1;
    //@ assert this.gg == 3;
    gg--;
    //@ assert gg == 2;
    gg*= 2;
    //@ assert gg == 2 * 2;
  }

  // test relative assignments to fields
  void test3() {
    int[] a = new int[10];
    int i = 6;
    
    int g= a[i]++;
    //@ assert a[i] == 1 ;
    //@ assert g == 0;
    g= ++a[i];
    //@ assert a[i] == 2 ;
    //@ assert g == 2;
    a[i]+= 1;
    //@ assert a[i] == 3;
    a[i]--;
    //@ assert a[i] == 2;
    a[i]*= 2;
    //@ assert a[i] == 2 * 2;
  }

  // some tests using String
  void test4() {
    String s0 = "test this!";
    String s1 = "yo" + " dude!";
    String s = s0 + s1;
    s += s0;
    s += null;
    s += 5;
    s += (float)0;
    char c = 'A';
    s += c;
    int i = 17;
    s += i;
    //@ assert s != null;
  }

  // test side effects in LHS
  void test5() {
    int x = 0;
    C c;
    
    (c = new C(x++)).f = 6;
    //@ assert x == 1;
    //@ assert c != null;
    //@ assert c.f == 6;

    (c = new C(x++)).f += 4;
    //@ assert x == 2;
    //@ assert c.f == 5;

    (2 < 5 ? c = new C(x++) : c = new C(x += 10)).f *= 2;
    //@ assert x == 3;
    //@ assert c.f == 2 * 2;

    c.f = 4;
    (c.f++ == 4 ? c : null).f -= 5;
    //@ assert c.f == 0;
  }
  
  // test side effects in LHS
  void test6() {
    int x = 0;
    gg = 0;
    C c;
    
    (c = new C(x++)).gg = 6;
    //@ assert x == 1;
    //@ assert c != null;
    //@ assert gg == 6;

    (c = new C(x++)).gg += 4;
    //@ assert x == 2;
    //@ assert gg == 10;

    (2 < 5 ? c = new C(x++) : c = new C(x += 10)).gg *= 2;
    //@ assert x == 3;
    //@ assert gg == 10 * 2;

    gg = 4;
    (gg++ == 4 ? (C)null : (C)null).gg -= 5;
    //@ assert gg == 0;
  }
  
  // test side effects in LHS
  void test7() {
    int x = 0;
    int[] a = new int[10];
    C c;

    a[x++] += 2;
    //@ assert x == 1;
    //@ assert a[0] == 2;
    //@ assert a[1] == 0;

    a[++x]++;
    //@ assert x == 2;
    //@ assert a[1] == 0;
    //@ assert a[2] == 1;

    a[5] = 17;
    (a[x += 3] == 17 ? a : new int[5])[x++] -= 7;
    //@ assert x == 6;
    //@ assert a[5] == 10;
    //@ assert (\forall int i; 6 <= i && i < a.length ==> a[i] == 0);

    a[2] = 4;
    (a[2]++ == 4 ? a : null)[2] -= 5;
    //@ assert a[2] == 0;
  }
  
  // test casts in relative assignment
  void cast0(byte i, int[] a) {
    if (a != null && 0 <= i) {
      i++;             // this introduces a cast
      if (i < a.length) {
	int x = a[i];  // "i" may be negative
      }
    }
  }

  // test casts in relative assignment
  void cast1(int i, int[] a) {
    if (a != null && 0 <= i && i != 127) {
      i++;             // cast introduced
      if (i < a.length) {
	int x = a[i];  // okay
      }
    }
  }

  // test casts in relative assignment
  void cast2(int i, int[] a) {
    if (a != null && 0 <= i) {
      i++;             // no cast introduced
      if (i < a.length) {
	int x = a[i];  // okay
      }
    }
  }
}
