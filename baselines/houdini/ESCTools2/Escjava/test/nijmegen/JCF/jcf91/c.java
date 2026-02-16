/*
 Adaptation of Bergstra & Loots's Java Class Family number 91:

 The results are stored in static integers, not printed.

 Monday 28 June 99 15:16:11 bart@irritatie.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf91;

class c {

    static int result1, result2, result3, result4, result5, result6,
            result7, result8, result9, result10, result11, result12,
            result13, result14, result15, result16, result17, result18,
            result19, result20, result21, result22, result23;

    static void m(){
        U u = new U();
        result1 = u.b;
        result2 = u.c;
        i p = u;
        result3 = p==u?-2:-1;
        result4 = u==p?-2:-1;
        result5 = u.b;
        result6 = u.c;
        result7 = p.b;
        result8 = p.c;
        result9 = ((U) p).b;
        result10 = ((U) p).c;
        j q = p;
        result11 = u.b;
        result12 = u.c;
        result13 = q.c;
        result14 = p==q?-2:-1;
        result15 = q==p?-2:-1;
        result16 = u==q?-2:-1;
        result17 = q==u?-2:-1;
        //cco.a();
        V v = new V();
        result18 = v.b;
        result19 = v.c;
        i pp = v;
        result20 = v.b;
        result21 = v.c;
        result22 = pp.b;
        result23 = pp.c;
        j qq = pp;
    }
}

class U implements i,j {int b;int c;}
class V implements i,j {int b = 12;int c =  8;}

interface i extends j {int b = 37;}
interface j {int c = 45;}

/*
 class Object {

 boolean equals(Object obj) {
 return this == obj;
 }

 }
 */

