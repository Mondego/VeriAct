//@ ensures idn;
//@ ensures this;
//@ ensures 1;
//@ ensures 1.0;
//@ ensures 1.0d;
//@ ensures "";
//@ ensures "literals";
//@ ensures 'a';
//@ ensures true;
//@ ensures false;
//@ ensures null;
//@ ensures compound.name;
//@ ensures l.o.n.g._.c.o.m.p.o.u.n.d._.n.a.m.e;

//@ ensures a ==> (b ==> c);
//@ ensures \type(Integer) <: \type(java.lang.Number) <: \type(Object);

//@ ensures \fresh(a);
//@ ensures \elemtype(b);
//@ ensures \max(1);
//@ ensures \old(x + y);
//@ ensures \typeof(this);

//@ ensures \type(int);
//@ ensures \type(float[]);
//@ ensures \type(boolean[][]);
//@ ensures \type(java.lang.Object);

//@ ensures (\lblpos label1 hi);
//@ ensures (\lblneg label1 there);

//@ ensures (\forall int a; true);
//@ ensures (\forall float[] t1, t2; t1 == t2);
//@ ensures (\forall java.lang.String s1, s2, s3; false);
//@ ensures (\exists long a; a == 1);
//@ ensures (\exists long[] a,b,c; true);

//@ ensures \java_math(0);
//@ ensures \safe_math(0);
//@ ensures \bigint_math(0);
