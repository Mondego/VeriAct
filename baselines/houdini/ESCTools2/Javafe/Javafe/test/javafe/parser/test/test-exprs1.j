// Contains test expressions separated by commas
// Fed into javafe.parser.Parse.main

// --- Literals --------------------

345,
2345l,
3456L,
0x23AB,
0xCDEFl,
0xCDEFL,
045,

2147483647,
0x7fffffff,
017777777777,

-2147483648,
0x80000000,
020000000000,

0xffffffff,
037777777777,

9223372036854775807L,
0x7fffffffffffffffL,
0777777777777777777777L,

-9223372036854775808L,
0x8000000000000000L,
01000000000000000000000L,

0l,
0777l,
0xC0B0L,

1e1f,
2.f,
.3f,
0f,
3.14f,
6.022137e+23f,

1e1,
2.,
.3,
0.0,
3.14,
1e-9d,
1e137,

3.40282347e+38f,
1.40239846e-45f,
1.79769313486231570e+308,
4.94065645841246544e-324,

true,
false,

'a',
'%',
'\t',
'\\',
'\'',
'\u03a9',
'\uFFFF',
'\177',

"",
"\"",
"This is a string",

null,

// --- Simple examples from grammar in 19.12 of JLS ----------

x=y,
x*=y,
x/=y,
x%=y,
x+=y,
x-=y,
x<<=y,
x>>=y,
x>>>=y, 
x&=y,
x^=y,
x|=y,
x?y:z,
x||y,
x&&y,
x|y,
x^y,
x&y,
x!=y,
x==y,
x instanceof y,
x<y,
x>y,
x<=y,
x>=y,
x<<y,
x>>y,
x>>>y,
x+y,
x-y,
x*y,
x/y,
x%y,
(int)x,
(int[][][])x,
(y)x,
(y[][])x,
~x,
!x,
--x,
++x,
+x,
-x,
x++,
x--,
x[y],
x.y(),
x.y(z),
x(),
x(y),
x.y,
new int[3],
new int[3][4][5],
new int[3][][][],
new int[3][4][5][][][],
new X[3],
new X[3][4][5],
new X[3][][][],
new X[3][4][5][][][],
new X(),
new X(z),
new X(z,y),
(x),
((y))x,
((e)x),
(((w))y),

// Some more complicated tests

// FieldAccess
dis.a.b.c,

(x)

(1).a,

(x).a,

(x).a.b,
new P(x).a,
f(x).a,
a[4].a,

// MethodInvocation
dis.a.b.c(args,args,args),
(x).a.b(args,args,args),
new P(x).a(args,args,args),
f(x).a(args,args,args),
a[4].a(args,args,args),

// ArrayAccess
dis.a.b.c[dim][dim][dim],
(x).a.b[dim][dim][dim],
new P(x).a[dim][dim][dim],
f(x).a[dim][dim][dim],
a[4].a[dim][dim][dim],

// Postfix
dis.a.b.c[dim][dim][dim]++ ++ -- ++,
(x).a.b[dim][dim][dim]-- ++ ,
new P(x).a[dim][dim][dim]++ --,
f(x).a[dim][dim][dim] -- ++,
a[4].a[dim][dim][dim] ++ --,

// Java 1.1 test cases
this,
C.this,
C.D.this,
java.lang.awt.Container.this,
int.class,
float.class,
int[].class,
boolean[][].class,
short[][][][][][][][].class,
void.class,
A.class,
A.B.class,
java.lang.awt.Container[].class,

last_expression
