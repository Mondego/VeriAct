class X {

  long l0 = 2147483647;
  long l1 = 0x7fffffff;
  long l2 = 017777777777;

  long l3 = -2147483648;
  long l4 = 0x80000000;
  long l5 = 020000000000;

  long l6 = 0xffffffff;
  long l7 = 037777777777;

  long l8 = 9223372036854775807L;
  long l9 = 0x7fffffffffffffffL;
  long la = 0777777777777777777777L;

  long lb = -9223372036854775808L;
  long lc = 0x8000000000000000L;
  long ld = 01000000000000000000000L;

  long le = 0xffffffffffffffffL;
  long lf = 01777777777777777777777L;

  String s0 = "\\u05e9";
  String s3 = "\n\\u05e9";
  String s4 = "\'\\u05e9";
  char x0 = 'A';
  char x00 = '\'';
  char x1 = '\u0026';
  char x2 = '\u005c\u0027';
  char x3 = '\u0028';
  char x4 = '\n';
  String s1 = "\u064A\u0646\u0627\u064A\u0631";
  String s2 = "\n\"verbose\": verbose output";
  int a1[][] = { { 1, 2}, {3}, null };
  Class c;

  static {
    final int y = 0;
    final int u;

    do x++; while( y!=0 );
    
    while( y!=0 ) x++;
    
    for(;;) x++;
    s1 = this.s;
    s1 = C.this.s;
    s1 = String.this.toString();
    s1 = java.awt.Container.this;

    c = X.class;
    c = X[].class;
    c = X[][].class;
    c = X[][][].class;
    c = boolean.class;
    c = char.class;
    c = byte.class;
    c = short.class;
    c = int.class;
    c = long.class;
    c = float.class;
    c = double.class;
    c = void.class;
    c = boolean[].class;
    c = char[].class;
    c = byte[].class;
    c = short[].class;
    c = int[].class;
    c = long[].class;
    c = float[].class;
    c = double[].class;
    c = void[].class;
    c = java.awt.Container.class;
  }

  {
    public final class Testing {
      public void foo() { }
      public void bar() {
        public final class TestingToo extends Testing {
	}
      }
    }
    this.is.a.test();
    testing(new int[5][]);
    testing(new int[] { 1, 2, 3, 4, 5 });
    testing(new int[][] { null, {  1}, {2, 3}, {4, 5, 6} });
    Runnable r = new Runnable() { void run() { while(1); } };
    e.new I();
    e.f.g.new Runnable() { void run() { while(1); } };
    e().new Runnable() { }.new Thread(a, b, c);
    e().new Runnable() { }.new Thread(a, b, c) { void run() { while(1); } };
  }

  final int x = 10;
  final int x;

  public X(T a) {
    a.super();
  }

  public X(S a) {
    a.b.c[10].super();
  }

  public X() {
    new Runnable() { void run() { super.run(); } }.super();
  }

    public X() {
      super(testing);
    }

    public X(int a) { }
}
