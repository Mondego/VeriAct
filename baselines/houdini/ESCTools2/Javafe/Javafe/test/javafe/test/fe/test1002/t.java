import java.io.IOException;

class C {

  void m() {

    Object[] oa = null;
    Object o = oa;

    switch( 'c' ) {
    case 'c':
    case D.x:
    }

    char asdf = ''';

    char c = 'c';

    switch( c ) {
    case '\u0BF1':
    }

  }

  static int n() throws IOException {
    int i;
    i = D.x;
    throw new MyException2();
    return i;
  }
}

class D {
  public static final int w=0;
  public static final int x=w+1;
}

class MyException extends IOException {
}

class MyException2 extends MyException {
}

