/**
 ** Test that static final fields are initialized
 **/

interface A
{
  int a = 0; //OK
  int a1;  //ERROR
}

interface B extends A
{
  int b = 0; //OK
  int b1;  //ERROR
}

class C
{
  public static final int c = 0;  //OK
  public static final int c1;  //ERROR

  class D
  {
    public static final int d = 0;  //OK
    public static final int d1;  //ERROR
  }
}

class E extends C{
  public static final int e = 0;  //OK
  public static final int e1;  //ERROR
}

