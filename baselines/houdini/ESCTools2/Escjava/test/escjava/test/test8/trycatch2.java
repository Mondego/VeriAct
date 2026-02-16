class Try2 {
  //@ requires t != null;
  void m1(Throwable t) throws Throwable {
    int x,y;
    try {
      x=0;
      //@ assume \typeof(t) == \type(Throwable);
      throw t;
    } catch(RuntimeException t3) {
      x=3;
    }

    //@ assert x==0;
  }
}
