class F {
   // bug in javac, is accepted
  void k() {
  x: 
    {
    x:
      k(); 
    }
  }
}
