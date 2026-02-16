/* Copyright Hewlett-Packard, 2002 */

class Good {

  //@ ghost public /*@ non_null // comment */ Object o;
  
  Good () {
    //@ set o = this;
  }

  /*@ requires a > 0;                 // comment 1 
      requires b > 0;  //@ nowarn Pre // comment 2 
      requires c > 0;                 // comment 3 */ 
  void m(int a, int b, int c) { }

}
