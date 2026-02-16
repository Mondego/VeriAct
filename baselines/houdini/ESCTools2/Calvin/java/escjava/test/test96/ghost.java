/* Copyright Hewlett-Packard, 2002 */

class Ghost21 {
  //@ ghost public int i;
  //@ghost public boolean b;

 //@invariant b && (b == true) && (i > 0);

Ghost21() {
  //@set i= 6; 
  //@set b= true;
}

void m() {
//@ assert (i > -1) && b;

}
}
