class Ghost98 {
  //@ ghost public Ghost98 link;
  //@ghost public boolean b;

 //@invariant b && (b == true) && (this == link);

Ghost98() {
  //@set link= this; 
  //@set b= true;
  if (this.m()) {
    //@set link= this;
  }
  else {
    //@set link= null;
}
}

//@ requires true;
//@ ensures \result == this.b
boolean m()
{ }  //@nowarn Post



}
