
public class loop {
  int m1_while_ok() {
    while (false) {}
  }
  int m2_while_fail() {
    while (false) {}
    //@ assert false;
  }
  int m3_while_ok() {
    while (true) {}
    //@ assert false;
  }
  
  int m1_do_ok() {
    do {} while(false);
  }
  int m2_do_fail() {
    do {} while(false);
    //@ assert false;
  }
  int m3_do_ok() {
    do {} while (true);
    //@ assert false;
  }
  
  
  int m1_for_ok() {
    for(;false;) {}
  }
  int m2_for_fail() {
    for(;false;) {}
    //@ assert false;
  }
  int m3_for_ok() {
    for(;true;) {}
    //@ assert false;
  }
  
}
