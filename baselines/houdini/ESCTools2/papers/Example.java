//@ model import java.util.List;
//@ model import java.util.ArrayList;
public class Example {
  /*@ spec_public */ private Object[] seq; 
                        //@ in list;
                        //@ maps seq[*] \into list;
  //@ invariant seq != null && \elemtype(\typeof(seq)) == \type(Object);

  //@ requires out != null;
  //@ modifies out[*];
  //@ signals (NullPointerException) false;
  //@ ensures seq.length > 0 ==> out[0] == seq[seq.length-1];
  public void reverse(Object[] out) {
    int i = 0;
    int j = seq.length;
    while (i < seq.length) out[i++] = seq[--j];
  }

  //@ public model List list;
  //@ private represents list <- toList(seq);

  /*@ requires input != null;
    @ ensures \result != null;
    @ pure
    @ private model List toList(Object[] input) {
    @   List list = new ArrayList(input.length);
    @   for (int i=0; i<input.length; ++i) list.add(input[i]);
    @   return list;
    @ }
    @*/

  //@ requires i >= 0 && i < length();
  //@ modifies list;
  public void insert(int i, Object o) { seq[i] = o; }

  //@ private normal_behavior
  //@   ensures \result == seq.length;
  //@ pure
  public int length() { return seq.length; }

}
