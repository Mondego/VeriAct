 // This problem case reported by Eric.
// Simplify dies.  

import java.util.Collection;

 public class Bug1 {

  public int m (/*@ non_null @*/ Collection c) {
   Object[] a;
   a = c.toArray(a);
  }
 }
