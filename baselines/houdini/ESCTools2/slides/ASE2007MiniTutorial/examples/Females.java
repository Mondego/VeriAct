import java.util.*;
public class Females {

  /*@ requires (\forall nullable Object e; 
    @            s.contains(e);
    @            e instanceof Gendered);
    @ ensures (\forall Gendered e; 
    @            \result.contains(e);
    @            s.contains(e) 
    @            && e.gender.equals("female"));
    @*/
  public List females(List s) {
    List r = new ArrayList();
    Iterator elems = s.iterator();
    while (elems.hasNext()) {
      Gendered e = (Gendered)elems.next();
      if (e.isFemale()) {
        //@ assert e.gender.equals("female");
        r.add(e);
      }
    }
    return r;
  }
}
