
class CheckedException extends Exception{
}

public class TestCheckedException{

  /* Specs that allow checked exceptions to be throw,
     either explcitly or implicitly */

  /*@ requires true;
    @ ensures  true;
    @ signals (CheckedException) b;
    @*/
  public void may_throw_checked(boolean b)
             throws CheckedException{
    if (b) throw new CheckedException();
  }

  /*@ requires true;
    @ ensures  true;
    @*/
  public void may_throw_checked1(boolean b)
             throws CheckedException{
    if (b) throw new CheckedException();
  }

  /* The following two spec below are equivalent:

       @ requires ...;
       @ ensures  ...;
       @ signals (CheckedException) false;
       ... some_ method(...) throws CheckedException{ ... }
     
     and

       @ requires ...;
       @ ensures  ...;
       ... some_ method(...) { ... }

     In other words, omitting 
               throws CheckedException
      is equivalent to saying 
               signals (CheckedException) false;
               
     That this is so is made clear by the examples below.

   */ 

  /* Methods that do not throw CheckedException satisfy both specs: */
 
  /*@ requires true;
    @ ensures  true;
    @ signals (CheckedException) false;
    @*/
  public void may_not_throw_checked()
             throws CheckedException{
    try{ throw new CheckedException();
    } catch (CheckedException e) { }
    may_throw_checked(false);
  }

  /*@ requires true;
    @ ensures  true;
    @*/
  public void may_not_throw_checked1() {
    try{ throw new CheckedException();
    } catch (CheckedException e) { }
    may_throw_checked(false);
  }

  /* ESC/Java treats both specs as equivalent: */

  /*@ requires true;
    @ ensures  true;
    @ signals (CheckedException) false;
    @*/
  public void use_checked1()
             throws CheckedException{
     may_not_throw_checked1();
  }

  /*@ requires true;
    @ ensures  true;
    @*/
  public void use_checked() {
     may_not_throw_checked();
  }

  /*  Methods that do throw CheckedException break both specs: */

  /*@ requires true;
    @ ensures  true;
    @ signals (CheckedException) false;
    @*/
  public void may_not_throw_checked_broken()
             throws CheckedException{
    throw new CheckedException();
  }

  /* commented out because it causes error rather than warning

     requires true;
     ensures  true;
   public void may_not_throw_checked1_broken() {
    throw new CheckedException();
   }

   */


}
