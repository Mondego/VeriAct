
class UncheckedException extends RuntimeException{
}

public class TestUncheckedException{

  /* Specs that allow checked exceptions to be throw,
     either explcitly or implicitly */

  /*@ requires true;
    @ ensures  true;
    @ signals (UncheckedException) b;
    @*/
  public int may_throw_unchecked(boolean b)
             throws UncheckedException{
    if (b) throw new UncheckedException();
  }

  /*@ requires true;
    @ ensures  true;
    @*/
  public int may_throw_unchecked1(boolean b)
             throws UncheckedException{
    if (b) throw new UncheckedException();
  }


  /* The following two spec below are equivalent:

       @ requires ...;
       @ ensures  ...;
       @ signals (UncheckedException) false;
       ... some_ method(...) throws UncheckedException{ ... }
     
     and

       @ requires ...;
       @ ensures  ...;
       ... some_ method(...) { ... }

     In other words, omitting 
               throws UncheckedException
      is equivalent to saying 
               signals (UncheckedException) false;
               
     That this is so is made clear by the examples below.

   */ 

  /* Methods that do not throw UncheckedException satisfy both specs: */
 
  /*@ requires true;
    @ ensures  true;
    @ signals (UncheckedException) false;
    @*/
  public int may_not_throw_unchecked()
             throws UncheckedException{
    try{ throw new UncheckedException();
    } catch (UncheckedException e) { }
    may_throw_unchecked(false);
  }

  /*@ requires true;
    @ ensures  true;
    @*/
  public int may_not_throw_unchecked1() {
    try{ throw new UncheckedException();
    } catch (UncheckedException e) { }
    may_throw_unchecked(false);
  }

  /* ESC/Java treats both specs as equivalent: */

  /*@ requires true;
    @ ensures  true;
    @ signals (UncheckedException) false;
    @*/
  public int use_unchecked1()
             throws UncheckedException{
     may_not_throw_unchecked1();
  }

  /*@ requires true;
    @ ensures  true;
    @*/
  public int use_unchecked() {
     may_not_throw_unchecked();
  }

  /*  Methods that do throw UncheckedException break both specs: */

  /*@ requires true;
    @ ensures  true;
    @ signals (UncheckedException) false;
    @*/
  public int may_not_throw_unchecked_broken()
             throws UncheckedException{
    throw new UncheckedException();
  }


  /*@ requires true;
    @ ensures  true;
    @*/
  public int may_not_throw_unchecked1_broken() {
    throw new UncheckedException();
   }



}
