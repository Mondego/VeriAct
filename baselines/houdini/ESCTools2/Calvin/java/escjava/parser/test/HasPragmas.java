/* Copyright Hewlett-Packard, 2002 */

//@ nowarn;
//@ nowarn Check1, Check2;
//@ nowarn Check1, Check_2, Check_3, check4, and, so, on;

/*@ readable_if 1 == 1 */
/*@ writable_if 1 == 1 */
public abstract
class HasPragmas
{
   /*@ invariant (i + 10) == 11 */
   /*@ invariant (i + 10) == 11 */

   int a /* writable_deferred */;

   int b /* writable_deferred readable_if a != 0 */;
   int c /* readable_if a != 0; writable_deferred */;

   int d /* writable_deferred writable_if a != 0 */;
   int e /* writable_if a != 0; writable_deferred */;

   public abstract int test1()
    /*@ requires true; ensures \result == 1 */;
}
    
