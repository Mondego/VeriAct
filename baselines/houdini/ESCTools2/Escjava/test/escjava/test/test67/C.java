class C {
  /** The following has been syntactically-legal ESC/Java pragmas
    * since version 1.2.2.
    **/

  /*@ non_null  */ Object a;
  //@ invariant a instanceof C  ;
  /*@(comment goes here) invariant a instanceof int[] ==>
                 ((int[])a).length == 5; */

  /*@@   requires p != null;  ensures
       \result != null */
  /**     <esc>modifies a;</esc>
<esc>   ensures \result instanceof Object[] ==>
                \nonnullelements((Object[])\result);</esc>
    */
  /*  <esc> gibberish!! </esc> ****/
  Object m(Integer p) {
    return p;
  }

  /** The following have become syntactically-legal ESC/Java pragmas
    * since the JML changes after version 1.2.3.
    **/

  /*@ non_null @*/ Object b;
  //@ invariant b instanceof C  ;
  /*@ invariant
    @   b instanceof int[] ==>
    @   ((int[])b).length == 5;
    @*/
  /*@@ invariant
    @@   b instanceof double[] ==>
    @@   ((double[])b).length == 5;
    @@*/
  
  /*@@   requires p != null;  ensures
     @  \result == \result; */
  /**     <esc>modifies a;</esc>
<esc>   ensures \result instanceof Object[] ==>
  *             \nonnullelements((Object[])\result);</esc>
    */
  /*  <esc> gibberish!! </esc> ****/
  Object n(Integer p) {
    return new Integer(p.intValue());
  }

  /*@
    @  requires x != null;
    @*/
  void p(Object x) {
    //@ assert x != null;
  }

  /*@
      requires x != null;
    @*/
  void q(Object x) {
    //@ assert x != null;
  }

  void r(/*@@@@@@@@ non_null @@@@@*/ Object x) {
  }
}
