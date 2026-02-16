
// Tests for resolution and typechecking of pragmas

/*
Commented out: pragmas outside of a class are no longer allowed.
<pre><esc>
axiom 1 < 2;
invariant v2 < 100;
</esc></pre>
*/

public abstract class TypeDeclElemPragmas1 extends Modifiers1 {

  /*@ still_deferred v2; */

  /*@ axiom 0 < 1 */

  /*@ invariant 10 < v2 */

  static class Inner { }
}

