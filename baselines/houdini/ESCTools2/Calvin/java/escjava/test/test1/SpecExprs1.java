/* Copyright Hewlett-Packard, 2002 */


// Tests for resolution and typechecking of pragmas

public abstract class SpecExprs1 {

  public SpecExprs1[][] a1;

  /*@ invariant typeof(a1) <: type(java.lang.Object[][]) */
  /*@ invariant elemType(typeof(a1)) <: type(java.lang.Object[]) */
  /*@ invariant elemType(typeof(a1)) == type(Modifiers1[]) */
  /*@ invariant (forall int i,j; 0 <= i && i < a1.length  ) */
  /*@ invariant (forall int i,j; (0 <= i && i < a1.length)
                               & (0 <= j && j < a1[i].length) ==>
                   a1[i][j] instanceof SpecExprs1) */

  public abstract void m1()
    /*@ modifies this.a1[*] */
    ;

  public abstract void m2(Object a1)
    /*@ ensures fresh(a1) & fresh(this.a1) */
    /*@ ensures PRE(this.a1).length == this.a1.length */
    /*@ ensures (forall int i; (i <= 0 & i < this.a1.length)
                  ==> this.a1[i] == PRE(this.a1)[i]) */
    ;

  public abstract void m3()
    /*@ requires LS[this.a1] */
    ;

  public abstract void m4(Object o)
    /*@ requires o < min(LS) */
    ;

  public abstract int[] permute(int[] a1, int[] a2)
    /*@ requires (forall int i,j; 
                   (0 <= i && i < a1.length) & (0 <= j && j < a2.length)
                   & a1[i] == a2[j] ==> i == j); */
    /*@ ensures fresh(RES) */
    /*@ ensures RES.length == (LBLPOS label1 a1.length) */
    /*@ ensures (forall int i; (LBLNEG label2 (0 <= i && i < a1.length)) ==>
                  (exists int j; (0 <= j && j < a2.length) 
		                     ==> a1[i] == a2[j])); */
    ;

  public abstract void m5(SpecExprs1 other, int j)
    /*@ modifies other.a1, a1[j] */
    ;

  public abstract void m6(SpecExprs1 other, int j)
    /*@ requires (0 <= j && j < a1.length) */
    /*@ modifies other.a1[*][*], this.a1[j][*] */
    ;

  public abstract void m7(SpecExprs1 other, int j)
    /*@ requires (forall int i; (0 <= i && i < a1.length) 
                                 ==> (0 <= j && j < a1[i].length)) */
    /*@ modifies other.a1[*], a1[*][j] */
    ;
}
