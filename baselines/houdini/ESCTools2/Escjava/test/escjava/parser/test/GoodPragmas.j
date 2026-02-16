/*@ nowarn */
/*@nowarn*/
//@ nowarn
/*@ nowarn NullPointerException; */
/*@ nowarn IndexOutOfBoundsException, ClassCastException; */
//@ nowarn ArrayStoreException, ArithmeticException, NegativeArraySizeException, AssertViolation, PreconditionViolation;
/*@ nowarn ModifiesViolation; */
//@nowarn ObjectInvariantViolation, InitialLoopInvariantViolation;
/*@nowarn IterativeLoopInvariantViolation,SharingViolation, LockingOrderViolation,InitializationViolation,DefinednessViolation; */
/*@nowarn
CodeReachabilityViolation,
ConstructorLeakWarning,
InitializerLeakWarning,WritableDeferredWarning,
UnenforcableObjectInvariantWarning,ModifiesExtensionWarning
;*/

/*@unreachable;*/
//@ unreachable;
/*@ assume false == false; */
//@ assume true;
//@ unreachable; assume assertt + modifies;
/*@ assert assertt + assume; unreachable; */
/*@ unreachable; loop_invariant assertt; unreachable; */

//@ still_deferred x;
//@still_deferred x;
/*@still_deferred
x,y;*/
//@    still_deferred axiom, invariant, still_deferred;
/*@still_deferred
x
,y,
z

,a


;*/
//@ axiom invariant;
//@ axiom invariant;
/*@ invariant false
== true ; */

/*@ uninitialized monitored non_null writable_deferred */
//@ readable_if uninitialized; requires true; ensures false;
/*@ uninitialized ensures
monitored;
monitored non_null
*/
//@ monitored monitored_by a;
/*@ monitored_by a,b
; */
//@ monitored_by monitored_by, monitored, non_null, a.b.c; uninitialized
/*@ modifies a; */
//@ requires true; modifies a[x], a[a[x]], a.b.c, a.b[q], a.b[*];
/*@ modifies this; */
//@ modifies this, this.bar ; modifies this[10], this.a[this];
//@ modifies this.a[*];
