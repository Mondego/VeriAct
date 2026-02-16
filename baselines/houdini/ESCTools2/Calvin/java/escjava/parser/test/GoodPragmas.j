/* Copyright Hewlett-Packard, 2002 */

/*@ nowarn; */
/*@nowarn;*/
//@ nowarn;
/*@ nowarn NullPointerException; */
/*@ nowarn IndexOutOfBoundsException, ClassCastException; */
//@ nowarn ArrayStoreException, ArithmeticException, NegativeArraySizeException, AssertViolation, PreconditionViolation;
/*@ nowarn ModifiesViolation ; */
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
//@ unreachable; assume assert + modifies;
/*@ assert assert + assume; unreachable; */
/*@ unreachable; loop_invariant assert; unreachable; */

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
/*@ uninitialized also_ensures
monitored;
monitored non_null
*/
//@ monitored monitored_by a;
/*@ monitored_by a,b
; */
//@ monitored_by monitored_by, monitored, non_null, a.b.c; uninitialized
/*@ modifies a; */
//@ requires true; modifies a[x], a[a[x]], a.b.c, a.b[q], a.b[*];
/*@ also_modifies this; */
//@ also_modifies this, this.bar ; also_modifies this[10], this.a[this];
//@ also_modifies this.a[*];

