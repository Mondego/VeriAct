(* Esc Java  - Coq extension  Sorted logic. *)


Require Import Bool.
Require Import ZArith.
Require Import Zdiv.
Require Import Zbool.
Require Import Sumbool.
Require Import Reals.
Require Import Classical.
Require Import String.
Open Scope Z_scope.

Load "defs_types.v".
Module EscJavaTypesDef <: EscJavaTypes.
Definition Boolean:= bool.
Lemma Boolean_dec: forall x y : Boolean, {x = y}+{x <> y}.
Proof with auto.
    intros; destruct x; destruct y...
    right; intro; discriminate.
Qed.

Definition java_lang_Boolean_TRUE:= true.
Definition java_lang_Boolean_FALSE:= false.

Definition time := Z.
Definition time_dec (x: time) (y: time) := (Z_eq_dec x y).
Module Time <: Arith.
     Definition t := time.
    Definition GE := fun (x y : Z) => x >= y.
    Definition GT := fun (x y : Z) =>  x > y.
    Definition LE  := fun (x y : Z) => x <= y.
    Definition LT  := fun (x y : Z) => x < y.
    Definition Div := 
        fun (x y : Z) =>(x  / y).
    Definition Add  := 
        fun (x y : Z) =>(x  + y).
    Definition Sub := 
        fun (x y : Z) =>(x - y).
    Definition Mul := 
        fun (x y : Z) =>(x  * y).
    Definition Neg := 
        fun (x : Z) =>(0 - x).
End Time.


Definition DiscreteNumber := Z.
Definition DiscreteNumber_dec (x: DiscreteNumber) (y: DiscreteNumber) := (Z_eq_dec x y).
Module Discrete <: Arith.
    Definition t := DiscreteNumber.

    Definition GE : DiscreteNumber -> DiscreteNumber -> Prop := fun (x y : Z) => x >= y.
    Definition GT : DiscreteNumber -> DiscreteNumber -> Prop := fun (x y : Z) =>  x > y.
    Definition LE : DiscreteNumber-> DiscreteNumber -> Prop := fun (x y : Z) => x <= y.
    Definition LT : DiscreteNumber -> DiscreteNumber -> Prop := fun (x y : Z) => x < y.

    Definition Div :  DiscreteNumber -> DiscreteNumber -> DiscreteNumber := 
        fun (x y : Z) =>(x  / y).
    Definition Add : DiscreteNumber -> DiscreteNumber -> DiscreteNumber := 
        fun (x y : Z) =>(x  + y).
    Definition Sub : DiscreteNumber -> DiscreteNumber -> DiscreteNumber := 
        fun (x y : Z) =>(x - y).
    Definition Mul : DiscreteNumber -> DiscreteNumber -> DiscreteNumber := 
        fun (x y : Z) =>(x  * y).
    Definition Neg : DiscreteNumber -> DiscreteNumber := 
        fun (x : Z) =>(0 - x).
End Discrete.

Module Discrete_bool <: Arith_bool.
    Definition t := DiscreteNumber.
    Module decType <: DecType.
        Definition t := DiscreteNumber.
        Definition t_dec := DiscreteNumber_dec.
    End decType.

    Module dec := Dec decType.
    Definition EQ : DiscreteNumber -> DiscreteNumber -> bool := dec.EQ.
    Definition GE : DiscreteNumber -> DiscreteNumber -> bool := fun (x y : Z) => Zge_bool x y.
    Definition GT : DiscreteNumber -> DiscreteNumber -> bool := fun (x y : Z) => Zgt_bool x y.
    Definition LE : DiscreteNumber -> DiscreteNumber -> bool := fun (x y : Z) =>  Zle_bool x y.
    Definition LT : DiscreteNumber -> DiscreteNumber -> bool := fun (x y : Z) => Zlt_bool x y.

    Definition EQ_refl:= dec.EQ_refl.

    Definition EQ_eq := dec.EQ_eq.
    Definition EQ_true := dec.EQ_true.
    Definition EQ_false := dec.EQ_false.
    Definition EQ_eq_not_false := dec.EQ_eq_not_false.
    Definition EQ_exists_eq := dec.EQ_exists_eq.
    Definition EQ_false_not_eq := dec.EQ_false_not_eq.

End Discrete_bool.
Definition dzero: DiscreteNumber := 0.

(* TODO: treat ContinuousNumbers, Real,...*)
Parameter ContinuousNumber : Set.
Variable czero : ContinuousNumber.
Parameter RealNumber : Set.
Variable rzero: RealNumber.
Parameter BigIntNumber : Set.
Variable bzero: BigIntNumber.

(* Java numbers are Continuous or Discrete *)
Inductive javaNumber: Set := 
| java_discrete: DiscreteNumber -> javaNumber
| java_continuous: ContinuousNumber -> javaNumber.
Definition JavaNumber := javaNumber.
Definition JavaNumber_to_DiscreteNumber (n: JavaNumber): DiscreteNumber :=
   match n with
   | java_discrete d => d
   | _ => dzero
   end.
Definition JavaNumber_to_ContinuousNumber (n: JavaNumber) : ContinuousNumber :=
   match n with
   | java_continuous c => c
   | _ => czero
   end.

(* JMLNumber are JavaNumber, RealNumber, BigIntNumber *)
Inductive jMLNumber: Set :=
| jml_java : JavaNumber -> jMLNumber
| jml_real : RealNumber -> jMLNumber
| jml_bigint: BigIntNumber -> jMLNumber.
Definition JMLNumber:= jMLNumber.
Parameter JMLNumber_to_JavaNumber: JMLNumber -> JavaNumber.
Parameter JMLNumber_to_RealNumber: JMLNumber -> RealNumber.
Parameter JMLNumber_to_BigIntNumber: JMLNumber -> BigIntNumber.

(* Numbers are JMLNumbers... don't see the point *)
Definition Number:= JMLNumber.
Definition Number_to_JMLNumber: Number -> JMLNumber :=
   fun v => v.

(* BaseType can be converted to boolean or to JavaNumber *)
Parameter BaseType: Set.
Parameter BaseType_to_Boolean: BaseType -> Boolean.
Parameter BaseType_to_JavaNumber: BaseType -> JavaNumber.
Variable T_byte: BaseType.
Variable T_char: BaseType.
Variable T_short: BaseType.
Variable T_int: BaseType.
Variable T_long: BaseType.
Variable T_float: BaseType.
Variable T_double: BaseType.


Definition RefType := Z.
Definition RefType_dec (x: RefType) (y: RefType) := (Z_eq_dec x y).

Inductive reference: Set :=
| var_ref: string -> time -> reference
| field_ref: reference -> time -> reference
| elem_ref: reference -> time  -> reference.

Definition Reference := reference.

Definition Reference_dec: forall x y : Reference, {x = y} + {x <> y}.
Proof with auto.
  induction x; 
  induction y;
  intros; try ( right; intro h; discriminate);
  try (
  assert( h:= string_dec s s0); destruct h;
  [assert( h1:= Z_dec t t0); destruct h1; subst; auto;
  destruct s1;
  right;
  intro h; injection h; intros; subst; omega |
  right; intro h; injection h; intros; subst; destruct n; auto ]).

assert (h:= IHx y);
destruct h as [h1 | h2]; subst;
assert( h1:= Z_dec t t0); [
destruct h1 as [s| s]; subst; auto; destruct s; 
right; intro h2; injection h2; intros; subst; auto; omega |
right; intro h; injection h; intros; subst; destruct h2; auto].

assert (h:= IHx y);
destruct h as [h1 | h2]; subst;
assert( h1:= Z_dec t t0); [
destruct h1 as [s| s]; subst; auto; destruct s; 
right; intro h2; injection h2; intros; subst; auto; omega |
right; intro h; injection h; intros; subst; destruct h2; auto].
Qed.
Definition null : Reference :=  var_ref EmptyString 0.


Variable java_lang_Object: RefType.
Variable java_lang_reflect_Array: RefType.
Variable java_lang_Cloneable: RefType.
Module Types.
    Parameter subtype: RefType -> RefType-> Prop.
    Parameter extends: RefType -> RefType -> Prop.
    Parameter typeof : Reference -> RefType.
    Parameter elemRefType: RefType -> RefType.
    Parameter isa: Reference -> RefType -> Prop.
    Axiom subtypes_includes_extends:
          forall (t u: RefType), extends t u -> subtype t u.
    Axiom subtype_is_the_smallest_relation_that_contains_extends:
          forall (t u: RefType), (subtype t u /\ (t <> u)) ->
               exists v: RefType, (extends t v /\ subtype v u).
    Axiom java_lang_Object_is_Top:
         forall (t: RefType), subtype t java_lang_Object.
    Axiom unique_dynamic_subtype:
         forall (r: Reference) (t: RefType),
              isa r t <-> (r = null) \/ (typeof r = t).
    Parameter instantiable: RefType -> Prop.
    Axiom instantiable_def:
          forall (t: RefType), subtype t java_lang_Object -> instantiable t.
End Types.


Definition LockMap := Z.
Variable LS: LockMap.
Variable maxLockSet: Reference.
Definition LockMap_dec : 
           forall x y : LockMap, {x = y} + {x <> y} := Z_eq_dec.
Module Lock.
    Parameter less : Reference -> Reference -> Prop.
    Parameter lock: LockMap -> Reference -> LockMap.
    Parameter release : LockMap -> Reference -> LockMap.
    Parameter select: LockMap -> Reference -> Prop.
    Axiom access_definition1: 
           forall (r: Reference), not (select (lock LS r) r).
    Axiom access_definition2:
           forall (r: Reference), select (release (lock LS r) r) r.
    Axiom ls_has_a_maximum_element:
           forall (r: Reference), (less r maxLockSet).
    Axiom null_belongs_to_ls:
           forall (r: Reference), less null r.
End Lock.


Definition Elem := Reference.
Definition Field := Reference.


Definition Path := string.
Definition Path_dec (x: Path) (y: Path) := (string_dec x y).

Definition zero: DiscreteNumber := 0%Z.



Open Scope string_scope.
Definition ecReturn : Path := "ecReturn".
Definition ecThrow : Path := "ecThrow".
Definition allocNew_ : Reference := var_ref "allocNew" 0.
Definition alloc_ : Reference := var_ref "alloc" 0.
Close Scope string_scope.

Inductive S: Set  := 
    |   bool_to_S: bool -> S
    |   DiscreteNumber_to_S : DiscreteNumber -> S
    |   Reference_to_S: Reference -> S
    |   Path_to_S: Path -> S
    |   Time_to_S: time -> S
    |   RefType_to_S: RefType -> S
    |    bottom: S.

Coercion bool_to_S : bool >-> S.
Coercion DiscreteNumber_to_S : DiscreteNumber >-> S.
Coercion Reference_to_S : Reference >-> S.
Coercion Time_to_S : time >-> S.
Coercion Path_to_S : Path >-> S.
Coercion RefType_to_S : RefType >-> S.

Definition S_to_bool (s:S): bool :=
match s with 
| bool_to_S b => b
| _ => false
end.
Coercion S_to_bool : S >-> bool.

Definition S_to_DiscreteNumber (s:S): DiscreteNumber :=
match s with 
| DiscreteNumber_to_S b => b
| _ => 0%Z
end.
Coercion S_to_DiscreteNumber : S >-> DiscreteNumber.


Definition S_to_Reference (s:S): Reference :=
match s with 
| Reference_to_S b => b
| _ => null
end.
Coercion S_to_Reference : S >-> Reference.

Definition emptyPath : Path :=  EmptyString.
Definition S_to_Path (s:S): Path :=
match s with 
| Path_to_S b => b
| _ => emptyPath
end.
Coercion S_to_Path : S >-> Path.

Definition S_to_Time (s:S): time :=
match s with 
| Time_to_S b => b
| _ => 0
end.
Coercion S_to_Time : S >-> time.
Definition S_to_RefType (s:S): RefType :=
match s with 
| RefType_to_S b => b
| _ => 0
end.
Coercion S_to_RefType : S >-> RefType.

Definition AnySet := S.

Definition cast : Reference -> RefType -> Reference := 
 fun (v:Reference)  (t: RefType) => v.




Definition distinctAxiom : ecReturn <> ecThrow.
Proof.
   intro.
   discriminate.
Qed.
Hint Immediate distinctAxiom.



(*Variable is : S -> S -> Prop. *)
Definition isField (r: Reference ) : Prop :=
     match r with 
     | field_ref _ _ => True
     | _ => False
     end.


Variable asField: Field -> RefType -> Prop.
Variable asElems: Elem -> Elem.
Variable asLockSet: LockMap -> LockMap.

Definition eClosedTime (e: Elem) : time :=
   match e with 
   | elem_ref _ t => t
   | _ => zero
   end.

Definition fClosedTime (f: Field) : time :=
   match f with 
   | field_ref _ t => t
   | _ => zero
   end.

Fixpoint vAllocTime (r : Reference) {struct r}: time :=
    match r with
    | var_ref n t => t
    | elem_ref r' _ => vAllocTime r'
    | field_ref r' _ => vAllocTime r'
    end.

Definition isAllocated: Reference -> time -> Prop := 
    fun (obj : Reference) (t: time) => vAllocTime obj < t.
Ltac unfoldEscTime := unfold isAllocated.


(* Array Logic *)
Module array.
Variable shapeOne: DiscreteNumber -> Reference.
Variable shapeMore: DiscreteNumber-> Reference -> Reference.
Variable isNew: Reference -> Prop.
Variable length: Reference -> DiscreteNumber.
Axiom lengthAx :
      forall (a : Reference), (Discrete.LE 0 (length a)).
Variable select: Reference -> DiscreteNumber -> S.
Variable store: Reference -> DiscreteNumber -> S -> Reference.
Axiom select_store1: 
    forall(var :Reference) (obj : DiscreteNumber)(val :AnySet), (select (store var obj val) obj) = val.
Axiom select_store2: 
    forall(var : Reference)(obj1 obj2 :DiscreteNumber) (val :  AnySet), 
         obj1 <> obj2 -> 
                 (select (store var obj1 val) obj2) = (select var obj2).

Variable fresh : Reference -> time -> time -> Reference -> Reference -> RefType ->  AnySet (* A *) -> Prop.
    (* array axioms2 *)
    Axiom axiom2 : 
      forall (array: Reference) ( a0:time) (b0:time) (obj : Reference) (n : DiscreteNumber)  (T: RefType)  (v :  AnySet),
        ((fresh array a0 b0 obj (shapeOne n)  T v) ->
         (Time.LE a0 (vAllocTime array))) /\
        ((fresh array  a0  b0 obj (shapeOne n) T v) ->
         (isAllocated array  b0)) /\
        ((fresh array  a0  b0 obj (shapeOne n) T v) ->
         (array <> null)) /\
        ((fresh array  a0  b0 obj (shapeOne n) T v) ->
         (Types.typeof array) = T) /\
        ((fresh array  a0  b0 obj (shapeOne n) T v) ->
         (length array) = n) (* /\
        ((arrayFresh a  a0  b0 e (arrayShapeOne n) T v) ->
         forall (i : Reference),
           (select (select e a) i) = v) *).
End array.


(* The operations on the heap - more or less *)
Inductive java_heap : Set := 
| heap_store: java_heap -> Reference ->  S -> java_heap
| heap_empty: java_heap.

Module refType <: DecType.
Definition t := Reference.
Definition t_dec := Reference_dec.
End refType.

Module ref := Dec refType.
Definition heap := java_heap.
Definition store := heap_store.
Fixpoint select (h: heap) (r: Reference) {struct h} : S :=
match h with 
| heap_store h r' v => if (ref.EQ r r') then v else select h r
| heap_empty => bottom
end.


Definition select_store1: 
    forall(var: heap) (obj : Reference)(val : AnySet), (select (store var obj val) obj) = val.
Proof.
intros.
simpl.
rewrite ref.EQ_refl; trivial.
Qed.

Definition select_store2: 
    forall(var: heap) (obj1 obj2 :Reference) (val :  AnySet), 
         obj1 <> obj2 -> 
                 (select (store var obj1 val) obj2) = (select var obj2).
Proof.
intros.
simpl.
rewrite ref.EQ_false; trivial.
intro; subst.
apply H; trivial.
Qed.




End EscJavaTypesDef.



