Require Import Bool.
Require Import Sumbool.
Module Type DecType.
Parameter t: Set.
Parameter t_dec: forall x y : t, {x = y} + {x <> y}.
End DecType.
Module Type TDec.
    Parameter t: Set.
    Parameter t_dec: forall x y : t, {x = y} + {x <> y}.
    Parameter EQ : t -> t -> bool.

    Axiom EQ_refl : forall x, EQ x x = true. 
    Axiom EQ_eq : forall x y, EQ x y = true -> x = y. 
    Axiom EQ_true : forall x y, x = y-> EQ x y = true.
    Axiom EQ_eq_not_false : forall x y, x=y-> EQ x y <> false.
    Axiom EQ_false_not_eq :  forall x y, EQ x y = false -> x <> y.
    Axiom EQ_exists_eq : forall x y, {b:bool  | EQ x y = b}.
    Axiom EQ_false : forall x y, x <> y -> EQ x y = false.
End TDec.

Module Dec (dec: DecType) <: TDec 
        with Definition t := dec.t 
        with Definition t_dec := dec.t_dec.
    Definition t := dec.t. 
    Definition t_dec := dec.t_dec.
    Definition EQ : t -> t -> bool :=  
              (fun x y => if (t_dec x y) then true else false).
    Definition EQ_refl : forall x, EQ x x = true. 
    Proof with auto.
    intro.
    cbv delta [EQ]  beta.
    destruct (t_dec x x)...
    Qed.

    Definition EQ_eq : forall x y, EQ x y = true -> x = y. 
    Proof with auto.
    intros.
    cbv delta [EQ]  beta in H.
    destruct (t_dec x y)...
    inversion H.
    Qed.

    Definition EQ_true : forall x y, x = y-> EQ x y = true.
    Proof fun x y (H : x = y) =>
    eq_ind_r (fun x0 => EQ x0 y = true) (EQ_refl y) H.
    
    Definition EQ_eq_not_false : forall x y, x=y-> EQ x y <> false.
    Proof fun x y  (e : x = y) => eq_ind_r (fun b  => b <> false)
    (fun H : true = false =>  False_ind False (eq_ind true 
    (fun ee => if ee then True else False) I false H)) 
    (EQ_true x y e).
    
    Definition EQ_false_not_eq :  forall x y, EQ x y = false -> x <> y.
    Proof fun x y H e => EQ_eq_not_false x y e H.
    
    Definition EQ_exists_eq : forall x y, {b:bool  | EQ x y = b}.
    Proof  fun x y  => let H := EQ x y in exist (fun b => H = b) H (refl_equal H).
    
    Definition EQ_false : forall x y, x <> y -> EQ x y = false.
    Proof fun x y H => not_true_is_false  (EQ x y) (fun h  => H (EQ_eq x y h)).
End Dec.
    
Module Type Arith.
    Parameter t: Set.
    
    Variable GE : t -> t -> Prop.
    Variable GT : t -> t -> Prop.
    Variable LE : t -> t -> Prop.
    Variable LT : t -> t -> Prop.
    
    Variable Div :  t -> t -> t.
    Variable Add : t -> t -> t.
    Variable Sub : t -> t -> t.
    Variable Mul : t -> t -> t.
    Variable Neg : t ->  t.
End Arith.


Module Type Arith_bool. 
    Parameter t: Set.
    Parameter EQ : t -> t -> bool.
   
    Axiom EQ_refl : forall x: t, EQ x x = true. 
    Axiom EQ_eq : forall x y: t, EQ x y = true -> x = y. 
    Axiom EQ_true : forall x y: t, x = y-> EQ x y = true.
    Axiom EQ_eq_not_false : forall x y: t, x=y-> EQ x y <> false.
    Axiom EQ_false_not_eq :  forall x y: t, EQ x y = false -> x <> y.
    Axiom EQ_exists_eq : forall x y: t, {b:bool  | EQ x y = b}.
    Axiom EQ_false : forall x y: t, x <> y -> EQ x y = false. 
    Variable GE : t -> t -> bool.
    Variable GT : t -> t -> bool.
    Variable LE : t -> t -> bool.
    Variable LT : t -> t -> bool.
End Arith_bool.



Module Type EscJavaTypes.
Parameter Boolean: Set.
Axiom Boolean_dec: forall x y : Boolean, {x = y}+{x <> y}.
Variable java_lang_Boolean_TRUE: Boolean.
Variable java_lang_Boolean_FALSE: Boolean.


Parameter time: Set.
Axiom time_dec: forall x y : time, {x = y}+{x <> y}.
Module Time.
    Variable GE : time -> time -> Prop.
    Variable GT : time -> time -> Prop.
    Variable LE : time-> time -> Prop.
    Variable LT : time -> time -> Prop.
End Time.


Parameter DiscreteNumber : Set.
Axiom DiscreteNumber_dec: forall x y : DiscreteNumber, {x = y}+{x <> y}.
Module Discrete <: Arith.
    Definition t := DiscreteNumber.
    
    Variable GE : t -> t -> Prop.
    Variable GT : t -> t -> Prop.
    Variable LE : t -> t -> Prop.
    Variable LT : t -> t -> Prop.
    
    Variable Div :  t -> t -> t.
    Variable Add : t -> t -> t.
    Variable Sub : t -> t -> t.
    Variable Mul : t -> t -> t.
    Variable Neg : t ->  t.
End Discrete.

Module Discrete_bool <: Arith_bool.
    Definition t := DiscreteNumber.
    Parameter EQ : t -> t -> bool.
   
    Axiom EQ_refl : forall x: t, EQ x x = true. 
    Axiom EQ_eq : forall x y: t, EQ x y = true -> x = y. 
    Axiom EQ_true : forall x y: t, x = y-> EQ x y = true.
    Axiom EQ_eq_not_false : forall x y: t, x=y-> EQ x y <> false.
    Axiom EQ_false_not_eq :  forall x y: t, EQ x y = false -> x <> y.
    Axiom EQ_exists_eq : forall x y: t, {b:bool  | EQ x y = b}.
    Axiom EQ_false : forall x y: t, x <> y -> EQ x y = false. 
    Variable GE : t -> t -> bool.
    Variable GT : t -> t -> bool.
    Variable LE : t -> t -> bool.
    Variable LT : t -> t -> bool.
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
Parameter JavaNumber: Set.
Parameter JavaNumber_to_DiscreteNumber: JavaNumber -> DiscreteNumber.
Parameter JavaNumber_to_ContinuousNumber: JavaNumber -> ContinuousNumber.

(* JMLNumber are JavaNumber, RealNumber, BigIntNumber *)
Parameter JMLNumber: Set.
Parameter JMLNumber_to_JavaNumber: JMLNumber -> JavaNumber.
Parameter JMLNumber_to_RealNumber: JMLNumber -> RealNumber.
Parameter JMLNumber_to_BigIntNumber: JMLNumber -> BigIntNumber.

(* Numbers are JMLNumbers... don't see the point *)
Parameter Number: Set.
Parameter Number_to_JMLNumber: Number -> JMLNumber.

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

Parameter RefType : Set.
Axiom RefType_dec: forall x y : RefType, {x = y}+{x <> y}.


Parameter Reference: Set.
Axiom Reference_dec : forall x y : Reference, {x = y} + {x <> y}.
Variable null: Reference.



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


Parameter LockMap : Set.
Variable LS: LockMap.
Variable maxLockSet: Reference.
Axiom LockMap_dec : 
           forall x y : LockMap, {x = y} + {x <> y}.
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


Parameter Path : Set.
Parameter Path_dec: forall x y : Path, {x = y} + {x <> y}.




Parameter Elem : Set.
Parameter Field : Set.
Parameter AnySet: Set.


Variable cast : Reference -> RefType -> Reference.
Variable fClosedTime : Field -> time.






Variable vAllocTime : Reference -> time.
Definition isAllocated: Reference -> time -> Prop := 
    fun (obj : Reference) (t: time) => Time.LT (vAllocTime obj) t.

Variable ecReturn : Path.
Variable ecThrow : Path.

Variable distinctAxiom : ecReturn <> ecThrow.
Hint Immediate distinctAxiom.



(*Variable is : S -> S -> Prop. *)
Variable isField : Field -> Prop.
Variable allocNew_ : Reference.
Variable alloc_ : Reference.


Variable asField: Field -> RefType -> Prop.
Variable asElems: Elem -> Elem.
Variable asLockSet: LockMap -> LockMap.

Variable eClosedTime: Elem -> time.
Variable zero : DiscreteNumber.
(* Array Logic *)
Module array.
    Parameter shapeOne: DiscreteNumber -> Reference.
    Parameter shapeMore: DiscreteNumber-> Reference -> Reference.
    Parameter isNew: Reference -> Prop.
    Parameter length: Reference -> DiscreteNumber.
    Axiom lengthAx :
       forall (a : Reference), (Discrete.LE zero (length a)).
    Parameter select: Reference -> DiscreteNumber -> AnySet.
    Parameter store: Reference -> DiscreteNumber -> AnySet -> Reference.
    Axiom select_store1: 
       forall(var :Reference) (obj : DiscreteNumber )(val :AnySet), (select (store var obj val) obj) = val.
    Axiom select_store2: 
       forall(var : Reference)(obj1 obj2 :DiscreteNumber) (val :  AnySet), 
         obj1 <> obj2 -> 
                 (select (store var obj1 val) obj2) = (select var obj2).
    Parameter fresh : Reference -> time -> time -> Reference -> Reference -> RefType ->  AnySet (* A *) -> Prop.

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

Parameter heap: Set.

Variable select: heap -> Reference -> AnySet.
Variable store: heap -> Reference -> AnySet -> heap.
Axiom select_store1: 
    forall(var:heap) (obj : Reference)(val : AnySet), (select (store var obj val) obj) = val.
Axiom select_store2: 
    forall(var: heap) (obj1 obj2 :Reference) (val :  AnySet), 
         obj1 <> obj2 -> 
                 (select (store var obj1 val) obj2) = (select var obj2).


End EscJavaTypes.





Module EscJavaLogic (types: EscJavaTypes).
(* the coercion from bool to prop to do alike simplify ;) *)
Coercion Is_true : bool >-> Sortclass.

Import types.

Definition Path_eq_bool (x: Path) (y: Path) := bool_of_sumbool (Path_dec x y).

Definition Reference_eq_bool (x: Reference) (y: Reference) := (Reference_dec x y).


Ltac unfoldEscTime := unfold isAllocated.

Lemma negb_elim :
    forall b, negb (negb b) = b.
Proof fun b  => sym_eq (negb_intro b).


(* Hint Rewrite ref.EQ_refl : escj.*)

End EscJavaLogic.


