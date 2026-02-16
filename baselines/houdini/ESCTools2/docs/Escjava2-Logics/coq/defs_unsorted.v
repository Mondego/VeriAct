Require Import Bool.
Require Import ZArith.
Require Import Zdiv.
Require Import Zbool.
Require Import Classical.
Open Scope Z_scope.

(* LA tactique primitive *)
Ltac genclear H := generalize H;clear H.

Variable S: Set.




(* Some arithmetic things *)
Variable S_to_Z : S -> Z.
Variable Z_to_S : Z -> S.
Axiom Z_to_S_elim : forall x,  S_to_Z (Z_to_S x) = x.
Variable S_to_bool : S -> bool.
Axiom S_to_Z_det: 
         forall x : S,   (S_to_Z x) = (S_to_Z x).
Axiom S_to_bool_det: 
         forall x y: S,  x = y -> (S_to_bool x) = (S_to_bool y).
Hint Immediate S_to_Z_det S_to_bool_det.
Hint Rewrite Z_to_S_elim:escj.
Lemma Zeq_bool_sym :
forall x:Z, Zeq_bool x x = true.
Proof.
unfold Zeq_bool; intro; rewrite Zcompare_refl; trivial.
Qed.

Lemma Zeq_bool_true:
forall x y: Z, x = y ->Zeq_bool x y = true.
Proof.
intros x y H; subst.
apply Zeq_bool_sym .
Qed.
Lemma Zeq_bool_false:
forall x y: Z, x <> y ->Zeq_bool x y = false.
Proof.
intros x y H; unfold Zeq_bool;
assert(h := Zcompare_Eq_iff_eq x y);
destruct h; destruct (x ?= y) ; auto.
assert(x = y); [apply H0; trivial | destruct H; trivial].
Qed.
Lemma negb_elim :
    forall b, negb (negb b) = b.
Proof.
intro b; apply sym_eq; apply negb_intro.
Qed.

Ltac subst_S_to :=
repeat match goal with 
|[ H: S_to_bool ?x = _ |- _] =>
      let v := fresh "X" in (set (v := S_to_bool x) in *; clearbody v; subst v)
|[ H: S_to_Z ?x = _ |- _] =>
      let v := fresh "X" in (set (v := S_to_bool x) in *; clearbody v; subst v)
end.

Hint Rewrite negb_elim :escj.
Hint Resolve Zeq_bool_sym Zeq_bool_true Zeq_bool_false.

Definition integralGE : Z -> Z -> Prop := fun (x y : Z) => x >= y.
Definition integralGT : Z -> Z -> Prop := fun (x y : Z) =>  x > y.
Definition integralLE : Z-> Z -> Prop := fun (x y : Z) => x <= y.
Definition integralLT : Z -> Z -> Prop := fun (x y : Z) => x < y.

Definition integralEQ_bool : Z -> Z -> bool := fun (x y : Z) => Zeq_bool x  y.
Definition integralGE_bool : Z -> Z -> bool := fun (x y : Z) => Zge_bool x y.
Definition integralGT_bool : Z -> Z -> bool := fun (x y : Z) => Zgt_bool x y.
Definition integralLE_bool : Z -> Z -> bool := fun (x y : Z) =>  Zle_bool x y.
Definition integralLT_bool : Z -> Z -> bool := fun (x y : Z) => Zlt_bool x y.

Definition integralDiv : Z -> Z -> Z := 
             fun (x y : Z) =>(x  / y).
Definition integralAdd : Z -> Z -> Z := 
             fun (x y : Z) =>(x  + y).
Definition integralSub : Z -> Z -> Z := 
             fun (x y : Z) =>(x - y).
Definition integralMul : Z -> Z -> Z := 
             fun (x y : Z) =>(x  * y).
Definition integralNeg : Z -> Z := 
             fun (x : Z) =>(0 - x).
Ltac unfoldEscArith := unfold  integralEQ_bool, integralGE_bool, integralLE_bool, integralGT_bool, integralLT_bool, integralGE, integralLE, integralGT, integralLT, integralAdd, integralSub, integralMul, integralNeg in *.



Variable select: S -> S -> S.
Variable store : S -> S -> S -> S.
Variable arr_store: S -> Z -> S -> S.

Axiom select_store1: 
    forall(var val obj :S), (select (store var obj val) obj) = val.
Axiom select_store2: 
    forall(var val obj1 obj2 :S), 
         obj1 <> obj2 -> 
                 (select (store var obj1 val) obj2) = (select var obj2).

Variable store_int: S -> S -> Z -> S.
Variable arr_store_int: S -> Z -> Z -> S.
Axiom select_store_int1: 
    forall(var : S) (val: Z) (obj :S), S_to_Z (select (store_int var obj val) obj) = val.
Axiom select_store_int2: 
    forall(var : S) (val: Z)( obj1 obj2 :S), 
         obj1 <> obj2 -> 
                 (select (store_int var obj1 val) obj2) = (select var obj2).

Variable store_bool: S -> S -> bool -> S.
Variable arr_store_bool: S -> Z -> bool -> S.
Axiom arr_select_store_bool : 
 forall (var: S) (val: bool) (i : Z) (j: S),
      i = (S_to_Z j) -> S_to_bool (select (arr_store_bool var i val) j) = val.
Axiom select_store_bool1: 
    forall(var : S) (val: bool) (obj :S), S_to_bool (select (store_bool var obj val) obj) = val.
Axiom select_store_bool2: 
    forall(var : S) (val: bool)( obj1 obj2 :S), 
         obj1 <> obj2 -> 
                 S_to_bool (select (store_bool var obj1 val) obj2) = S_to_bool (select var obj2).
Hint Resolve select_store2 select_store_int2 select_store_bool2.
Hint Rewrite  -> select_store1 select_store_int1 select_store_bool1: escj_select.

Variable null : S.
Variable isAllocated: S -> S-> Prop.
Variable fClosedTime : S -> S.

(* The following are grabbed from Jack's Coq Plugin *)
Variable refEQ : S -> S-> bool.
Variable refEQ_refl : forall x, refEQ x x = true. 
Variable refEQ_eq : forall x y, refEQ x y = true -> x = y. 
Lemma refEQ_eq_true : forall x y, x = y-> refEQ x y = true.
Proof. 
intros x y H. rewrite H. apply refEQ_refl.
Qed.  
Lemma refEQ_eq_not_false : forall x y, x=y-> refEQ x y <> false.
Proof. 
intros x y e. rewrite refEQ_eq_true; trivial; discriminate.
Qed.  
Lemma refEQ_false_not_eq :  forall x y, refEQ x y = false -> x <> y.
Proof.
exact (fun x y H e => refEQ_eq_not_false x y e H).
Qed.
Definition exists_refEQ_eq : forall x y, {b:bool  | refEQ x y = b}.
Proof.
 intros. exists (refEQ x y). constructor.
Qed.
Lemma not_eq_false_refEQ : forall x y, x <> y -> refEQ x y = false.
Proof.
 intros x y H. apply not_true_is_false.
 intro. apply H. apply refEQ_eq. assumption.
Qed.
(* End of the stolen part *)




Hint Rewrite refEQ_refl : escj.
 
Variable Types : Set.
Variable subtypes: Types -> Types-> Prop.
Variable typeof : S -> Types.
Variable lockLT : S -> S -> Prop.
(*Variable is : S -> S -> Prop. *)
Variable isField : S -> Prop.
Variable allocNew_ : S.


Variable asField: S -> Types -> Prop.
Variable asElems: S -> Prop.
Variable asLockSet: S -> Prop.

Variable eClosedTime: S ->S.

(* Array Logic *)
Variable arrayFresh : S -> S -> S -> S -> S -> Types -> S -> Prop.
Variable arrayShapeOne: S -> S.
Variable arrayShapeMore: S -> S-> S.
Variable isNewArray: S -> Prop.
Variable arrayLength: S -> S.
Axiom arrayLengthAx :
      forall (a : S), (integralLE 0 (S_to_Z (arrayLength a))).

(* array axioms2 *)
(* Axiom array_axiom2_1 : 
      forall (a, a0, b0, e, n : S) (T: Types) (v : S):
        arrayFresh(a, a0, b0, e, arrayShapeOne(n), T, v) ->
        (a0 <= (vAllocTime a)). *)
Axiom array_axiom2_2 : 
      forall (a  a0  b0  e  n: S)  (T: Types)  (v : S),
        (arrayFresh a  a0  b0 e (arrayShapeOne n) T v) ->
         (isAllocated a  b0).
Axiom array_axiom2_3 : 
     forall (a  a0  b0  e  n: S)  (T: Types)  (v : S),
        (arrayFresh a  a0  b0 e (arrayShapeOne n) T v) ->
         (a <> null). 
Axiom array_axiom2_4 : 
     forall (a  a0  b0  e  n: S)  (T: Types)  (v : S),
        (arrayFresh a  a0  b0 e (arrayShapeOne n) T v) ->
         (typeof a) = T.
Axiom array_axiom2_5 : 
     forall (a  a0  b0  e  n: S)  (T: Types)  (v : S),
        (arrayFresh a  a0  b0 e (arrayShapeOne n) T v) ->
         (arrayLength a) = n.
Axiom array_axiom2_6 : 
     forall (a  a0  b0  e  n: S)  (T: Types)  (v : S),
        (arrayFresh a  a0  b0 e (arrayShapeOne n) T v) ->
         forall (i : S),
           (select (select e a) i) = v.

(* maybe another day... Hint Resolve array_axiom2_2 array_axiom2_3 array_axiom2_4 array_axiom2_5 array_axiom2_6. *)

Ltac unfoldArrAx_intern R :=
         match goal with
          | [H : (arrayFresh ?a  ?a0  ?b0 ?e (arrayShapeOne ?n) ?T ?v) |- _ ] =>
                 let H1 := fresh "H" in (assert (H1 :=  (array_axiom2_2 a a0  b0  e  n  T  v H)));
                 let H1 := fresh "H" in (assert (H1 :=  (array_axiom2_3 a a0  b0  e  n  T  v H)));
                 let H1 := fresh "H" in (assert (H1 :=  (array_axiom2_4 a a0  b0  e  n  T  v H)));
                 let H1 := fresh "H" in (assert (H1 :=  (array_axiom2_5 a a0  b0  e  n  T  v H))); 
                 let H1 := fresh "H" in (assert (H1 :=  (array_axiom2_6 a a0  b0  e  n  T  v H)));
                genclear H; unfoldArrAx_intern R
          | [H: _ |- _ ] => genclear H; unfoldArrAx_intern R
          | _ => intros
        end.

Ltac unfoldArrAx  :=
         match goal with
          | [H : _|- ?R ] =>  unfoldArrAx_intern R
        end.

Ltac arrtac :=
match goal with
| [H1: (arrayFresh null  ?a0  ?b0 ?e (arrayShapeOne ?n) ?T ?v) |- _] => 
                 let H:= fresh "H" in (assert (H :=  (array_axiom2_3 null a0  b0  e  n  T  v H1)); destruct H; trivial)
| [H1: (arrayFresh ?a  ?a0  ?b0 ?e (arrayShapeOne ?n) ?T ?v) |- _] => 
                 let H:= fresh "H" in (assert (H :=  (array_axiom2_3 null a0  b0  e  n  T  v H1)); destruct H; trivial)
end.
Variable ecReturn : S.
Variable ecThrow : S.
Axiom distinctAxiom : ecReturn <> ecThrow.
Hint Immediate distinctAxiom.




(* Sans elimOr/elimAnd c'est un peu la deche *)
(**
 ** elimAnd
 elimAnd is used mainly to eliminate and within the hypothesis.
For the goals the preferred tactic is still split.

*)


Ltac map_hyp Tac :=
  repeat match goal with
    | [ H : _ |- _ ] => try (Tac H);genclear H
    end; intros.

Inductive Plist : Prop :=
  | Pnil : Plist
  | Pcons : Prop -> Plist -> Plist.

Ltac build_imp Pl C :=
  match Pl with
  | Pnil => constr:C
  | Pcons ?A ?Pl' =>
        let C' := constr:(A->C) in
        build_imp Pl' C'
  end.

Inductive PPlProd : Prop :=
 | PPlPair : Plist -> Prop -> PPlProd.

Ltac elim_aT Pl T :=
  match T with
  | ?A /\ ?B =>
      let A' := elim_aT Pl A in
      let B' := elim_aT Pl B in
      constr:(A' /\ B')
  | ?A => build_imp Pl T
  end.
Ltac elim_iaT Pl T :=
   match T with
   | (?B /\ ?C) =>
        let P := elim_aT Pl T in
        constr:(PPlPair Pl P)
   | (?A -> ?D) =>
        let Pl' := constr:(Pcons A Pl) in
        elim_iaT Pl' D
   end.

Ltac splitAndH H :=
  match type of H with
  | ?A /\ ?B =>
             case H;clear H;
             let H1 := fresh "H" in
             let H2 := fresh "H" in
             (intros H1 H2; splitAndH H1; splitAndH H2)
  | _ => idtac
  end.
Ltac introPl Pl H :=
 match Pl with
 | Pnil => splitAndH H
 | Pcons _ ?pl =>
     let H1 := fresh "H" in
     let H2 := fresh "H" in
     (intro H1;assert (H2 := H H1);introPl pl H2)
 end.

Ltac splitAnd :=
  match goal with
  | [|- ?A /\ ?B] => split;splitAnd
  | _ => idtac
  end.
Ltac elimAnd :=
  match goal with
  | [H : ? A /\ ?B |- _ ] =>
             case H;clear H;
             let H1 := fresh "H" in
             let H2 := fresh "H" in
             intros H1 H2; elimAnd
  | [H : ?HT|- _ ] =>
       let pair := elim_iaT Pnil HT in
       match pair with
       | PPlPair ?Pl ?newT =>
         assert newT;
         [splitAnd; introPl Pl H;trivial
         | clear H;elimAnd]
       end
  | [H: _ |- _ ] => genclear H;elimAnd
  | _ => intros; repeat match goal with [H: _ |- _ /\ _] => split end
  end.

(**
 ** elimNor
This tactic is used to remove the not in front of a or (in the hypothesis),
turning [~ (A \/  B) ] into [(~ A) /\ (~ B)].

*)

Definition distr_or : forall A B, ~ (A \/ B) ->  ~ A.
 intros A B a b.
 elim a; trivial.  left; trivial.
Qed.

Definition distr_or_inv : forall A B, ~ (A \/ B) ->  ~B.
 intros A B a b.
 elim a; trivial.  right; trivial.
Qed.
Ltac elimNorCons a b := let H1 := fresh "H" in
                                          assert (H1 : ~ (a)); cut (~(a \/ b)) ; auto;
                                          let H2 := fresh "H" in
                                          assert (H2 : ~ (b)); cut (~(a \/ b)); auto.

Ltac elimNor :=
  repeat match goal with
  |   [H: ~ (?a \/ ?b) |- _ ] => elimNorCons a b; clear H;  let H0 := fresh "H" in (let H1 := fresh "H" in  (intros H0 H1; clear H0 H1))
end.

Ltac elim_or H HT R:=
        match HT with
           | ?A \/ R =>
                 let h := fresh "H" in assert(h : A \/  R); [apply H; intros; auto | idtac]
         | R\/ ?B  =>
                 let h := fresh "H" in assert(h : R \/  B); [apply H; intros; auto |idtac]
           |  ?C -> ?D => elim_or H D R
        end.

Ltac solve_or R :=
         match goal with
          | [H : R \/ ?B |-_ ] => destruct H; auto; try contradiction; intros
          | [H : ?A \/ R |- _ ] => destruct H;  auto; try contradiction; intros
          | [H : ?HT|- _ ] => elim_or H HT R; genclear H;solve_or R
          | [H: _ |- _ ] => genclear H; solve_or R
          | _ => intros
        end.

Ltac solveOr :=
         match goal with
          | [H : _|- ?R ] =>  solve_or R
        end.

Ltac simplOr_rec h name tail :=
        match h with
        |    ?A \/ ?B -> ?C  => assert(tail -> A -> C); [intros; apply name;trivial; left;trivial| idtac];
                                                        assert(tail -> B -> C); [intros; apply name; trivial; right;trivial| idtac]; clear name
        |    ?A -> ?B => simplOr_rec B name (tail -> A)
        end.

Ltac simplOr_im h name :=
        match h with
        |    ?A \/ ?B -> ?C  => assert(A -> C); [intros; apply name; left;trivial| idtac];
                                                        assert(B -> C); [intros; apply name; right;trivial| idtac]; clear name
        |    ?A -> ?B => simplOr_rec B name A
        end.
Ltac simplOr :=
        repeat match goal with
        |   [H: ?A |- _] =>simplOr_im A H
        end.

Ltac elimOr := simplOr.





(**
 ** Cleansing
To clean up the mess (sometimes).

*)
Lemma a_ou_a_donne_a: forall a : Prop, a \/ a -> a.
Proof.
   intros a H; destruct H; assumption.
Qed.

Ltac cleanUp :=
repeat match goal with
| [H1: ?a, H2: ?a |- _] => clear H2
| [H1: ?a = ?a |- _ ] => clear H1
| [ H: ?a \/ ?a |- _] => let H1 := fresh "H" in (assert(H1 := a_ou_a_donne_a a H); clear H)
| [H1: ?a < ?b, H2: ?a <= ?b |- _] => clear H2
end.



Ltac autosc := intros; subst; elimOr; elimAnd; intros; subst;  
repeat match goal with
|[H: _ |- _] => rewrite select_store1 in H
end;
repeat match goal with
|[H: _ |- _] => rewrite select_store_int1 in H
end;
repeat match goal with
|[H: _ |- _] => rewrite select_store_bool1 in H
end; autorewrite with escj; auto.

Ltac startsc := unfold not; unfoldEscArith; autorewrite with escj; autorewrite with escj_select; intros; subst.
(* unfoldArrAx. *)

Definition int := Z.
Definition Field := S.

Definition Path := S.
Definition RefType := Types.
Definition time := S.
Coercion S_to_Z : S >-> Z.
Coercion Z_to_S : Z >-> S.


