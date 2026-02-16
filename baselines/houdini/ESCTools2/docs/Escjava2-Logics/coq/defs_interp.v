
Require Import String.

Inductive Var : Set := 
| Path_var : string -> Var
| Reference_var : string -> Var
| Types_var : string -> Var
| int_var : string -> Var
| Time_var : string -> Var.
Variable var : string -> S.
Variable ivar: string -> int.



Require Import List.
Inductive jReference : Set :=
rvar : string -> jReference
| rval: Reference -> jReference
| select: jReference -> jReference -> jReference.

Inductive jTime: Set :=
tmvar: string -> jTime
| tmval: Time -> jTime.
Inductive jTypes: Set :=
tvar: string -> jTypes
| tval: Types -> jTypes
| ttypeof: jReference -> jTypes.

Inductive jProp : Set :=
| bforall :  Var -> jProp-> jProp
| true : jProp
| anyEQ : S -> S -> jProp
| refEQ : jReference -> jReference -> jProp
| refNE : jReference -> jReference -> jProp
| pathEQ: Path -> Path -> jProp
| typeLE : jTypes -> jTypes -> jProp
| false : jProp
| bnot: jProp -> jProp
| band: jProp -> jProp -> jProp
| bor: jProp -> jProp -> jProp
| bimplies : jProp -> jProp -> jProp
| allocLT: jTime -> jTime -> jProp
| timeEQ: jTime -> jTime -> jProp
| bIsAllocated: jReference -> jTime -> jProp.
Open Scope string_scope.



(* Notation "A -> B" := (jimply A B) (at level 70) : jProp_scope. *)

(* Notation "'forall' f" := (jforall f) (at level 70): jProp_scope. *)

Definition jLemma := ((list jProp) * jProp)%type.

Definition simplify : list jProp -> jProp -> jLemma := fun l p => (l, p).
Require Import Bool.

Definition env : Set := list (string * S).
Definition basic_env := (null, Reference_to_S null) ::
               (ecThrow, Path_to_S ecThrow)::
               (ecReturn, Path_to_S ecReturn):: nil.
Fixpoint eval_path (v: Path) (e: env) {struct e}: Path :=
match e with
| (pair n var) :: e' => if  (Path_dec v n) then var else eval_path v e'
| nil => v
end.
Fixpoint eval_ref (v: string) (e: env) {struct e}: S :=
match e with
| (pair n var) :: e' => if  (Reference_dec v n) then var else eval_ref v e'
| nil => null
end.

Fixpoint eval_time (v: string) (e: env) {struct e}: Time :=
match e with
| (pair n var) :: e' => if  (string_dec v n) then var else eval_time v e'
| nil => 0
end.
Fixpoint interpRef (r : jReference) (e: env) {struct r} : Reference := 
    match r with
    | rvar v => eval_ref v e
    | rval v => v
    | select own obj => Heap.select (interpRef own e) (interpRef obj e)
    end.

Definition interpTime (r : jTime) (e: env): Time := 
    match r with
    | tmvar v => eval_time v e
    | tmval v => v
    end.
Fixpoint eval_type (v: string) (e: env) {struct e}: Types :=
match e with
| (pair n var) :: e' => if  (string_dec v n) then var else eval_type v e'
| nil => 0
end.
Definition interpType (r : jTypes) (e: env): Types := 
    match r with
    | tvar v => eval_type v e
    | tval v => v
    | ttypeof r => typeof (interpRef r e)
    end.

Fixpoint interpProp (p : jProp) (e: env) {struct p} :  Prop :=
match p with
| bforall l p => match l with
                                   |   Path_var pat => forall (path: Path),   interpProp p  ((pat, Path_to_S path) :: e)
                                   |   Reference_var pat => forall (ref: Reference),   interpProp p  ((pat, Reference_to_S ref) :: e)
                                   |   Types_var pat => forall (type: Types),   interpProp p  ((pat, Types_to_S type) :: e)
                                  |   Time_var pat => forall (time: Time),   interpProp p  ((pat, Time_to_S time) :: e)
                                  
                        (* |   int_to_S i => 
                                   |   long_to_S: long -> S
                                   |   float_to_S: float -> S
                                   |   double_to_S: double -> S
                                   |   Reference_to_S: Reference -> S
                                   |   Path_to_S: Path -> S
                                   |   Time_to_S: Time -> S
                                   |   Types_to_S: Types -> S.*)
                                    | _ =>  interpProp p e
                                   end 
| anyEQ v1 v2 => v1 = v2
| refEQ v1 v2 => (interpRef v1 e) = (interpRef v2 e)
| pathEQ v1 v2 => (eval_path v1 e) = (eval_path v2 e)
| typeLE v1 v2 => (interpType v1 e) <= (interpType v2 e)
| refNE v1 v2 =>  (interpRef v1 e) <> (interpRef v2 e)
| true => True
| false => False
| bnot p =>  not (interpProp p e) 
| band p1 p2 =>  (interpProp p1 e) /\ (interpProp p2 e)
| bor p1 p2 =>  (interpProp p1 e) \/ (interpProp p2 e)
| bimplies p1 p2 => (interpProp p1 e) -> interpProp p2 e
| allocLT v1 v2 => (interpTime v1 e) < (interpTime v2 e)
| timeEQ v1 v2 => (interpTime v1 e) = (interpTime v2 e)
| bIsAllocated v1 v2 => isAllocated (interpRef v1 e) (interpTime v2 e)
(* | _ => False *)
end.


Fixpoint interp_hypos (l: list jProp) (g: jProp) {struct l} : Prop :=
match l with
| cons a L => (interp_hypos L a) -> interpProp g nil
| nil => interpProp g nil
end.

Definition interp (pr: jLemma): Prop :=
match pr with
   pair l g => interp_hypos l g
end.

Variable bjava_lang_Throwable__stackTrace : S -> S -> Reference.
Variable bgetStackTrace__state : S -> S -> Reference.
Variable bjava_lang_Object_Object_49_4 : S -> S -> Reference.

Definition lemma1 :=  
(bforall (Reference_var "objectToBeConstructed")
       (bforall (Reference_var "XRES_2_18_2_18")
            (bforall (Types_var "java_lang_Object")
      (bforall (Reference_var "elems")
(bforall (Reference_var "|i:3.12|")
(bforall (Reference_var "|i@pre:3.12|")
(bforall (Reference_var "|RES-2.18|")
(bforall (Reference_var "RES")
(bforall (Reference_var "XRes")
(bforall (Reference_var "LS")
(bforall (Reference_var "|elems@pre|")
(bforall (Reference_var "state")
(bforall (Reference_var "|state@pre|")
(bforall (Time_var "|alloc@pre|")
(bforall (Time_var "alloc")
(bforall (Time_var"|after@2.18-2.18|")
(bforall (Types_var "java.lang.Exception")
(bforall (Types_var "Blue")
(bforall (Reference_var"|owner:4.44.28|")
(*
((* %NotHandled *) brokenObj : Reference)
(java_lang_Exception : Types)
((* %NotHandled *) brokenObj : Reference)
(ReferenceType : Types)
((* %NotHandled *) brokenObj : Reference) (state : Reference) *)
(bforall (Path_var "|EC-2.18:2.18|")
(bimplies 
  (refEQ (rvar "|i@pre:3.12|") (rvar "|i:3.12|")) 
  (bimplies 
    (refEQ (rvar "|elems@pre|") (rvar "elems")) 
    (bimplies 
      (allocLT 
        (tmval (eClosedTime "elems")) (tmvar "alloc")) 
      (bimplies 
        (timeEQ (tmvar "|alloc@pre|")  (tmvar "alloc")) 
        (bimplies 
          (refEQ (rvar "|state@pre|") (rvar "state")) 
          (bimplies 
            (refNE (rvar "RES") (rval null)
            ) 
            (bimplies 
              (bnot 
                (bIsAllocated (rvar "RES") (tmvar "alloc"))
              ) 
              (bimplies 
                (bforall (Types_var "Unknown tag <247>")
                  (refEQ 
                    (rval (bjava_lang_Throwable__stackTrace (var "state") (var "brokenObj"))) 
                    (rval (bgetStackTrace__state (var "state") (var "brokenObj")))
                  )) 
                (bimplies 
                  (bforall (Types_var "Unknown tag <247>")  
                    (refEQ 
                      (rval (bjava_lang_Throwable__stackTrace (var "state") (var "|brokenObj<1>|"))) 
                      (rval (bgetStackTrace__state (var "state") (var "|brokenObj<1>|")))
                    )) 
                  (bimplies 
                    (bnot 
                      (bIsAllocated (rvar "objectToBeConstructed") (tmvar "alloc"))
                    ) 
                    (bimplies 
                      (refNE (rvar "|RES-2.18|") (rval null)
                      ) 
                      (bimplies 
                        (bnot 
                          (bIsAllocated (rvar "|RES-2.18|") (tmvar "alloc"))
                        ) 
                        (bimplies true 
                          (bimplies 
                            (pathEQ "|EC-2.18:2.18|" "ecReturn") 
                            (bimplies 
                              (allocLT (tmvar "alloc") (tmvar "|after@2.18-2.18|")) 
                              (bimplies 
                                (refNE (rvar "|RES-2.18|") (rval null)
                                ) 
                                (bimplies 
                                  (bnot 
                                    (bIsAllocated (rvar "|RES-2.18|") (tmvar "alloc"))
                                  ) 
                                  (bimplies 
                                    (bIsAllocated (rvar "|RES-2.18|") (tmvar "|after@2.18-2.18|")) 
                                    (bimplies 
                                      (refEQ (rvar "|RES-2.18|")
                                        (rval (bjava_lang_Object_Object_49_4 (var "alloc_") (var "allocNew_")))
                                      ) 
                                      (bimplies 
                                        (refEQ 
                                          (select  (rvar "|owner:4.44.28|") (rvar "|RES-2.18|")) (rval null)
                                        ) 
                                        (bimplies 
                                          (bnot 
                                            (band 
                                              (pathEQ "|EC-2.18:2.18|" ecThrow) 
                                              (typeLE 
                                                (ttypeof (rvar "|XRES-2.18:2.18|")) (tvar "java.lang.Exception"))
                                            )
                                          ) 
                                          (bimplies 
                                            (typeLE 
                                              (ttypeof (rvar "|RES-2.18|")) (tvar "Blue")) 
                                            (bimplies 
                                              (refEQ (rvar "objectToBeConstructed") (rvar "|RES-2.18|")) 
                                              (bimplies 
                                                (bforall (Types_var "Unknown tag <247>")  
                                                  (refEQ 
                                                    (rval (bjava_lang_Throwable__stackTrace (var "state") (var "brokenObj"))) 
                                                    (rval (bgetStackTrace__state (var "state") (var "brokenObj")))
                                                  )) 
                                                (bimplies 
                                                  (bforall  (Types_var "Unknown tag <247>") 
                                                    (refEQ 
                                                      (rval (bjava_lang_Throwable__stackTrace (var "state") (var "|brokenObj<1>|"))) 
                                                      (rval (bgetStackTrace__state (var "state") (var "|brokenObj<1>|")))
                                                    )) 
                                                  (bimplies 
                                                    (bor 
                                                      (bnot 
                                                        (pathEQ ecReturn ecReturn)
                                                      ) 
                                                      (band 
                                                        (pathEQ ecReturn ecReturn) 
                                                        (bnot 
                                                          (refEQ 
                                                            (select  (rvar "|owner:4.44.28|") (rvar "|RES-2.18|")) (rval null)
                                                          )
                                                        )
                                                      )
                                                    ) false
                                                  )
                                                )
                                              )
                                            )
                                          )
                                        )
                                      )
                                    )
                                  )
                                )
                              )
                            )
                          ))
                      )
                    )
                  )
                ))
              )))
            )
          ))
        )))
      )))
    ))
  ))))))
))
))))).

Check lemma1.
Variable java_lang_Throwable__stackTrace : S -> S -> Reference.
Variable getStackTrace__state : S -> S -> Reference.
Variable java_lang_Object_Object_49_4 : S -> S -> Reference.

Definition lemma1_interp :=forall (objectToBeConstructed : Reference)
(XRES_2_18_2_18 : Reference)(java_lang_Object : Types)(i_3_12 : Reference)
 ((* %NotHandled *) brokenObj : Reference)(java_lang_Exception : Types)
(elems_pre : Reference)((* %NotHandled *) brokenObj : Reference)
(after_2_18_2_18 : Time)(ReferenceType : Types)
(owner_4_44_28 : Reference)(state_pre : Reference)
(RES_2_18 : Reference)(XRes : Reference)(LS : Reference)
(RES : Reference)(alloc_pre : Time)(elems : Reference)
((* %NotHandled *) brokenObj : Reference)(state : Reference)
(alloc : Time)(i_pre_3_12 : Reference)(EC_2_18_2_18 : Path) ,
(
  (i_pre_3_12 (* Any Eq *) = i_3_12) -> 
  (
    (elems_pre (* Any Eq *) = elems) -> 
    (
      (
        (eClosedTime elems)
         < alloc) -> 
      (
        (alloc_pre (* Any Eq *) = alloc) -> 
        (
          (state_pre (* Any Eq *) = state) -> 
          (
            (RES <>  null) -> 
            (
              (not 
                (isAllocated RES alloc)
              ) -> 
              (
                ((forall ((* %NotHandled *) brokenObj: Reference), 
                  (
                    (java_lang_Throwable__stackTrace state (* %NotHandled *) brokenObj) (* Ref Eq *) = 
                    (getStackTrace__state state (* %NotHandled *) brokenObj)
                  ))
                ) -> 
                (
                  ((forall ((* %NotHandled *) brokenObj: Reference), 
                    (
                      (java_lang_Throwable__stackTrace state (* %NotHandled *) brokenObj) (* Ref Eq *) = 
                      (getStackTrace__state state (* %NotHandled *) brokenObj)
                    ))
                  ) -> 
                  (
                    (not 
                      (isAllocated objectToBeConstructed alloc)
                    ) -> 
                    (
                      (RES_2_18 <>  null) -> 
                      (
                        (not 
                          (isAllocated RES_2_18 alloc)
                        ) -> 
                        (True -> 
                          (
                            (EC_2_18_2_18 (* Any Eq *) = ecReturn) -> 
                            (
                              (alloc < after_2_18_2_18) -> 
                              (
                                (RES_2_18 <>  null) -> 
                                (
                                  (not 
                                    (isAllocated RES_2_18 alloc)
                                  ) -> 
                                  (
                                    (isAllocated RES_2_18 after_2_18_2_18) -> 
                                    (
                                      (RES_2_18 (* Any Eq *) = 
                                        (java_lang_Object_Object_49_4 alloc_ allocNew_)
                                      ) -> 
                                      (
                                        (
                                          (Heap.select  owner_4_44_28 RES_2_18) (* Ref Eq *) =  null) -> 
                                        (
                                          (not 
                                            (
                                              (EC_2_18_2_18 (* Any Eq *) = ecThrow) /\ 
                                              (subtypes 
                                                (typeof XRES_2_18_2_18) java_lang_Exception)
                                            )
                                          ) -> 
                                          (
                                            (subtypes 
                                              (typeof RES_2_18) ReferenceType) -> 
                                            (
                                              (objectToBeConstructed (* Ref Eq *) = RES_2_18) -> 
                                              (
                                                ((forall ((* %NotHandled *) brokenObj: Reference), 
                                                  (
                                                    (java_lang_Throwable__stackTrace state (* %NotHandled *) brokenObj) (* Ref Eq *) = 
                                                    (getStackTrace__state state (* %NotHandled *) brokenObj)
                                                  ))
                                                ) -> 
                                                (
                                                  ((forall ((* %NotHandled *) brokenObj: Reference), 
                                                    (
                                                      (java_lang_Throwable__stackTrace state (* %NotHandled *) brokenObj) (* Ref Eq *) = 
                                                      (getStackTrace__state state (* %NotHandled *) brokenObj)
                                                    ))
                                                  ) -> 
                                                  (
                                                    (
                                                      (not 
                                                        (ecReturn (* Any Eq *) = ecReturn)
                                                      ) \/ 
                                                      (
                                                        (ecReturn (* Any Eq *) = ecReturn) /\ 
                                                        (not 
                                                          (
                                                            (Heap.select  owner_4_44_28 RES_2_18) (* Ref Eq *) =  null)
                                                        )
                                                      )
                                                    ) -> False
                                                  )
                                                )
                                              )
                                            )
                                          )
                                        )
                                      )
                                    )
                                  )
                                )
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
).


Lemma l1 : (interpProp lemma1 basic_env) <-> lemma1_interp.
Proof with auto.
simpl in *; intros.
unfold lemma1_interp; intros.
split.
intros...
destruct H25...
destruct H25...
intros...
destruct H25...
destruct H25...
assert (h := H objectToBeConstructed  XRES_2_18_2_18  java_lang_Object 
            elems i_3_12
            i_pre_3_12 RES_2_18 RES XRes LS elems_pre 
            state state_pre
            alloc_pre alloc after_2_18_2_18 
            java_lang_Exception ReferenceType owner_4_44_28);
clear H.
subst...
destruct H25...
destruct H...
eapply h; eauto...
Focus 9.
clear h.
right; destruct H...
split...
subst.
vm_compute.
cbv delta [interpProp basic_env] beta delta [lemma1] iota beta;
intros.

destruct H24.
cbv delta [eval_path basic_env ecReturn ecThrow null] in H24.
vm_compute.
simpl in *; intros.
destruct H24; subst...
destruct H24...

destruct H24.
cbv delta [eval_path basic_env ecReturn ecThrow null] in H24.
simpl in H24.
assert (eval_path ecReturn
        ((ecReturn, Path_to_S ecReturn) :: nil) = ecReturn).
simpl.
unfold ecReturn.
Check ecReturn.
simpl in H24.
simpl in *; intros.

simpl in *.
subst.
simpl.
trivial.
simpl in H21.
subst.
simpl in H24.

simpl in H23.
simpl in *.
cbv delta [eval_path basic_env ecReturn ecThrow null] in H24.
cbv iota in H24.
cbv beta in H24.
cbv iota in H24.
simpl in H24.
iota beta.
 basic_env ecReturn] iota beta zeta in H24.
cbv delta 
simpl in H24.
compute in H24.
intros.
destruct H24...
subst.
intuition.
Qed.
Lemma eval : forall l p, interp (pair l p) -> interp (simplify l p).
Proof with auto.
intros.
simpl in *.
assumption.
Qed.
(* Lemma vc_Blue__constructor__2_18 : 
forall (objectToBeConstructed : Reference)(XRES_2_18_2_18 : Reference)(java_lang_Object : Types)(i_3_12 : Reference)((* %NotHandled *) brokenObj : Reference)(java_lang_Exception : Types)(elems_pre : Reference)((* %NotHandled *) brokenObj : Reference)(after_2_18_2_18 : Time)(ReferenceType : Types)(owner_4_44_28 : Reference)(state_pre : Reference)(RES_2_18 : Reference)(XRes : Reference)(LS : Reference)(RES : Reference)(alloc_pre : Time)(elems : Reference)((* %NotHandled *) brokenObj : Reference)(state : Reference)(alloc : Time)(i_pre_3_12 : Reference)(EC_2_18_2_18 : Path) , *)
