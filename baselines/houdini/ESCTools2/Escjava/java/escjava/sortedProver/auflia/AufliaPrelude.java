package escjava.sortedProver.auflia;

import java.util.HashSet;

public class AufliaPrelude
{	
	static public final String prelude =
	"(benchmark esc_java_query \n" +
	":source {  \n" +
	"  ESC/Java2 query. \n" +
	"} \n" +
	":logic AUFLIA \n" +
	":status unknown \n" +
	":difficulty { 5 } \n" +
	":category { industrial } \n" +
	" \n" +
	"; ------------------------------------------------------------------------ \n" +
	"; BG Pred Extra functions/predicates \n" +
	"; ------------------------------------------------------------------------ \n" +
	" \n" +
	":extrafuns (( alloc  Int )) \n" +
	":extrafuns (( array_ Int Int )) \n" +
	":extrafuns (( arrayLength Int Int )) \n" +
	":extrafuns (( arrayMake Int Int Int Int Int Int )) \n" +
	":extrafuns (( arrayParent Int Int )) \n" +
	":extrafuns (( arrayPosition Int Int )) \n" +
	":extrafuns (( arrayShapeMore Int Int Int )) \n" +
	":extrafuns (( arrayShapeOne Int Int )) \n" +
	":extrafuns (( arrayType  Int )) \n" +
	":extrafuns (( asChild Int Int Int )) \n" +
	":extrafuns (( asElems Int Int )) \n" +
	":extrafuns (( asField Int Int Int )) \n" +
	":extrafuns (( asLockSet Int Int )) \n" +
	":extrafuns (( bool_false  Int )) \n" +
	":extrafuns (( cast Int Int Int )) \n" +
	":extrafuns (( classDown Int Int Int )) \n" +
	":extrafuns (( classLiteral Int Int )) \n" +
	":extrafuns (( CONCVARSYM Int Int )) \n" +
	":extrafuns (( eClosedTime Int Int )) \n" +
	":extrafuns (( elems  Int )) \n" +
	":extrafuns (( elemtype Int Int )) \n" +
	":extrafuns (( F_0.0  Int )) \n" +
	":extrafuns (( fClosedTime Int Int )) \n" +
	":extrafuns (( floatingEQ Int Int Int )) \n" +
	":extrafuns (( floatingGE Int Int Int )) \n" +
	":extrafuns (( floatingGT Int Int Int )) \n" +
	":extrafuns (( floatingLE Int Int Int )) \n" +
	":extrafuns (( floatingLT Int Int Int )) \n" +
	":extrafuns (( floatingMOD Int Int Int )) \n" +
	":extrafuns (( floatingDiv Int Int Int )) \n" +
	":extrafuns (( floatingMUL Int Int Int )) \n" +
	":extrafuns (( floatingADD Int Int Int )) \n" +
	":extrafuns (( floatingSUB Int Int Int )) \n" +
	":extrafuns (( floatingNEG Int Int )) \n" +
	":extrafuns (( floatingNE Int Int Int )) \n" +
	":extrafuns (( integralAnd Int Int Int )) \n" +
	":extrafuns (( integralDiv Int Int Int )) \n" +
	":extrafuns (( integralEQ Int Int Int )) \n" +
	":extrafuns (( integralGE Int Int Int )) \n" +
	":extrafuns (( integralGT Int Int Int )) \n" +
	":extrafuns (( integralLE Int Int Int )) \n" +
	":extrafuns (( integralLT Int Int Int )) \n" +
	":extrafuns (( integralMod Int Int Int )) \n" +
	":extrafuns (( integralNE Int Int Int )) \n" +
	":extrafuns (( integralOr Int Int Int )) \n" +
	":extrafuns (( integralXor Int Int Int )) \n" +
	":extrafuns (( intern_ Int Int Int )) \n" +
	":extrafuns (( intFirst  Int )) \n" +
	":extrafuns (( intLast  Int )) \n" +
	":extrafuns (( intShiftL Int Int Int )) \n" +
	":extrapreds (( is Int Int )) \n" +
	":extrapreds (( isNewArray Int )) \n" +
	":extrafuns (( LE Int Int Int )) \n" +
	":extrafuns (( longFirst  Int )) \n" +
	":extrafuns (( longLast  Int )) \n" +
	":extrafuns (( longShiftL Int Int Int )) \n" +
	":extrafuns (( max Int Int )) \n" +
	":extrafuns (( modulo Int Int Int )) \n" +
	":extrafuns (( multiply Int Int Int )) \n" +
	":extrafuns (( next Int Int Int )) \n" +
	":extrafuns (( null  Int )) \n" +
	":extrafuns (( refEQ Int Int Int )) \n" +
	":extrafuns (( refNE Int Int Int )) \n" +
	":extrafuns (( select1 Int Int Int )) \n" +
	":extrafuns (( select2 Int Int Int Int )) \n" +
	":extrafuns (( Smt.false  Int )) \n" +
	":extrafuns (( Smt.true  Int )) \n" +
	":extrafuns (( store1 Int Int Int Int )) \n" +
	":extrafuns (( store2 Int Int Int Int Int )) \n" +
	":extrafuns (( stringCat Int Int Int Int )) \n" +
	":extrafuns (( T_bigint  Int )) \n" +
	":extrafuns (( T_boolean  Int )) \n" +
	":extrafuns (( T_byte  Int )) \n" +
	":extrafuns (( T_char  Int )) \n" +
	":extrafuns (( T_double  Int )) \n" +
	":extrafuns (( T_float  Int )) \n" +
	":extrafuns (( T_int  Int )) \n" +
	":extrafuns (( T_java.lang.Class  Int )) \n" +
	":extrafuns (( T_java.lang.Cloneable  Int )) \n" +
	":extrafuns (( T_java.lang.Object  Int )) \n" +
	":extrafuns (( T_java.lang.String  Int )) \n" +
	":extrafuns (( T_long  Int )) \n" +
	":extrafuns (( T_real  Int )) \n" +
	":extrafuns (( T_short  Int )) \n" +
	":extrafuns (( T_void  Int )) \n" +
	":extrafuns (( typeof Int Int )) \n" +
	":extrafuns (( unset Int Int Int Int )) \n" +
	":extrafuns (( vAllocTime Int Int )) \n" +
	":extrapreds (( arrayFresh Int Int Int Int Int Int Int )) \n" +
	":extrapreds (( boolAnd Int Int )) \n" +
	":extrapreds (( boolEq Int Int )) \n" +
	":extrapreds (( boolImplies Int Int )) \n" +
	":extrapreds (( boolNE Int Int )) \n" +
	":extrapreds (( boolNot Int )) \n" +
	":extrapreds (( boolOr Int Int )) \n" +
	":extrapreds (( interned_ Int )) \n" +
	":extrapreds (( isAllocated Int Int )) \n" +
	":extrapreds (( lockLE Int Int )) \n" +
	":extrapreds (( lockLT Int Int )) \n" +
	":extrapreds (( nonnullelements Int Int )) \n" +
	":extrapreds (( stringCatP Int Int Int Int Int )) \n" +
	":extrapreds (( subtypes Int Int )) \n" +
	" \n" +
	" \n" +
	"; ------------------------------------------------------------------------ \n" +
	"; BG Pred Assumptions \n" +
	"; ------------------------------------------------------------------------ \n" +
	" \n" +
	" \n" +
	":assumption \n" +
	"  (forall (?d Int) (?dd Int) \n" +
	"   (implies (= Smt.true (floatingLT ?dd F_0.0)) (= Smt.true (floatingGT (floatingMOD ?d ?dd) ?dd)))) \n" +
	":assumption \n" +
	"  (forall (?d Int) (?dd Int) \n" +
	"   (implies (= Smt.true (floatingLT ?dd F_0.0)) (= Smt.true (floatingLT (floatingMOD ?d ?dd) (floatingNEG ?dd))))) \n" +
	":assumption \n" +
	"  (forall (?d Int) (?dd Int) \n" +
	"   (implies \n" +
	"    (= Smt.true (floatingGT ?dd F_0.0)) \n" +
	"    (and \n" +
	"     (= Smt.true (floatingGT (floatingMOD ?d ?dd) (floatingNEG ?dd))) \n" +
	"     (= Smt.true (floatingLT (floatingMOD ?d ?dd) ?dd))))) \n" +
	":assumption \n" +
	"  (forall (?d Int) (?dd Int) \n" +
	"   (implies \n" +
	"    (= Smt.true (floatingNE ?dd F_0.0)) \n" +
	"    (and \n" +
	"     (implies (= Smt.true (floatingGE ?d F_0.0)) (= Smt.true (floatingGE (floatingMOD ?d ?dd) F_0.0))) \n" +
	"     (implies (= Smt.true (floatingLE ?d F_0.0)) (= Smt.true (floatingLE (floatingMOD ?d ?dd) F_0.0)))))) \n" +
	":assumption \n" +
	"  (forall (?d Int) (?dd Int) \n" +
	"   (implies (= Smt.true (floatingGT ?d (floatingNEG ?dd))) (= Smt.true (floatingGT (floatingADD ?d ?dd) F_0.0)))) \n" +
	":assumption \n" +
	"  (forall (?d Int) (?dd Int) \n" +
	"   (iff (= Smt.true (floatingEQ ?d ?dd)) (not (= Smt.true (floatingNE ?d ?dd))))) \n" +
	":assumption \n" +
	"  (forall (?d Int) (?dd Int) \n" +
	"   (and \n" +
	"    (or (= Smt.true (floatingLT ?d ?dd)) (= Smt.true (floatingEQ ?d ?dd)) (= Smt.true (floatingGT ?d ?dd))) \n" +
	"    (iff (= Smt.true (floatingLE ?d ?dd)) (or (= Smt.true (floatingEQ ?d ?dd)) (= Smt.true (floatingLT ?d ?dd)))) \n" +
	"    (iff (= Smt.true (floatingGE ?d ?dd)) (or (= Smt.true (floatingEQ ?d ?dd)) (= Smt.true (floatingGT ?d ?dd)))) \n" +
	"    (iff (= Smt.true (floatingLT ?d ?dd)) (= Smt.true (floatingGT ?dd ?d))) \n" +
	"    (iff (= Smt.true (floatingLE ?d ?dd)) (= Smt.true (floatingGE ?dd ?d))) \n" +
	"    (not (iff (= Smt.true (floatingLT ?d ?dd)) (= Smt.true (floatingGE ?d ?dd)))) \n" +
	"    (not (iff (= Smt.true (floatingGT ?d ?dd)) (= Smt.true (floatingLE ?d ?dd)))))) \n" +
	":assumption \n" +
	"  (forall (?n Int) \n" +
	"   (implies (and (<= 0 ?n) (< ?n 63)) (<= 1 (longShiftL 1 ?n))) \n" +
	"   :pat {(longShiftL 1 ?n)}) \n" +
	":assumption \n" +
	"  (forall (?n Int) \n" +
	"   (implies (and (<= 0 ?n) (< ?n 31)) (<= 1 (intShiftL 1 ?n))) \n" +
	"   :pat {(intShiftL 1 ?n)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (<= 0 ?x) (<= 0 ?y)) (<= 0 (integralXor ?x ?y))) \n" +
	"   :pat {(integralXor ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (= 0 ?x) (not (= 0 ?y))) (= 0 (integralDiv ?x ?y))) \n" +
	"   :pat {(integralDiv ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (<= ?x 0) (< 0 ?y)) (<= (integralDiv ?x ?y) 0)) \n" +
	"   :pat {(integralDiv ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (<= ?x 0) (< ?y 0)) (<= 0 (integralDiv ?x ?y))) \n" +
	"   :pat {(integralDiv ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (<= 0 ?x) (< ?y 0)) (<= (integralDiv ?x ?y) 0)) \n" +
	"   :pat {(integralDiv ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) (?z Int) \n" +
	"   (implies (and (<= ?x ?z) (< 0 ?y)) (<= (integralDiv ?x ?y) (integralDiv ?z ?y))) \n" +
	"   :pat {(integralDiv ?z ?y) (LE ?x ?z)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (<= ?x 0) (< 0 ?y)) (and (<= (integralDiv ?x ?y) 0) (<= ?x (integralDiv ?x ?y)))) \n" +
	"   :pat {(integralDiv ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (<= 0 ?x) (< 0 ?y)) (and (<= 0 (integralDiv ?x ?y)) (<= (integralDiv ?x ?y) ?x))) \n" +
	"   :pat {(integralDiv ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (and (<= 0 ?x) (<= 0 ?y)) (and (<= ?x (integralOr ?x ?y)) (<= ?y (integralOr ?x ?y)))) \n" +
	"   :pat {(integralOr ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (<= 0 ?y) (<= (integralAnd ?x ?y) ?y)) \n" +
	"   :pat {(integralAnd ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (<= 0 ?x) (<= (integralAnd ?x ?y) ?x)) \n" +
	"   :pat {(integralAnd ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (implies (or (<= 0 ?x) (<= 0 ?y)) (<= 0 (integralAnd ?x ?y))) \n" +
	"   :pat {(integralAnd ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (= (classLiteral ?t) ?t) \n" +
	"   :pat {(classLiteral ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (and \n" +
	"    (not (= (classLiteral ?t) null)) \n" +
	"    (is (classLiteral ?t) T_java.lang.Class) \n" +
	"    (isAllocated (classLiteral ?t) alloc)) \n" +
	"   :pat {(classLiteral ?t)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?e Int) \n" +
	"   (iff \n" +
	"    (nonnullelements ?x ?e) \n" +
	"    (and \n" +
	"     (not (= ?x null)) \n" +
	"     (forall (?i Int) \n" +
	"      (implies (and (<= 0 ?i) (< ?i (arrayLength ?x))) (not (= (select1 (select1 ?e ?x) ?i) null))))))) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (refNE ?x ?y) Smt.true) (not (= ?x ?y))) \n" +
	"   :pat {(refNE ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (refEQ ?x ?y) Smt.true) (= ?x ?y)) \n" +
	"   :pat {(refEQ ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (integralNE ?x ?y) Smt.true) (not (= ?x ?y))) \n" +
	"   :pat {(integralNE ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (integralLT ?x ?y) Smt.true) (< ?x ?y)) \n" +
	"   :pat {(integralLT ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (integralLE ?x ?y) Smt.true) (<= ?x ?y)) \n" +
	"   :pat {(integralLE ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (integralGT ?x ?y) Smt.true) (> ?x ?y)) \n" +
	"   :pat {(integralGT ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (integralGE ?x ?y) Smt.true) (>= ?x ?y)) \n" +
	"   :pat {(integralGE ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (= (integralEQ ?x ?y) Smt.true) (= ?x ?y)) \n" +
	"   :pat {(integralEQ ?x ?y)}) \n" +
	":assumption \n" +
	"  (forall (?r Int) (?x Int) (?y Int) (?a0 Int) (?a1 Int) \n" +
	"   (implies \n" +
	"    (stringCatP ?r ?x ?y ?a0 ?a1) \n" +
	"    (and \n" +
	"     (not (= ?r null)) \n" +
	"     (isAllocated ?r ?a1) \n" +
	"     (not (isAllocated ?r ?a0)) \n" +
	"     (< ?a0 ?a1) \n" +
	"     (= (typeof ?r) T_java.lang.String))) \n" +
	"   :pat {(stringCatP ?r ?x ?y ?a0 ?a1)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) (?xx Int) (?yy Int) (?a Int) (?b Int) \n" +
	"   (implies (= (stringCat ?x ?y ?a) (stringCat ?xx ?yy ?b)) (= ?a ?b)) \n" +
	"   :pat {(stringCat ?x ?y ?a) (stringCat ?xx ?yy ?b)}) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?i Int) \n" +
	"   (< ?a (next ?a ?i)) \n" +
	"   :pat {(next ?a ?i)}) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?b Int) (?i Int) (?j Int) \n" +
	"   (iff (= (next ?a ?i) (next ?b ?j)) (and (= ?a ?b) (= ?i ?j))) \n" +
	"   :pat {(next ?a ?i) (next ?b ?j)}) \n" +
	":assumption \n" +	
	"  (forall (?x Int) (?y Int) (?a Int) (?b Int) \n" +
	"   (iff (= (stringCat ?x ?y ?a) (stringCat ?x ?y ?b)) (= ?a ?b)) \n" +
	"   :pat {(stringCat ?x ?y ?a) (stringCat ?x ?y ?b)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) (?a Int) \n" +
	"   (and \n" +
	"    (not (= (stringCat ?x ?y ?a) null)) \n" +
	"    (not (isAllocated (stringCat ?x ?y ?a) ?a)) \n" +
	"    (= (typeof (stringCat ?x ?y ?a)) T_java.lang.String)) \n" +
	"   :pat {(stringCat ?x ?y ?a)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?k Int) \n" +
	"   (interned_ (intern_ ?i ?k)) \n" +
	"   :pat {(intern_ ?i ?k)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?ii Int) (?k Int) (?kk Int) \n" +
	"   (iff (= (intern_ ?i ?k) (intern_ ?ii ?kk)) (= ?i ?ii)) \n" +
	"   :pat {(intern_ ?i ?k) (intern_ ?ii ?kk)}) \n" +
	":assumption \n" +
	"  (forall (?s Int) \n" +
	"   (is (ite (interned_ ?s) Smt.true Smt.false) T_boolean) \n" +
	"   :pat {(interned_ ?s)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?k Int) \n" +
	"   (and \n" +
	"    (not (= (intern_ ?i ?k) null)) \n" +
	"    (= (typeof (intern_ ?i ?k)) T_java.lang.String) \n" +
	"    (isAllocated (intern_ ?i ?k) alloc)) \n" +
	"   :pat {(intern_ ?i ?k)}) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?b Int) \n" +
	"   (iff (boolOr ?a ?b) (or (= ?a Smt.true) (= ?b Smt.true)))) \n" +
	":assumption \n" +
	"  (forall (?a Int) \n" +
	"   (iff (boolNot ?a) (not (= ?a Smt.true)))) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?b Int) \n" +
	"   (iff (boolNE ?a ?b) (not (iff (= ?a Smt.true) (= ?b Smt.true))))) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?b Int) \n" +
	"   (iff (boolImplies ?a ?b) (implies (= ?a Smt.true) (= ?b Smt.true)))) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?b Int) \n" +
	"   (iff (boolEq ?a ?b) (iff (= ?a Smt.true) (= ?b Smt.true)))) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?b Int) \n" +
	"   (iff (boolAnd ?a ?b) (and (= ?a Smt.true) (= ?b Smt.true)))) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (= (multiply (integralDiv (multiply ?x ?y) ?y) ?y) (multiply ?x ?y)) \n" +
	"   :pat {(multiply (integralDiv (multiply ?x ?y) ?y) ?y)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?j Int) \n" +
	"   (implies (not (= ?j 0)) (implies (<= ?i 0) (<= (integralMod ?i ?j) 0))) \n" +
	"   :pat {(integralMod ?i ?j)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?j Int) \n" +
	"   (implies (not (= ?j 0)) (implies (<= 0 ?i) (<= 0 (integralMod ?i ?j)))) \n" +
	"   :pat {(integralMod ?i ?j)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?j Int) \n" +
	"   (implies (and (not (= ?j 0)) (= ?i 0)) (= 0 (integralMod ?i ?j))) \n" +
	"   :pat {(integralMod ?i ?j)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?j Int) \n" +
	"   (implies (and (< 0 ?j) (<= 0 ?i)) (and (<= 0 (integralMod ?i ?j)) (< (integralMod ?i ?j) ?j))) \n" +
	"   :pat {(integralMod ?i ?j)}) \n" +
	":assumption \n" +
	"  (forall (?i Int) (?j Int) \n" +
	"   (implies (not (= ?j 0)) (= (+ (multiply (integralDiv ?i ?j) ?j) (integralMod ?i ?j)) ?i)) \n" +
	"   :pat {(integralDiv ?i ?j)}) \n" +
	":assumption \n" +
	"  (forall (?s Int) \n" +
	"   (implies (isNewArray ?s) (subtypes (typeof ?s) arrayType)) \n" +
	"   :pat {(isNewArray ?s)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (subtypes (array_ ?t) arrayType) \n" +
	"   :pat {(array_ ?t)}) \n" +
	":assumption (= arrayType (asChild arrayType T_java.lang.Object)) \n" +
	":assumption \n" +
	"  (forall (?a0 Int) (?b0 Int) (?a1 Int) (?b1 Int) (?s1 Int) (?s2 Int) (?T1 Int) (?T2 Int) (?v Int) \n" +
	"   (implies \n" +
	"    (= (arrayMake ?a0 ?b0 ?s1 ?T1 ?v) (arrayMake ?a1 ?b1 ?s2 ?T2 ?v)) \n" +
	"    (and (= ?b0 ?b1) (= ?T1 ?T2) (= ?s1 ?s2))) \n" +
	"   :pat {(arrayMake ?a0 ?b0 ?s1 ?T1 ?v) (arrayMake ?a1 ?b1 ?s2 ?T2 ?v)}) \n" +
	":assumption \n" +
	"  (forall (?a0 Int) (?b0 Int) (?e Int) (?s Int) (?T Int) (?v Int) \n" +
	"   (arrayFresh (arrayMake ?a0 ?b0 ?s ?T ?v) ?a0 ?b0 elems ?s ?T ?v) \n" +
	"   :pat {(arrayMake ?a0 ?b0 ?s ?T ?v)}) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?a0 Int) (?b0 Int) (?e Int) (?n Int) (?T Int) (?v Int) \n" +
	"   (iff \n" +
	"    (arrayFresh ?a ?a0 ?b0 ?e (arrayShapeOne ?n) ?T ?v) \n" +
	"    (and \n" +
	"     (<= ?a0 (vAllocTime ?a)) \n" +
	"     (isAllocated ?a ?b0) \n" +
	"     (not (= ?a null)) \n" +
	"     (= (typeof ?a) ?T) \n" +
	"     (= (arrayLength ?a) ?n) \n" +
	"     (forall (?i Int) \n" +
	"      (and (= (select1 (select1 ?e ?a) ?i) ?v)) \n" +
	"      :pat {(select1 (select1 ?e ?a) ?i)}))) \n" +
	"   :pat {(arrayFresh ?a ?a0 ?b0 ?e (arrayShapeOne ?n) ?T ?v)}) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?a0 Int) (?b0 Int) (?e Int) (?n Int) (?s Int) (?T Int) (?v Int) \n" +
	"   (iff \n" +
	"    (arrayFresh ?a ?a0 ?b0 ?e (arrayShapeMore ?n ?s) ?T ?v) \n" +
	"    (and \n" +
	"     (<= ?a0 (vAllocTime ?a)) \n" +
	"     (isAllocated ?a ?b0) \n" +
	"     (not (= ?a null)) \n" +
	"     (= (typeof ?a) ?T) \n" +
	"     (= (arrayLength ?a) ?n) \n" +
	"     (forall (?i Int) \n" +
	"      (and \n" +
	"       (arrayFresh (select1 (select1 ?e ?a) ?i) ?a0 ?b0 ?e ?s (elemtype ?T) ?v) \n" +
	"       (= (arrayParent (select1 (select1 ?e ?a) ?i)) ?a) \n" +
	"       (= (arrayPosition (select1 (select1 ?e ?a) ?i)) ?i)) \n" +
	"      :pat {(select1 (select1 ?e ?a) ?i)}))) \n" +
	"   :pat {(arrayFresh ?a ?a0 ?b0 ?e (arrayShapeMore ?n ?s) ?T ?v)}) \n" +
	":assumption \n" +
	"  (forall (?a Int) \n" +
	"   (and (<= 0 (arrayLength ?a)) (is (arrayLength ?a) T_int)) \n" +
	"   :pat {(arrayLength ?a)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) \n" +
	"   (implies (subtypes (typeof ?x) T_java.lang.Object) (lockLE null ?x)) \n" +
	"   :pat {(lockLE null ?x)} \n" +
	"   :pat {(lockLT null ?x)} \n" +
	"   :pat {(lockLE ?x null)} \n" +
	"   :pat {(lockLT ?x null)}) \n" +
	":assumption \n" +
	"  (forall (?S Int) (?mu Int) \n" +
	"   (implies (= (select1 (asLockSet ?S) ?mu) Smt.true) (lockLE ?mu (max (asLockSet ?S))))) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (lockLT ?x ?y) (< ?x ?y))) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?y Int) \n" +
	"   (iff (lockLE ?x ?y) (<= ?x ?y))) \n" +
	":assumption \n" +
	"  (forall (?S Int) \n" +
	"   (= (select1 (asLockSet ?S) null) Smt.true) \n" +
	"   :pat {(asLockSet ?S)}) \n" +
	":assumption \n" +
	"  (forall (?S Int) \n" +
	"   (= (select1 (asLockSet ?S) (max (asLockSet ?S))) Smt.true) \n" +
	"   :pat {(select1 (asLockSet ?S) (max (asLockSet ?S)))}) \n" +
	":assumption \n" +
	"  (forall (?a Int) (?e Int) (?i Int) (?a0 Int) \n" +
	"   (implies (and (< (eClosedTime ?e) ?a0) (isAllocated ?a ?a0)) (isAllocated (select1 (select1 ?e ?a) ?i) ?a0)) \n" +
	"   :pat {(isAllocated (select1 (select1 ?e ?a) ?i) ?a0)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?f Int) (?a0 Int) \n" +
	"   (implies (and (< (fClosedTime ?f) ?a0) (isAllocated ?x ?a0)) (isAllocated (select1 ?f ?x) ?a0)) \n" +
	"   :pat {(isAllocated (select1 ?f ?x) ?a0)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?a0 Int) \n" +
	"   (iff (isAllocated ?x ?a0) (< (vAllocTime ?x) ?a0))) \n" +
	":assumption \n" +
	"  (forall (?e Int) (?a Int) (?i Int) \n" +
	"   (is (select1 (select1 (asElems ?e) ?a) ?i) (elemtype (typeof ?a))) \n" +
	"   :pat {(select1 (select1 (asElems ?e) ?a) ?i)}) \n" +
	":assumption \n" +
	"  (forall (?f Int) (?t Int) (?x Int) \n" +
	"   (is (select1 (asField ?f ?t) ?x) ?t) \n" +
	"   :pat {(select1 (asField ?f ?t) ?x)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?t Int) \n" +
	"   (implies (subtypes ?t T_java.lang.Object) (iff (is ?x ?t) (or (= ?x null) (subtypes (typeof ?x) ?t)))) \n" +
	"   :pat {(subtypes ?t T_java.lang.Object) (is ?x ?t)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) \n" +
	"   (iff (is ?x T_void) (= (typeof ?x) T_void)) \n" +
	"   :pat {(typeof ?x)}) \n" +
	":assumption (= T_bigint T_long) \n" +
	":assumption (< intLast longLast) \n" +
	":assumption (< 1000000 intLast) \n" +
	":assumption (< intFirst (~ 1000000)) \n" +
	":assumption (< longFirst intFirst) \n" +
	":assumption \n" +
	"  (forall (?x Int) \n" +
	"   (iff (is ?x T_long) (and (<= longFirst ?x) (<= ?x longLast))) \n" +
	"   :pat {(is ?x T_long)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) \n" +
	"   (iff (is ?x T_int) (and (<= intFirst ?x) (<= ?x intLast))) \n" +
	"   :pat {(is ?x T_int)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) \n" +
	"   (iff (is ?x T_short) (and (<= (~ 32768) ?x) (<= ?x 32767))) \n" +
	"   :pat {(is ?x T_short)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) \n" +
	"   (iff (is ?x T_byte) (and (<= (~ 128) ?x) (<= ?x 127))) \n" +
	"   :pat {(is ?x T_byte)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) \n" +
	"   (iff (is ?x T_char) (and (<= 0 ?x) (<= ?x 65535))) \n" +
	"   :pat {(is ?x T_char)}) \n" +
	":assumption (distinct bool_false Smt.true) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?t Int) \n" +
	"   (implies (is ?x ?t) (= (cast ?x ?t) ?x)) \n" +
	"   :pat {(cast ?x ?t)}) \n" +
	":assumption \n" +
	"  (forall (?x Int) (?t Int) \n" +
	"   (is (cast ?x ?t) ?t) \n" +
	"   :pat {(cast ?x ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t0 Int) (?t1 Int) \n" +
	"   (iff (subtypes ?t0 (array_ ?t1)) (and (= ?t0 (array_ (elemtype ?t0))) (subtypes (elemtype ?t0) ?t1))) \n" +
	"   :pat {(subtypes ?t0 (array_ ?t1))}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (= (elemtype (array_ ?t)) ?t) \n" +
	"   :pat {(elemtype (array_ ?t))}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (subtypes (array_ ?t) T_java.lang.Cloneable) \n" +
	"   :pat {(array_ ?t)}) \n" +
	":assumption (subtypes T_java.lang.Cloneable T_java.lang.Object) \n" +
	":assumption \n" +
	"  (forall (?t0 Int) (?t1 Int) (?t2 Int) \n" +
	"   (implies (subtypes ?t0 (asChild ?t1 ?t2)) (= (classDown ?t2 ?t0) (asChild ?t1 ?t2))) \n" +
	"   :pat {(subtypes ?t0 (asChild ?t1 ?t2))}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_void ?t) (= ?t T_void)) \n" +
	"   :pat {(subtypes T_void ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_real ?t) (= ?t T_real)) \n" +
	"   :pat {(subtypes T_real ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_bigint ?t) (= ?t T_bigint)) \n" +
	"   :pat {(subtypes T_bigint ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_double ?t) (= ?t T_double)) \n" +
	"   :pat {(subtypes T_double ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_float ?t) (= ?t T_float)) \n" +
	"   :pat {(subtypes T_float ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_long ?t) (= ?t T_long)) \n" +
	"   :pat {(subtypes T_long ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_int ?t) (= ?t T_int)) \n" +
	"   :pat {(subtypes T_int ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_short ?t) (= ?t T_short)) \n" +
	"   :pat {(subtypes T_short ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_byte ?t) (= ?t T_byte)) \n" +
	"   :pat {(subtypes T_byte ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_char ?t) (= ?t T_char)) \n" +
	"   :pat {(subtypes T_char ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes T_boolean ?t) (= ?t T_boolean)) \n" +
	"   :pat {(subtypes T_boolean ?t)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_void) (= ?t T_void)) \n" +
	"   :pat {(subtypes ?t T_void)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_real) (= ?t T_real)) \n" +
	"   :pat {(subtypes ?t T_real)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_bigint) (= ?t T_bigint)) \n" +
	"   :pat {(subtypes ?t T_bigint)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_double) (= ?t T_double)) \n" +
	"   :pat {(subtypes ?t T_double)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_float) (= ?t T_float)) \n" +
	"   :pat {(subtypes ?t T_float)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_long) (= ?t T_long)) \n" +
	"   :pat {(subtypes ?t T_long)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_int) (= ?t T_int)) \n" +
	"   :pat {(subtypes ?t T_int)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_short) (= ?t T_short)) \n" +
	"   :pat {(subtypes ?t T_short)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_byte) (= ?t T_byte)) \n" +
	"   :pat {(subtypes ?t T_byte)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_char) (= ?t T_char)) \n" +
	"   :pat {(subtypes ?t T_char)}) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (implies (subtypes ?t T_boolean) (= ?t T_boolean)) \n" +
	"   :pat {(subtypes ?t T_boolean)}) \n" +
	":assumption \n" +
	"  (forall (?t0 Int) (?t1 Int) \n" +
	"   (implies (and (subtypes ?t0 ?t1) (subtypes ?t1 ?t0)) (= ?t0 ?t1)) \n" +
	"   :pat {(subtypes ?t0 ?t1) (subtypes ?t1 ?t0)}) \n" +
	":assumption \n" +
	"  (forall (?t0 Int) (?t1 Int) (?t2 Int) \n" +
	"   (implies (and (subtypes ?t0 ?t1) (subtypes ?t1 ?t2)) (subtypes ?t0 ?t2)) \n" +
	"   :pat {(subtypes ?t0 ?t1) (subtypes ?t1 ?t2)}) \n" +
	":assumption (subtypes T_java.lang.Object T_java.lang.Object) \n" +
	":assumption \n" +
	"  (forall (?t Int) \n" +
	"   (subtypes ?t ?t) \n" +
	"   :pat {(subtypes ?t ?t)}) \n" +
	":assumption \n" +
	"  (forall (?m Int) (?i Int) (?j Int) (?k Int) (?x Int) \n" +
	"   (implies (or (< ?k ?i) (< ?j ?k)) (= (select1 (unset ?m ?i ?j) ?k) (select1 ?m ?k)))) \n" +
	":assumption \n" +
	"  (forall (?m Int) (?i Int) (?j Int) (?x Int) \n" +
	"   (implies (not (= ?i ?j)) (= (select1 (store1 ?m ?i ?x) ?j) (select1 ?m ?j)))) \n" +
	":assumption \n" +
	"  (forall (?m Int) (?i Int) (?x Int) \n" +
	"   (= (select1 (store1 ?m ?i ?x) ?i) ?x)) \n" +
	" \n" +
	"; ------------------------------------------------------------------------ \n" +
	"; End of BG Pred\n" +
	"; ======================================================================== \n\n\n";
	
	static final String[] predef = new String[] {
		"alloc", "array_", "arrayLength", "arrayMake", "arrayParent",
		"arrayPosition", "arrayShapeMore", "arrayShapeOne", "arrayType",
		"asChild", "asElems", "asField", "asLockSet", "bool_false", "cast",
		"classDown", "classLiteral", "CONCVARSYM", "eClosedTime",
		"elems", "elemtype", "F_0.0", "fClosedTime", "floatingADD", "floatingEQ",
		"floatingGE", "floatingGT", "floatingLE", "floatingLT", "floatingMOD",
		"floatingNEG", "floatingNE", "integralAnd", "integralDiv", "integralEQ",
		"integralGE", "integralGT", "integralLE", "integralLT", "integralMod",
		"integralNE", "integralOr", "integralXor", "intern_", "intFirst",
		"intLast", "intShiftL", "is", "isNewArray", "LE", "longFirst",
		"longLast", "longShiftL", "max", "modulo", "multiply", "next", "null",
		"refEQ", "refNE", "select1", "select2", "Smt.false", "Smt.true",
		"store1", "store2", "stringCat", "T_bigint", "T_boolean", "T_byte",
		"T_char", "T_double", "T_float", "T_int", "T_java.lang.Class",
		"T_java.lang.Cloneable", "T_java.lang.Object", "T_java.lang.String",
		"T_long", "T_real", "T_short", "T_void", "typeof", "unset",
		"vAllocTime", "arrayFresh", "boolAnd", "boolEq", "boolImplies", "boolNE",
		"boolNot", "boolOr", "interned_", "isAllocated", "lockLE", "lockLT",
		"nonnullelements", "stringCatP", "subtypes",
		"<", "<=", "<:"
	};
	
	static final public HashSet predefined;
	
	static {
		predefined = new HashSet();
		for (int i = 0; i < predef.length; ++i)
			predefined.add(predef[i]);
	}
}
