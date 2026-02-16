package escjava.backpred;
class DefaultUnivBackPred {
  static String s =
    "(PROMPT_OFF)\n" +
    ";----------------------------------------------------------------------\n" +
    "; \"Universal\", or class-independent part, of the background predicate\n" +
    "\n" +
    "; === ESCJ 8: Section 0.4\n" +
    "    \n" +
    "(BG_PUSH (FORALL (m i x) (EQ (select (store m i x) i) x)))\n" +
    "\n" +
    "(BG_PUSH (FORALL (m i j x) \n" +
    "		 (IMPLIES (NEQ i j)\n" +
    "			  (EQ (select (store m i x) j)\n" +
    "			      (select m j)))))\n" +
    "\n" +
    "(BG_PUSH (FORALL (m i j k x)\n" +
    "                 (IMPLIES (OR (< k i) (< j k))\n" +
    "                          (EQ (select (unset m i j) k) (select m k)))))\n" +
    "\n" +
    "; === ESCJ 8: Section 1.1\n" +
    "\n" +
    "(DEFPRED (<: t0 t1))\n" +
    "\n" +
    "; <: reflexive\n" +
    "(BG_PUSH \n" +
    "  (FORALL (t)\n" +
    "    (PATS (<: t t))\n" +
    "    (<: t t)))\n" +
    "\n" +
    "; a special case, for which the above may not fire\n" +
    "\n" +
    "(BG_PUSH (<: |T_java.lang.Object| |T_java.lang.Object|))\n" +
    "\n" +
    "; <: transitive	\n" +
    "(BG_PUSH \n" +
    "  (FORALL (t0 t1 t2)\n" +
    "    (PATS (MPAT (<: t0 t1) (<: t1 t2)))\n" +
    "    (IMPLIES (AND (<: t0 t1) (<: t1 t2))\n" +
    "      (<: t0 t2))))\n" +
    "\n" +
    ";anti-symmetry\n" +
    "(BG_PUSH\n" +
    " (FORALL\n" +
    "  (t0 t1)\n" +
    "  (PATS (MPAT (<: t0 t1) (<: t1 t0)))\n" +
    "  (IMPLIES (AND (<: t0 t1) (<: t1 t0)) (EQ t0 t1))))\n" +
    "\n" +
    "; primitive types are final\n" +
    "\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_boolean|))\n" +
    "	(IMPLIES (<: t |T_boolean|) (EQ t |T_boolean|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_char|))\n" +
    "	(IMPLIES (<: t |T_char|) (EQ t |T_char|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_byte|))\n" +
    "	(IMPLIES (<: t |T_byte|) (EQ t |T_byte|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_short|))\n" +
    "	(IMPLIES (<: t |T_short|) (EQ t |T_short|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_int|))\n" +
    "	(IMPLIES (<: t |T_int|) (EQ t |T_int|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_long|))\n" +
    "	(IMPLIES (<: t |T_long|) (EQ t |T_long|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_float|))\n" +
    "	(IMPLIES (<: t |T_float|) (EQ t |T_float|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_double|))\n" +
    "	(IMPLIES (<: t |T_double|) (EQ t |T_double|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_bigint|))\n" +
    "	(IMPLIES (<: t |T_bigint|) (EQ t |T_bigint|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_real|))\n" +
    "	(IMPLIES (<: t |T_real|) (EQ t |T_real|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: t |T_void|))\n" +
    "	(IMPLIES (<: t |T_void|) (EQ t |T_void|))))\n" +
    "\n" +
    "; (New as of 12 Dec 2000)\n" +
    "; primitive types have no proper supertypes\n" +
    "\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_boolean| t))\n" +
    "	(IMPLIES (<: |T_boolean| t) (EQ t |T_boolean|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_char| t))\n" +
    "	(IMPLIES (<: |T_char| t) (EQ t |T_char|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_byte| t))\n" +
    "	(IMPLIES (<: |T_byte| t) (EQ t |T_byte|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_short| t))\n" +
    "	(IMPLIES (<: |T_short| t) (EQ t |T_short|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_int| t))\n" +
    "	(IMPLIES (<: |T_int| t) (EQ t |T_int|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_long| t))\n" +
    "	(IMPLIES (<: |T_long| t) (EQ t |T_long|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_float| t))\n" +
    "	(IMPLIES (<: |T_float| t) (EQ t |T_float|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_double| t))\n" +
    "	(IMPLIES (<: |T_double| t) (EQ t |T_double|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_bigint| t))\n" +
    "	(IMPLIES (<: |T_bigint| t) (EQ t |T_bigint|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_real| t))\n" +
    "	(IMPLIES (<: |T_real| t) (EQ t |T_real|))))\n" +
    "(BG_PUSH (FORALL (t) (PATS (<: |T_void| t))\n" +
    "	(IMPLIES (<: |T_void| t) (EQ t |T_void|))))\n" +
    "	\n" +
    "; === ESCJ 8: Section 1.2\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (t0 t1 t2)\n" +
    "    (PATS (<: t0 (asChild t1 t2)))\n" +
    "    (IMPLIES\n" +
    "      (<: t0 (asChild t1 t2))\n" +
    "      (EQ (classDown t2 t0) (asChild t1 t2)))))\n" +
    "\n" +
    "; === ESCJ 8: Section 1.3\n" +
    "    \n" +
    "; new\n" +
    "\n" +
    "(BG_PUSH \n" +
    "  (<: |T_java.lang.Cloneable| |T_java.lang.Object|))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (t)\n" +
    "    (PATS (|_array| t))\n" +
    "    (<: (|_array| t) |T_java.lang.Cloneable|)))\n" +
    "    \n" +
    "(BG_PUSH\n" +
    "  (FORALL (t)\n" +
    "    (PATS (elemtype (|_array| t)))\n" +
    "    (EQ (elemtype (|_array| t)) t)))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (t0 t1) \n" +
    "    (PATS (<: t0 (|_array| t1)))\n" +
    "    (IFF (<: t0 (|_array| t1))\n" +
    "      (AND\n" +
    "	(EQ t0 (|_array| (elemtype t0)))\n" +
    "	(<: (elemtype t0) t1)))))\n" +
    "\n" +
    "; === ESCJ 8: Section 2.1\n" +
    "\n" +
    "(DEFPRED (is x t))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x t)\n" +
    "	 (PATS (cast x t))\n" +
    "	 (is (cast x t) t)))\n" +
    "     \n" +
    "(BG_PUSH\n" +
    " (FORALL (x t)\n" +
    "	 (PATS (cast x t))\n" +
    "	 (IMPLIES (is x t) (EQ (cast x t) x))))\n" +
    "     \n" +
    "; === ESCJ 8: Section 2.2\n" +
    "\n" +
    "(BG_PUSH (DISTINCT |bool$false| |@true|))\n" +
    "\n" +
    "; === ESCJ 8: Section 2.2.1\n" +
    "\n" +
    "(BG_PUSH (FORALL (x) \n" +
    "	   (PATS (is x |T_char|)) \n" +
    "	   (IFF (is x |T_char|) (AND (<= 0 x) (<= x 65535)))))\n" +
    "(BG_PUSH (FORALL (x)\n" +
    "	   (PATS (is x |T_byte|))\n" +
    "	   (IFF (is x |T_byte|) (AND (<= -128 x) (<= x 127)))))\n" +
    "(BG_PUSH (FORALL (x) \n" +
    "	   (PATS (is x |T_short|))\n" +
    "	   (IFF (is x |T_short|) (AND (<= -32768 x) (<= x 32767)))))\n" +
    "(BG_PUSH (FORALL (x) \n" +
    "	   (PATS (is x |T_int|))\n" +
    "	   (IFF (is x |T_int|) (AND (<= intFirst x) (<= x intLast)))))\n" +
    "(BG_PUSH (FORALL (x) \n" +
    "	   (PATS (is x |T_long|))\n" +
    "	   (IFF (is x |T_long|) (AND (<= longFirst x) (<= x longLast)))))\n" +
    "\n" +
    "(BG_PUSH (< longFirst intFirst))\n" +
    "(BG_PUSH (< intFirst -1000000))\n" +
    "(BG_PUSH (< 1000000 intLast))\n" +
    "(BG_PUSH (< intLast longLast))\n" +
    "\n" +
    "; == Defining bigint\n" +
    "(BG_PUSH (EQ |T_bigint| |T_long|))  ; FIXME - change this eventually\n" +
    "\n" +
    "; === Define typeof for primitive types - DRCok\n" +
    "; Removed because numeric values can be more than one type -\n" +
    "; e.g. is(0,|T_byte|) and is(0,|T_int|) are both true\n" +
    "; Thus these axioms introduce an inconsistency.\n" +
    "; Pointed out by Michal Moskal 2/2006.\n" +
    "\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   (IFF (is x |T_int|) (EQ (typeof x) |T_int|))\n" +
    ";    ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_short|))\n" +
    ";	   (IFF (is x |T_short|) (EQ (typeof x) |T_short|))\n" +
    ";    ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_long|))\n" +
    ";	   (IFF (is x |T_long|) (EQ (typeof x) |T_long|))\n" +
    ";     ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_byte|))\n" +
    ";	   (IFF (is x |T_byte|) (EQ (typeof x) |T_byte|))\n" +
    ";     ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_char|))\n" +
    ";	   (IFF (is x |T_char|) (EQ (typeof x) |T_char|))\n" +
    ";     ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_boolean|))\n" +
    ";	   (IFF (is x |T_boolean|) (EQ (typeof x) |T_boolean|))\n" +
    ";     ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_double|))\n" +
    ";	   (IFF (is x |T_double|) (EQ (typeof x) |T_double|))\n" +
    ";     ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_float|))\n" +
    ";	   (IFF (is x |T_float|) (EQ (typeof x) |T_float|))\n" +
    ";     ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_real|))\n" +
    ";	   (IFF (is x |T_real|) (EQ (typeof x) |T_real|))\n" +
    ";     ))\n" +
    ";(BG_PUSH (FORALL (x) \n" +
    ";	   (PATS (typeof x))\n" +
    ";	   ;(PATS (is x |T_bigint|))\n" +
    ";	   (IFF (is x |T_bigint|) (EQ (typeof x) |T_bigint|))\n" +
    ";     ))\n" +
    "\n" +
    ";; Not sure why (or if) this should be here\n" +
    "(BG_PUSH (FORALL (x) \n" +
    "	   (PATS (typeof x))\n" +
    "	   ;(PATS (is x |T_void|))\n" +
    "	   (IFF (is x |T_void|) (EQ (typeof x) |T_void|))\n" +
    "     ))\n" +
    "     \n" +
    "; === ESCJ 8: Section 2.3\n" +
    "      \n" +
    "(BG_PUSH\n" +
    " (FORALL (x t)\n" +
    "	 (PATS (MPAT (<: t |T_java.lang.Object|) (is x t)))\n" +
    "	 (IMPLIES (<: t |T_java.lang.Object|)\n" +
    "		  (IFF (is x t)\n" +
    "		       (OR (EQ x null) (<: (typeof x) t))))))\n" +
    "\n" +
    "; === ESCJ 8: Section 2.4\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (f t x) (PATS (select (asField f t) x))\n" +
    "	 (is (select (asField f t) x) t)))\n" +
    "\n" +
    "; === ESCJ 8: Section 2.5\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (e a i) (PATS (select (select (asElems e) a) i))\n" +
    "	 (is (select (select (asElems e) a) i)\n" +
    "	     (elemtype (typeof a)))))\n" +
    "\n" +
    "; === ESCJ 8: Section 3.0\n" +
    "\n" +
    "(DEFPRED (isAllocated x a0) (< (vAllocTime x) a0))\n" +
    "\n" +
    "; === ESCJ 8: Section 3.1\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x f a0) (PATS (isAllocated (select f x) a0))\n" +
    "	 (IMPLIES (AND (< (fClosedTime f) a0)\n" +
    "		       (isAllocated x a0))\n" +
    "		  (isAllocated (select f x) a0))))\n" +
    "\n" +
    "; === ESCJ 8: Section 3.2\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (a e i a0) (PATS (isAllocated (select (select e a) i) a0))\n" +
    "	 (IMPLIES (AND (< (eClosedTime e) a0)\n" +
    "		       (isAllocated a a0))\n" +
    "		  (isAllocated (select (select e a) i) a0))))\n" +
    " \n" +
    "; === ESCJ 8: Section 4 \n" +
    "\n" +
    "; max(lockset) is in lockset\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (S)\n" +
    "  (PATS (select (asLockSet S) (max (asLockSet S))))\n" +
    "  (EQ\n" +
    "   (select (asLockSet S) (max (asLockSet S)))\n" +
    "   |@true|)))\n" +
    "\n" +
    "; null is in lockset (not in ESCJ 8)\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (S)\n" +
    "  (PATS (asLockSet S))\n" +
    "  (EQ (select (asLockSet S) null) |@true|)))\n" +
    "\n" +
    "(DEFPRED (lockLE x y) (<= x y))\n" +
    "\n" +
    "(DEFPRED (lockLT x y) (< x y))\n" +
    "\n" +
    "; all locks in lockset are below max(lockset) (not in ESCJ 8)\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (S mu)\n" +
    "  (IMPLIES\n" +
    "   (EQ (select (asLockSet S) mu) |@true|)\n" +
    "   (lockLE mu (max (asLockSet S))))))\n" +
    "\n" +
    "; null precedes all objects in locking order (not in ESCJ 8)\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x)\n" +
    "    (PATS (lockLE null x) (lockLT null x) (lockLE x null) (lockLT x null))\n" +
    "    (IMPLIES\n" +
    "      (<: (typeof x) |T_java.lang.Object|)\n" +
    "      (lockLE null x))))\n" +
    "\n" +
    "\n" +
    "; === ESCJ 8: Section 5.0\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (a) \n" +
    "	 (PATS (arrayLength a))\n" +
    "	 (AND (<= 0 (arrayLength a))\n" +
    "	      (is (arrayLength a) |T_int|))))\n" +
    "\n" +
    "(DEFPRED (arrayFresh a a0 b0 e s T v))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (a a0 b0 e n s T v)\n" +
    "    (PATS (arrayFresh a a0 b0 e (arrayShapeMore n s) T v))\n" +
    "    (IFF\n" +
    "      (arrayFresh a a0 b0 e (arrayShapeMore n s) T v)\n" +
    "      (AND\n" +
    "	(<= a0 (vAllocTime a))\n" +
    "	(isAllocated a b0)\n" +
    "	(NEQ a null)\n" +
    "	(EQ (typeof a) T)\n" +
    "	(EQ (arrayLength a) n)\n" +
    "	(FORALL (i)\n" +
    "	  (PATS (select (select e a) i))\n" +
    "	  (AND\n" +
    "	    (arrayFresh (select (select e a) i) a0 b0 e s (elemtype T) v)\n" +
    "	    (EQ (arrayParent (select (select e a) i)) a)\n" +
    "	    (EQ (arrayPosition (select (select e a) i)) i)))))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (a a0 b0 e n T v)\n" +
    "    (PATS (arrayFresh a a0 b0 e (arrayShapeOne n) T v))\n" +
    "    (IFF\n" +
    "      (arrayFresh a a0 b0 e (arrayShapeOne n) T v)\n" +
    "      (AND\n" +
    "	(<= a0 (vAllocTime a))\n" +
    "	(isAllocated a b0)\n" +
    "	(NEQ a null)\n" +
    "	(EQ (typeof a) T)\n" +
    "	(EQ (arrayLength a) n)\n" +
    "	(FORALL (i)\n" +
    "	  (PATS (select (select e a) i))\n" +
    "	  (AND\n" +
    "	    (EQ (select (select e a) i) v)))))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (a0 b0 s T v e A)\n" +
    "    (PATS (arrayFresh A a0 b0 (asElems e) s T v))\n" +
    "    (IMPLIES\n" +
    "      (EQ A (arrayMake a0 b0 s T v))\n" +
    "      (arrayFresh A a0 b0 (asElems e) s T v)\n" +
    "    )\n" +
    "  )\n" +
    ")\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (a0 b0 a1 b1 s1 s2 T1 T2 v)\n" +
    "    (PATS (MPAT (arrayMake a0 b0 s1 T1 v) (arrayMake a1 b1 s2 T2 v) ))\n" +
    "    (IMPLIES\n" +
    "	(EQ (arrayMake a0 b0 s1 T1 v) (arrayMake a1 b1 s2 T2 v))\n" +
    "	(AND (EQ b0 b1) (EQ T1 T2) (EQ s1 s2))\n" +
    ")))\n" +
    "\n" +
    "; === code to ensure that (isNewArray x) ==> x has no invariants\n" +
    "\n" +
    "\n" +
    "; arrayType is distinct from all types with invariants (due to the\n" +
    "; generated type-distinctness axiom)\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (EQ arrayType (asChild arrayType |T_java.lang.Object|)))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "   (FORALL (t)\n" +
    "      (PATS (|_array| t))\n" +
    "      (<: (|_array| t) arrayType)))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (s)\n" +
    "	  (PATS (isNewArray s))\n" +
    "	  (IMPLIES (EQ |@true| (isNewArray s))\n" +
    "		   (<: (typeof s) arrayType))))\n" +
    "\n" +
    "; === ESCJ 8: Section 5.1\n" +
    "; integralMod is the computer mod - the sign of the result is the sign of the\n" +
    "; first operand; it is not the mathematical mod, whose result is always positive\n" +
    "\n" +
    "; j != 0 ==> ((i/j)*j + (i%j) == i\n" +
    "(BG_PUSH\n" +
    " (FORALL (i j) (PATS (integralDiv i j) )\n" +
    " ;(FORALL (i j) (PATS (integralMod i j) (integralDiv i j))\n" +
    "	 (IMPLIES (NOT ( EQ j 0))\n" +
    "	 (EQ (+ (* (integralDiv i j) j) (integralMod i j))\n" +
    "	     i))))\n" +
    "\n" +
    ";; FIXME? - If the above has a trigger of (integralMod i j) and the axiom\n" +
    ";; below is also included, then we get some inconsistency and a failure of\n" +
    ";; test 72\n" +
    "\n" +
    "; (i>=0 && j>0) ==> (i%j >= 0 && i%j < j)\n" +
    "(BG_PUSH\n" +
    " (FORALL (i j) (PATS (integralMod i j))\n" +
    "	 (IMPLIES (AND (< 0 j) (<= 0 i))\n" +
    "		  (AND (<= 0 (integralMod i j))\n" +
    "		       (< (integralMod i j) j)))))\n" +
    "\n" +
    "; (j != 0) ==> (0%j == 0)\n" +
    "(BG_PUSH\n" +
    " (FORALL (i j) (PATS (integralMod i j))\n" +
    "	 (IMPLIES (AND (NOT (EQ j 0)) (EQ i 0))\n" +
    "	  (EQ 0 (integralMod i j)))))\n" +
    "\n" +
    "; (j != 0) ==> ((i>=0) ==> (i%j)>=0 ))\n" +
    "(BG_PUSH\n" +
    " (FORALL (i j) (PATS (integralMod i j))\n" +
    "	 (IMPLIES (NOT (EQ j 0)) \n" +
    "	   (IMPLIES (<= 0 i) (<= 0 (integralMod i j))))))\n" +
    "\n" +
    "; (j != 0) ==> ((i<=0) ==> (i%j)<=0 ))\n" +
    "(BG_PUSH\n" +
    " (FORALL (i j) (PATS (integralMod i j))\n" +
    "  (IMPLIES (NOT (EQ j 0)) \n" +
    "		(IMPLIES (<= i 0) (<= (integralMod i j) 0)))))\n" +
    "\n" +
    "\n" +
    "; Only true for mathMod\n" +
    ";(BG_PUSH\n" +
    "; (FORALL (i j) \n" +
    ";	 (PATS (integralMod (+ i j) j))\n" +
    ";	 (EQ (integralMod (+ i j) j) \n" +
    ";	     (integralMod i j))))\n" +
    ";\n" +
    ";(BG_PUSH\n" +
    "; (FORALL (i j)\n" +
    ";	 (PATS (integralMod (+ j i) j))\n" +
    ";	 (EQ (integralMod (+ j i) j) \n" +
    ";	     (integralMod i j))))\n" +
    "\n" +
    "; to prevent a matching loop\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (* (integralDiv (* x y) y) y))\n" +
    "  (EQ (* (integralDiv (* x y) y) y) (* x y))))\n" +
    "\n" +
    "\n" +
    "; === ESCJ 8: Section 5.2\n" +
    "\n" +
    "(DEFPRED (boolAnd a b)\n" +
    "  (AND\n" +
    "    (EQ a |@true|) \n" +
    "    (EQ b |@true|)))\n" +
    "\n" +
    "(DEFPRED (boolEq a b)\n" +
    "  (IFF\n" +
    "    (EQ a |@true|)\n" +
    "    (EQ b |@true|)))\n" +
    "\n" +
    "(DEFPRED (boolImplies a b)\n" +
    "  (IMPLIES\n" +
    "    (EQ a |@true|)\n" +
    "    (EQ b |@true|)))\n" +
    "    \n" +
    "(DEFPRED (boolNE a b)\n" +
    "  (NOT (IFF\n" +
    "	 (EQ a |@true|)\n" +
    "	 (EQ b |@true|))))\n" +
    "\n" +
    "(DEFPRED (boolNot a)\n" +
    "  (NOT (EQ a |@true|)))\n" +
    "\n" +
    "(DEFPRED (boolOr a b)\n" +
    "  (OR\n" +
    "    (EQ a |@true|)\n" +
    "    (EQ b |@true|)))\n" +
    "\n" +
    "; Axioms related to Strings - DRCok\n" +
    "\n" +
    "(DEFPRED (|interned:| s))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (i k)\n" +
    "    (PATS (|intern:| i k))\n" +
    "    (AND (NEQ (|intern:| i k) null)\n" +
    "         (EQ (typeof (|intern:| i k)) |T_java.lang.String|)\n" +
    "         (isAllocated (|intern:| i k) alloc))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (s)\n" +
    "    (PATS (|interned:| s))\n" +
    "    (is (|interned:| s) |T_boolean|)\n" +
    "  ))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (i ii k kk)\n" +
    "    (PATS (MPAT (|intern:| i k) (|intern:| ii kk)))\n" +
    "    (IFF (EQ (|intern:| i k) (|intern:| ii kk))  (EQ i ii)) ))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (i k)\n" +
    "    (PATS (|intern:| i k))\n" +
    "    (|interned:| (|intern:| i k))\n" +
    "  ))\n" +
    "        \n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y a)\n" +
    "    (PATS (stringCat x y a))\n" +
    "    (AND (NEQ (stringCat x y a) null)\n" +
    "    	 (NOT (isAllocated (stringCat x y a) a))\n" +
    "         (EQ (typeof (stringCat x y a)) |T_java.lang.String|))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y a b)\n" +
    "    (PATS (MPAT (stringCat x y a) (stringCat x y b)))\n" +
    "    (IFF (EQ (stringCat x y a) (stringCat x y b)) (EQ a b))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (a b i j)\n" +
    "    (PATS (MPAT (next a i) (next b j)))\n" +
    "	(IFF (EQ (next a i)(next b j))\n" +
    "		 (AND (EQ a b)(EQ i j)))))\n" +
    "	\n" +
    "(BG_PUSH\n" +
    "  (FORALL (a i) \n" +
    "    (PATS (next a i))\n" +
    "    (< a (next a i))))\n" +
    "    	 \n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y xx yy a b)\n" +
    "     (PATS (MPAT (stringCat x y a) (stringCat xx yy b)))\n" +
    "     (IMPLIES\n" +
    "     	  (EQ (stringCat x y a) (stringCat xx yy b))\n" +
    "     	  (EQ a b))))\n" +
    "     	  \n" +
    "(DEFPRED (stringCatP r x y a0 a1))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (r x y a0 a1)\n" +
    "    (PATS (stringCatP r x y a0 a1))\n" +
    "    (IMPLIES (stringCatP r x y a0 a1)\n" +
    "    		(AND (NEQ r null)\n" +
    "    	 		(isAllocated r a1)\n" +
    "    	 		(NOT (isAllocated r a0))\n" +
    "    	 		(< a0 a1)\n" +
    "         		(EQ (typeof r) |T_java.lang.String|)))))\n" +
    "\n" +
    "; Not in ESCJ8, but should be\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (integralEQ x y))\n" +
    "    (IFF\n" +
    "      (EQ (integralEQ x y) |@true|)\n" +
    "      (EQ x y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (integralGE x y))\n" +
    "    (IFF\n" +
    "      (EQ (integralGE x y) |@true|)\n" +
    "      (>= x y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (integralGT x y))\n" +
    "    (IFF\n" +
    "      (EQ (integralGT x y) |@true|)\n" +
    "      (> x y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (integralLE x y))\n" +
    "    (IFF\n" +
    "      (EQ (integralLE x y) |@true|)\n" +
    "      (<= x y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (integralLT x y))\n" +
    "    (IFF\n" +
    "      (EQ (integralLT x y) |@true|)\n" +
    "      (< x y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (integralNE x y))\n" +
    "    (IFF\n" +
    "      (EQ (integralNE x y) |@true|)\n" +
    "      (NEQ x y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (refEQ x y))\n" +
    "    (IFF\n" +
    "      (EQ (refEQ x y) |@true|)\n" +
    "      (EQ x y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    "  (FORALL (x y)\n" +
    "    (PATS (refNE x y))\n" +
    "    (IFF\n" +
    "      (EQ (refNE x y) |@true|)\n" +
    "      (NEQ x y))))\n" +
    "\n" +
    "; === ESCJ 8: Section 5.3\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "	 (PATS (termConditional |@true| x y))\n" +
    "	 (EQ (termConditional |@true| x y) x)))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (b x y)\n" +
    "	 (PATS (termConditional b x y))\n" +
    "	 (IMPLIES (NEQ b |@true|)\n" +
    "		  (EQ (termConditional b x y) y))))\n" +
    "\n" +
    "; === Implementation of nonnullelements; not in ESCJ 8 (yet?):\n" +
    "\n" +
    "(DEFPRED (nonnullelements x e)\n" +
    "   (AND (NEQ x null)\n" +
    "	(FORALL (i)\n" +
    "	   (IMPLIES (AND (<= 0 i)\n" +
    "			 (< i (arrayLength x)))\n" +
    "		    (NEQ (select (select e x) i) null)))))\n" +
    "\n" +
    "\n" +
    "; === Axioms about classLiteral; not in ESCJ 8 (yet?):\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (t)\n" +
    "	 (PATS (classLiteral t))\n" +
    "	 (AND (NEQ (classLiteral t) null)\n" +
    "	      (is (classLiteral t) |T_java.lang.Class|)\n" +
    "              (isAllocated (classLiteral t) alloc))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (t)\n" +
    "	 (PATS (classLiteral t))\n" +
    "	 (EQ (classLiteral t) t)\n" +
    "))\n" +
    "\n" +
    "; === Axioms about properties of integral &, |, and /\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralAnd x y))\n" +
    "  (IMPLIES\n" +
    "   (OR (<= 0 x) (<= 0 y))\n" +
    "   (<= 0 (integralAnd x y)))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralAnd x y))\n" +
    "  (IMPLIES\n" +
    "   (<= 0 x)\n" +
    "   (<= (integralAnd x y) x))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralAnd x y))\n" +
    "  (IMPLIES\n" +
    "   (<= 0 y)\n" +
    "   (<= (integralAnd x y) y))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralOr x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= 0 x) (<= 0 y))\n" +
    "   (AND (<= x (integralOr x y)) (<= y (integralOr x y))))))\n" +
    "\n" +
    "; ((x >= 0) & (y > 0)) ==> (x/y >= 0 & x >= (x/y) )\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralDiv x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= 0 x) (< 0 y))\n" +
    "   (AND (<= 0 (integralDiv x y)) (<= (integralDiv x y) x)))))\n" +
    "\n" +
    "; ((x <= 0) & (y > 0)) ==> (x/y <= 0 & x <= x/y))\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralDiv x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= x 0) (< 0 y))\n" +
    "   (AND (<= (integralDiv x y) 0) (<= x (integralDiv x y))))))\n" +
    "\n" +
    "; ((x <= z) & (y > 0)) ==> (x/y <= z/y))\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y z)\n" +
    "  (PATS (MPAT (integralDiv z y) (LE x z)))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= x z) (< 0 y))\n" +
    "   (<= (integralDiv x y) (integralDiv z y)))))\n" +
    "\n" +
    "; ( x >= 0 & y < 0) ==> (x/y)<=0\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralDiv x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= 0 x) (< y 0))\n" +
    "   (<= (integralDiv x y) 0))))\n" +
    "\n" +
    "; ( x <= 0 & y < 0 ) ==> (x/y) >= 0\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralDiv x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= x 0) (< y 0))\n" +
    "   (<= 0 (integralDiv x y)))))\n" +
    "\n" +
    "; (x <= 0 & y > 0) ==> (x/y) <= 0\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralDiv x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= x 0) (< 0 y))\n" +
    "   (<= (integralDiv x y) 0))))\n" +
    "\n" +
    "; (x == 0 & y != 0) ==> (x/y)==0\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralDiv x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (EQ 0 x) (NOT (EQ 0 y)))\n" +
    "   (EQ 0 (integralDiv x y)))))\n" +
    "\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (x y)\n" +
    "  (PATS (integralXor x y))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= 0 x) (<= 0 y))\n" +
    "   (<= 0 (integralXor x y)))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (n)\n" +
    "  (PATS (intShiftL 1 n))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= 0 n) (< n 31))\n" +
    "   (<= 1 (intShiftL 1 n)))))\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (FORALL (n)\n" +
    "  (PATS (longShiftL 1 n))\n" +
    "  (IMPLIES\n" +
    "   (AND (<= 0 n) (< n 63))\n" +
    "   (<= 1 (longShiftL 1 n)))))\n" +
    "\n" +
    "; === A few floating point axioms - DRCok\n" +
    ";; FIXME - floatingLT etc are predicates here, but are functions in ESC/java - is that a problem?\n" +
    ";; FIXME - have to include NaN and infinity\n" +
    "\n" +
    ";; Order axioms\n" +
    "(BG_PUSH\n" +
    " (FORALL (d dd)\n" +
    "  (AND\n" +
    "     (OR\n" +
    "	(EQ |@true| (floatingLT d dd))\n" +
    "	(EQ |@true| (floatingEQ d dd))\n" +
    "	(EQ |@true| (floatingGT d dd))\n" +
    "     )\n" +
    "     (IFF (EQ |@true| (floatingLE d dd)) (OR (EQ |@true| (floatingEQ d dd)) (EQ |@true| (floatingLT d dd))))\n" +
    "     (IFF (EQ |@true| (floatingGE d dd)) (OR (EQ |@true| (floatingEQ d dd)) (EQ |@true| (floatingGT d dd))))\n" +
    "     (IFF (EQ |@true| (floatingLT d dd)) (EQ |@true| (floatingGT dd d)))\n" +
    "     (IFF (EQ |@true| (floatingLE d dd)) (EQ |@true| (floatingGE dd d)))\n" +
    "     (NOT (IFF (EQ |@true| (floatingLT d dd)) (EQ |@true| (floatingGE d dd))))\n" +
    "     (NOT (IFF (EQ |@true| (floatingGT d dd)) (EQ |@true| (floatingLE d dd))))\n" +
    "  )))\n" +
    "\n" +
    "\n" +
    ";; floatingNE\n" +
    "(BG_PUSH (FORALL (d dd) (IFF (EQ |@true| (floatingEQ d dd)) (NOT (EQ |@true| (floatingNE d dd)))) ))\n" +
    "\n" +
    ";; floatADD\n" +
    "(BG_PUSH (FORALL (d dd)\n" +
    "  (IMPLIES (EQ |@true| (floatingGT d (floatingNEG dd))) (EQ |@true| (floatingGT (floatingADD d dd) |F_0.0|)))\n" +
    "))\n" +
    "\n" +
    ";; floatMUL\n" +
    ";;(BG_PUSH (FORALL (d dd) (AND\n" +
    "  ;;(IFF (OR (floatingEQ d |F_0.0|) (floatingEQ dd |F_0.0|)) (floatingEQ (floatingMUL d dd) |F_0.0|))\n" +
    "  ;;(IMPLIES (AND (floatingGT d |F_0.0|) (floatingGT dd |F_0.0|)) (floatingGT (floatingMUL d dd) |F_0.0|))\n" +
    "  ;;(IMPLIES (AND (floatingLT d |F_0.0|) (floatingLT dd |F_0.0|)) (floatingGT (floatingMUL d dd) |F_0.0|))\n" +
    "  ;;(IMPLIES (AND (floatingLT d |F_0.0|) (floatingGT dd |F_0.0|)) (floatingLT (floatingMUL d dd) |F_0.0|))\n" +
    "  ;;(IMPLIES (AND (floatingGT d |F_0.0|) (floatingLT dd |F_0.0|)) (floatingLT (floatingMUL d dd) |F_0.0|))\n" +
    ";;)))\n" +
    "\n" +
    ";; floatingMOD\n" +
    "(BG_PUSH\n" +
    " (FORALL (d dd)\n" +
    "  (IMPLIES (EQ |@true| (floatingNE dd |F_0.0|)) (AND\n" +
    "    (IMPLIES (EQ |@true| (floatingGE d |F_0.0|)) (EQ |@true| (floatingGE (floatingMOD d dd) |F_0.0|)))\n" +
    "    (IMPLIES (EQ |@true| (floatingLE d |F_0.0|)) (EQ |@true| (floatingLE (floatingMOD d dd) |F_0.0|)))\n" +
    "))))\n" +
    "(BG_PUSH (FORALL (d dd) \n" +
    "    (IMPLIES (EQ |@true| (floatingGT dd |F_0.0|)) (AND\n" +
    "        (EQ |@true| (floatingGT (floatingMOD d dd) (floatingNEG dd))) \n" +
    "        (EQ |@true| (floatingLT (floatingMOD d dd) dd)) ))\n" +
    "))\n" +
    "(BG_PUSH (FORALL (d dd) \n" +
    "    (IMPLIES (EQ |@true| (floatingLT dd |F_0.0|)) (EQ |@true| (floatingLT (floatingMOD d dd) (floatingNEG dd)))) ))\n" +
    "(BG_PUSH (FORALL (d dd) \n" +
    "    (IMPLIES (EQ |@true| (floatingLT dd |F_0.0|)) (EQ |@true| (floatingGT (floatingMOD d dd) dd))) ))\n" +
    "\n" +
    "; === Temporary kludge to speed up distinguishing small integers:\n" +
    "\n" +
    "(BG_PUSH\n" +
    " (DISTINCT -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9\n" +
    "	   10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29\n" +
    "	   30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49\n" +
    "	   50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69\n" +
    "	   70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89\n" +
    "	   90 91 92 93 94 95 96 97 98 99\n" +
    "	   100 101 102 103 104 105 106 107 108 109\n" +
    "	   110 111 112 113 114 115 116 117 118 119\n" +
    "	   120 121 122 123 124 125 126 127 128 129\n" +
    "	   130 131 132 133 134 135 136 137 138 139\n" +
    "	   140 141 142 143 144 145 146 147 148 149\n" +
    "	   150 151 152 153 154 155 156 157 158 159\n" +
    "	   160 161 162 163 164 165 166 167 168 169\n" +
    "	   170 171 172 173 174 175 176 177 178 179\n" +
    "	   180 181 182 183 184 185 186 187 188 189\n" +
    "	   190 191 192 193 194 195 196 197 198 199\n" +
    "	   200 201 202 203 204 205 206 207 208 209\n" +
    "	   210 211 212 213 214 215 216 217 218 219\n" +
    "	   220 221 222 223 224 225 226 227 228 229\n" +
    "	   230 231 232 233 234 235 236 237 238 239\n" +
    "	   240 241 242 243 244 245 246 247 248 249\n" +
    "	   250 251 252 253 254 255 256 257 258 259\n" +
    "	   260 261 262 263 264 265 266 267 268 269\n" +
    "	   270 271 272 273 274 275 276 277 278 279\n" +
    "	   280 281 282 283 284 285 286 287 288 289\n" +
    "	   290 291 292 293 294 295 296 297 298 299\n" +
    "	   300 301 302 303 304 305 306 307 308 309\n" +
    "	   310 311 312 313 314 315 316 317 318 319\n" +
    "	   320 321 322 323 324 325 326 327 328 329\n" +
    "	   330 331 332 333 334 335 336 337 338 339\n" +
    "	   340 341 342 343 344 345 346 347 348 349\n" +
    "	   350 351 352 353 354 355 356 357 358 359\n" +
    "	   360 361 362 363 364 365 366 367 368 369\n" +
    "	   370 371 372 373 374 375 376 377 378 379\n" +
    "	   380 381 382 383 384 385 386 387 388 389\n" +
    "	   390 391 392 393 394 395 396 397 398 399))\n" +
    "\n" +
    ";----------------------------------------------------------------------\n" +
    "; End of Universal background predicate\n" +
    ";----------------------------------------------------------------------\n" +
    "(PROMPT_ON)\n" +
    "";
}
