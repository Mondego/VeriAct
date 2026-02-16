(theory escjava2-object-logic
  :notes "SMT-LIB realization of ESC/Java2's object logic.
          by Cesare Tinelli and Joe Kiniry
          Begun 24 June 2004
          $Revision$
          Based upon SRC ESC/Java object logic
            (design document ESCJ8a)"

  :sorts ( # sort that represents *values* of Java's boolean base type
           Boolean
           # sort that represents *values* of all Java's base types
           # but for Boolean
           Number
           # sort that represents all Java non-base types
           ReferenceType
           # ... represents object references
           Reference
           # ... represents object values
           Object
           # Boolean, Number, Object fields
           BooleanField
           NumberField
           ReferenceField
           # ... represents the heap
           Memory )

  # If we had subsorts, we would probably like to introduce the following:
  #   Time < Number
  #   Array < Object
  #   Boolean, Number, Reference < JavaType

  :funs ( # The heap.
          (HEAP    Memory)
          
          # Top of the object hierarchy.
          (java.lang.Object ReferenceType)
          
          (0       Number)
          (1       Number)
          
          (java.lang.Long.MAX_VALUE    Number)
          (java.lang.Long.MIN_VALUE    Number)
          (java.lang.Integer.MAX_VALUE Number)
          (java.lang.Integer.MIN_VALUE Number)
          (java.lang.Short.MAX_VALUE   Number)
          (java.lang.Short.MIN_VALUE   Number)
          (java.lang.Byte.MAX_VALUE    Number)
          (java.lang.Byte.MIN_VALUE    Number)
          (java.lang.Char.MAX_VALUE    Number)
          (java.lang.Char.MIN_VALUE    Number)

          (java.lang.Boolean.TRUE Boolean)
          (java.lang.Boolean.FALSE Boolean)

          # numeric downcasting (truncation)
          (narrowDouble2Float Number Number)
          (narrowDouble2Long  Number Number)
          (narrowDouble2Int   Number Number)

          (narrowFloat2Long  Number Number)
          (narrowFloat2Int   Number Number)

          (narrowLong2Int   Number Number)
          (narrowLong2Short Number Number)
          (narrowLong2Byte  Number Number)
          (narrowLong2Char  Number Number)

          (narrowInt2Short Number Number)
          (narrowInt2Byte  Number Number)
          (narrowInt2Char  Number Number)

          (narrowShort2Byte  Number Number)
          (narrowShort2Char  Number Number)

          (castByte2Char  Number Number)
          (castChar2Byte  Number Number)

          # 
          (NULL     Reference)
          (+        Number Number Number)
          (-        Number Number Number)
          ...etc...

          (!        Boolean Boolean)
          (&&       Boolean Boolean Boolean)
          (||       Boolean Boolean Boolean)
          ...etc...

          # Java ternary ?: operator
          (?:Boolean     Boolean Boolean   Boolean)
          (?:Number      Boolean Number    Number)
          (?:Object      Boolean Reference Reference)

          # array type constructors
          (arrayBoolean   ReferenceType)
          (arrayNumber    ReferenceType)
          (arrayReference ReferenceType ReferenceType)
          
          # \elementtype of object array
          (elementReferenceType ReferenceType ReferenceType)

          # dynamic type of a reference
          (typeOf Reference ReferenceType)

          # Java typecast
          (cast   Reference ReferenceType Reference)

          # TODO: perhaps rename getX and xGet
          
          # Get/set a value from/at a specific index in an
          # array of numbers.
          (getNumber Object Number Number)
          (setNumber Object Number Number Object)
          # Get/set a value from/at a specific index in an
          # array of booleans.
          (getBoolean Object Number Boolean)
          (setBoolean Object Number Boolean Object)
          # Get/set a value from/at a specific index in an
          # array of objects.
          (getObject Object Number Reference)
          (setObject Object Number Reference Object)

          # Get/set objects in the memory (heap).
          (memGet Memory Reference Object)
          (memSet Memory Reference Object Memory)

          # Get/set boolean fields of objects.
          (boolSelect   BooleanField Object Boolean)
          (boolStore    BooleanField Object Boolean BooleanField)
          # Get/set number fields of objects.
          (numberSelect NumberField Object Number)
          (numberStore  NumberField Object Number NumberField)
          # Get/set object fields of objects.
          (referenceSelect ReferenceField Object Reference)
          (referenceStore  ReferenceField Object Reference ReferenceField)

          # Allocation time of a reference.
          (vAllocTime  Reference Number :injective)

          # Closure of allocation over objects and arrays.
          (fClosedTime ReferenceField Number)
          (eClosedTime Memory Number)

          # Array length as a first-order construct.
          (arrayLength Reference Number)
         )

  :preds ( # Java type predicates on numbers
           (isChar   Number)
           (isByte   Number)
           (isShort  Number)
           (isInt    Number)
           (isLong   Number)
           (isFloat  Number)
           (isDouble Number)
           
           # type predicate for mathematical integers
           (isZ      Number)

           # Java class and interface inheritance ("extends" and
           # "implements", but only smallest "implements" of a class,
           # not inherited implements)
           (extends ReferenceType ReferenceType)
           # Java subtyping
           (<:      ReferenceType ReferenceType :reflex :trans)
           
           # Cesare says these are built-in, since they are 
           # not easily axiomatized.
           (<  Number Number)
           (<= Number Number)
           ...etc...

           # is-a predicate.  Corresponds to the static type of a
           # Java reference.
           (isa Reference ReferenceType)

           # abstract classes and interfaces are not instantiable
           # in Java
           (instantiable ReferenceType)

           # Is the referenced object allocated at the specified time?
           (isAllocated Reference Number)

           # Allocation of array elements.
           (arrayFresh Reference Memory Number )
      )

  :extensions ()

  :definition "Unlike the logic in ESCJ8a, this is a many-sorted logic.
    Additionally, this is a prover-independent logic.  We only
    assume that the prover provides:
      o a theory of rational arithemetic
      o a theory of arrays
      o a theory of uninterpreted function symbols on which
        equality is defined"

  :axioms ( # extends is antireflexive
            (forall ?x ReferenceType (not (extends ?x ?x)) )
            # subtype includes extends
            (forall ?x ReferenceType
               (forall ?y ReferenceType
                  (implies (extends ?x ?y) (<: ?x ?y))))
            # subtype transitivity
            (forall ?x ReferenceType
               (forall ?y ReferenceType
                  (forall ?z ReferenceType
                     (implies (and (<: ?x ?y)
                                   (<: ?y ?z))
                              (<: ?x ?z)))))
            # subtype reflexivity
            (forall ?x ReferenceType (<: ?x ?x))
            # subtype is the smallest relation that contains extends
            (forall ?x ReferenceType
               (forall ?y ReferenceType
                  (implies (and (<: ?x ?y)
                                (not (= ?x ?y)))
                           (exists ?z ReferenceType
                              (and (extends ?x ?z)
                                   (<: ?z ?y))))))

            # <: is anti-symmetric
            (forall ?x ReferenceType
               (forall ?y ReferenceType
                 (implies (and (<: ?x ?y) (<: ?y ?x))
                          (= (?x ?y)))))

            # java.lang.Object is top of subtype hierarchy
            (forall ?x ReferenceType (<: ?x java.lang.Object))

            # subtype rules for arrays
            (forall ?x ReferenceType
               (forall ?y ReferenceType
                 (implies (<: ?x ?y)
                          (<: (array ?x) (array ?y)))))

            # left inverses of Object constructors
            (forall ?x ReferenceType
               (= (elementReferenceType (arrayObject ?x))
                  ?x))

            # integral type boundaries
            (= minByte -128)
            (= maxByte 127)
            (forall ?x Number
               (iff (isByte ?x) (and (<= minByte ?x)
                                     (<= ?x maxByte))))
            ...

            # define an approximation of Z
            (isZ 0)
            (isZ 1)
            (forall ?x Number
              (iff (isZ ?x) (isZ (+ ?x 1))))
            (forall ?x Number
              (implies (isZ ?x)
                (forall ?y Number
                  (implies (and (< ?x ?y)
                                (< ?y (+ ?x 1)))
                           (not (isZ ?y))))))

            # all Object Reference are NULL or have a unique
            # dynamic subtype
            (forall ?x Reference
              (forall ?t ReferenceType 
                (iff (isa ?x ?t)
                     (or (= ?x NULL)
                         (typeOf ?x ?t)))))

            # definition of cast
            (forall ?x Reference
              (forall ?t ReferenceType 
                (isa (cast ?x ?t) ?t)))

            # upcasting a value does not change it
            (forall ?x Reference
              (forall ?t ReferenceType 
                (implies (isa ?x ?t)
                         (= (cast ?x ?t) ?x))))


            # TRUE and FALSE are distinct values
            (distinct TRUE FALSE)

	    # TRUE and FALSE are the only Boolean values
            (forall ?x Boolean (or (= ?x TRUE)
                                   (= ?x FALSE)))

            # definition of Boolean operators
            ...

	    # definition of instantiable type
            (instantiable arrayBoolean)
            (instantiable arrayNumber)
            (instantiable java.lang.Object)
            (forall ?x Reference (instantiable (typeOf ?x)))
            (forall ?x ReferenceType (iff (instantiable ?x)
                                       (instantiable (arrayObject ?x))))

            # to be checked / modified by clement
            (forall ?x ReferenceType
              (forall ?y ReferenceType 
                (implies (and (subtype ?x ?y)
                              (instantiable ?y))
                         (instantiable ?x))))
		

            # for simplicity, get* and set* functions are defined
            # on all Object values, whether they are arrays or not

            (forall ?a object
              (for all ?i Number 
                (forall ?x Number
                   (= (getNumber (setNumber ?a ?i ?x) ?i) 
                      ?x))))
             (forall ?a Object 
               (for all ?i Number 
                 (for all ?j Number 
                   (forall ?x Number
                     (or (= ?i ?j)
                         (= (getNumber (setNumber ?a ?i ?x) ?j)
                            (getNumber ?a ?j)))))))

            (forall ?a object
              (for all ?i Number 
                (forall ?x Boolean
                  (= (getBoolean (setBoolean ?a ?i ?x) ?i) 
                     ?x))))
            (forall ?a Object 
              (for all ?i Number 
                (for all ?j Number 
                  (forall ?x Boolean
                    (or (= ?i ?j)
                        (= (getBoolean (setBoolean ?a ?i ?x) ?j)
                           (getBoolean ?a ?j)))))))

            (forall ?a object
              (for all ?i Number 
                (forall ?x Reference
                  (= (getObject (setObject ?a ?i ?x) ?i) 
                     ?x))))
            (forall ?a Object 
              (for all ?i Number 
                (for all ?j Number 
                  (forall ?x Reference
                    (or (= ?i ?j)
                        (= (getObject (setObject ?a ?i ?x) ?j)
                           (getObject ?a ?j)))))))


            (forall ?m Memory
              (for all ?o Reference 
                (forall ?x object
                  (= (memGet (memSet ?m ?o ?x) ?o) 
                     ?x))))
            (forall ?m Memory 
              (for all ?o1 Reference 
                (for all ?o2 Reference 
                  (forall ?x object
                    (or (= ?o1 ?o2)
                        (= (memGet (memSet ?m ?o1 ?x) ?o2)
                           (memGet ?m ?o2)))))))

            # axioms for select
            ...

            # axioms for vAllocTime
            # injectivity axiom of vAllocTime
            (forall ?r1 Reference
               (forall ?r2 Reference
                  (implies (= (vAllocTime ?r1) (vAllocTime ?r2))
                           (= ?r1 ?r2))))

            # definition of isAllocated in terms of vAllocTime
            (forall ?r Reference
              (forall ?t Number
                (iff (isAllocated ?r ?t) (< (vAllocTime ?r ?t) ?t))))

            # definition of fClosedTime.  Intuitively, what we are
            # representing is that, if an object is allocated, then
            # all of its fields are allocated.
            (forall ?h Memory
              (forall ?r Reference
                 (forall ?f ReferenceField
                    (forall ?t Number
                      (implies (and (< (fClosedTime ?f) ?t)
                                    (isAllocated ?r ?t))
                               (isAllocated (referenceSelect ?f
                                               (memGet ?h ?r)) ?t))))))

            # and the same axiom applies to arrays and their values
            (forall ?h Memory
              (forall ?r Reference
                 (forall ?i Number
                    (forall ?t Number
                      (implies (and (< (eClosedTime ?h) ?t)
                                    (isAllocated ?r ?t))
                               (isAllocated (getObject
                                               (memGet ?h ?r) ?i) ?t))))))

            # length of arrays
            (forall ?r Reference
               (let (length (arrayLength ?r))
                 (and (<= 0 length)
                       (isZ length))))

            
  )

end)
