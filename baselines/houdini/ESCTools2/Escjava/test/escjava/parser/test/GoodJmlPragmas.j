// contract shorthand keywords

/*@ pre true; @*/
/*@ post false; */
//@ pre a_boolean_expression;

// various kinds of frame condition keywords

//@ modifies x;
//@ modifies y, z;
//@ modifies a, b, c;
/*@ assignable x, y; @*/
//@ modifiable z;
/*@ assignable x[0]; modifiable z[0][y]; */

// exceptional postconditions

//@ signals (Exception e) true == false;
/*@ signals (Throwable t) x < y; */
//@ signals (\TYPE tt) tt == \type(SomeException);

// loop (in)variants

/*@ maintaining x; */
//@ decreasing z-1;

// also clauses

//@ also post q;
/*@ also post q2; @*/
//@ also pre x[0];
//@ also modifies z;

// definedness modifiers

//@ writable_if x;
//@ readable_if y > 0;

// heavyweight specifications

//@ behavior
//@ normal_behavior
//@ exceptional_behavior
// abrupt_behavior

// redundant specifications

//@ accessible_redundantly a;
//@ accessible_redundantly a, b, c;
//@ accessible_redundantly \nothing;
//@ accessible_redundantly \everything;
//@ accessible_redundantly \not_specified;
//@ accessible_redundantly \private_data;

// Optional keyword not yet supported.
// see http://sourceforge.net/tracker/index.php?func=detail&aid=729466&group_id=65346&atid=545306
//@ assert_redundantly predicate;
// assert_redundantly predicate : expression;

//@ assignable_redundantly a;
//@ assignable_redundantly a, b, c;


// still need to add store_refs

//@ assume_redundantly predicate;
// assume_redundantly predicate : anExpression;

//@ breaks_redundantly;
//@ breaks_redundantly target_label;
//@ breaks_redundantly target_label predicate;
//@ breaks_redundantly target_label \not_specified;
//@ breaks_redundantly a == b;
//@ breaks_redundantly \not_specified;

//@ callable_redundantly m;
//@ callable_redundantly m1, m2, m3;
//@ callable_redundantly \nothing;
//@ callable_redundantly \everything;
//@ callable_redundantly \not_specified;
//@ callable_redundantly \private_data;

//@ constraint_redundantly predicate;
//@ constraint_redundantly predicate for \everything;
//@ constraint_redundantly predicate for m;
//@ constraint_redundantly predicate for m1, m2, m3;
// need to add full support for param_disambig_list

//@ continues_redundantly;
//@ continues_redundantly target_label;
//@ continues_redundantly target_label predicate;
//@ continues_redundantly target_label \not_specified;
//@ continues_redundantly a == b;
//@ continues_redundantly \not_specified;

//@ decreases_redundantly spec_expression;
//@ decreasing_redundantly spec_expression;

// depends_redundantly store_ref_expression <- store_ref_list;

//@ diverges_redundantly \not_specified;
//@ diverges_redundantly predicate;

//@ duration_redundantly \not_specified;
//@ duration_redundantly spec_expression;
//@ duration_redundantly spec_expression if predicate;

//@ ensures_redundantly \not_specified;
//@ ensures_redundantly predicate;

//@ exsures_redundantly (Exception e) \not_specified;
//@ exsures_redundantly (Exception e) predicate;

//@ hence_by_redundantly predicate;

//@ invariant_redundantly predicate;

//@ loop_invariant_redundantly predicate;

//@ maintaining_redundantly predicate;

//@ measured_by_redundantly \not_specified;
//@ measured_by_redundantly spec_expression;
//@ measured_by_redundantly spec_expression if predicate;

//@ modifiable_redundantly \not_specified;
//@ modifiable_redundantly a;
//@ modifiable_redundantly a[*];
//@ modifiable_redundantly a, b[*], c[1];

//@ modifies_redundantly a;
//@ modifies_redundantly a[*];
//@ modifies_redundantly a, b[*], c[1];



//@ post_redundantly predicate;

//@ pre_redundantly predicate;
//@ represents_redundantly store_ref_expression = spec_expression;
//@ represents_redundantly store_ref_expression <- spec_expression;
//@ represents_redundantly store_ref_expression \such_that predicate;

//@ requires_redundantly predicate;

//@ returns_redundantly \not_specified;
//@ returns_redundantly predicate;

//@ signals_redundantly (Exception) \not_specified;
//@ signals_redundantly (Exception) predicate;
//@ signals_redundantly (Exception e) \not_specified;
//@ signals_redundantly (Exception e) predicate;

//@ when_redundantly \not_specified;
//@ when_redundantly predicate;

//@ working_space_redundantly \not_specified;
//@ working_space_redundantly spec_expression_of_type_int;
//@ working_space_redundantly spec_expression_of_type_int if predicate;

//@ duration \duration(bar());
//@ duration \duration(Foo.bar());
//@ duration \duration(this.foo());
//@ duration \duration(super.foo());
//@ working_space \space(Foo.bar());
//@ working_space \working_space(Foo.bar());

//@ duration \not_specified;
//@ duration spec_expression;
//@ duration spec_expression if predicate;

//@ working_space \not_specified;
//@ working_space spec_expression_of_type_int;
//@ working_space spec_expression_of_type_int if predicate;

/*@ \peer */
/*@ \readonly */
/*@ \rep */

//@ immutable

//@ nullable_by_default
//@ non_null_by_default
//@ obs_pure
//@ nullable

//@ code_java_math
//@ code_safe_math
//@ code_bigint_math
//@ spec_java_math
//@ spec_safe_math
//@ spec_bigint_math
