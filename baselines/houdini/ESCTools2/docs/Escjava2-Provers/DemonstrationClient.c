// $Id$
//
// Demonstration prover client written in C.
//
// This client is the JNI interface to the OCaml prover
// interface.  It is used by its Java sibling,
// DemonstrationClient.java.
//
// author: Joseph Kiniry

#include <stdio.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/callback.h>

value start_prover () {
  CAMLlocal1(f);
  f = *(caml_named_value("start_solver"));
  CAMLreturn(callback(f, Val_unit));
}

value set_prover_resource_flags (value properties) {
  CAMLparam1(properties);
  CAMLlocal1(f);
  f = *(caml_named_value("set_flags"));
  CAMLreturn (callback(f, properties));
}

value signature (value signature) {
  CAMLparam1(signature);
  CAMLlocal1(f);
  f = *(caml_named_value("declaration"));
  CAMLreturn (callback(f, signature));
}

value declare_axiom (value axiom) {
  CAMLparam1(axiom);
  CAMLlocal1(f);
  f = *(caml_named_value("add_axiom"));
  CAMLreturn (callback(f, axiom));
}

value make_assumption (value formula) {
  CAMLparam1(formula);
  CAMLlocal1(f);
  f = *(caml_named_value("add_assertion"));
  CAMLreturn (callback(f, formula));
}

value retract_assumption (value count) {
  CAMLparam1(count);
  CAMLlocal1(f);
  f = *(caml_named_value("backtrack"));
  CAMLreturn (callback(f, count));
}

value is_valid (value formula, value properties) {
  CAMLparam2(formula, properties);
  CAMLlocal1(f);
  f = *(caml_named_value("query"));
  CAMLreturn (callback2(f, formula, properties));
}

value stop_prover () {
  CAMLlocal1(f);
  f = *(caml_named_value("stop_solver"));
  // how do we call an OCaml function that has no arguments?
  CAMLreturn (callback(f, Val_unit));
}

int main(int argc, char** argv) {
  printf("Starting OCaml prover from C.\n");
  start_prover();
  printf("Stopping OCaml prover from C.\n");
  stop_prover();
}

