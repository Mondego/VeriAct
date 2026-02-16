class DttfsaDoesntEvenParse3 {
  int x;
  //@ invariant \dttfsa(UndeclaredType[], x); // fails (expecting type)
}
