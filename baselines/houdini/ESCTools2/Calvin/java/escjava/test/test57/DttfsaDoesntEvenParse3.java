/* Copyright Hewlett-Packard, 2002 */

class DttfsaDoesntEvenParse3 {
  int x;
  //@ invariant \dttfsa(UndeclaredType[], x); // fails (expecting type)
}
