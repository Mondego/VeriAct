/* Copyright Hewlett-Packard, 2002 */

class DttfsaDoesntEvenParse2 {
  int x;
  //@ invariant \dttfsa(x, x); // fails (expecting type)
}
