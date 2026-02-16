/* Copyright Hewlett-Packard, 2002 */

class Error4 {
  // The following is not allowed, because the @-signs must be contiguous.

  /*@@@@@
    @@@@@      requires
    @@ @@@@@@  x
    @@@        instanceof
    @          Error4;   @@@*/
  void r(Object x) {
  }
}
