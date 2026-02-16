public class RaceWarning {
  //@ monitored
  int i;

  void m() {
    i = 0; // should have a synchronization guard
  }
}
