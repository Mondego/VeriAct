// static inner classes caused a loop in the typechecker at one time.

class B {
  static class C extends B {
  }
}
