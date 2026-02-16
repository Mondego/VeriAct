
class ParseErrors {
  void m6() {

    // empty switch is ok

    switch(0) {
    }
    // Switch body must start with a label
    switch(0) {
      int i=0;
    case 0:
      i=1;
    }
  }
}

