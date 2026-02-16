class causeErrors {

  void m65() {

    // Enclosing statement labelled '...' is not a while, do, or for statement
  b:
    switch(0) {
    case 0:
      continue b;
    }

    // Enclosing statement labelled '...' is not a switch, while, do or for statement
    // This is not an error
  z:
    break z;

  }


  void m66() {

    // No enclosing unlabelled while, do, or for statement
    continue;

  }

  void m661() {

    // No enclosing unlabelled switch, while, do, or for statement
    break;

  }

  void m662() {

    int i = 0;

    // Label '...' already used in this scope
  a: 
  a: 

    // Label '...' already used in this scope
  y: 
    { 
    y: 
      i = i;
    }
    
    if (i==1)
    // No enclosing while, do, or for statement labelled '...'
    continue y;

    // No enclosing switch, while, do, or for statement labelled '...'
    break y;

  }

}

interface I {
  //  int i;      // we not longer check that final fields must be initialized
}


/* TO DO:

Ambiguous method invocation

No such ... in type ...

Ambiguous ... in type ...

No ... matching given argument types

*/

