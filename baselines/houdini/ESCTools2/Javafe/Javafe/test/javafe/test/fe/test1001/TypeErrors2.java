class causeErrors {

  // This routine is not expected to return a value
  void m4() { return 1; }

  // Returns are not allowed in a static initializer
  static { return; }

  // Cannot convert .../ to return type ...
  int m5() { return false; }

  // This routine must return a value
  int m51() { return; }

  // Keyword super cannot be used in a static context
  static {
    super.foo();
  }
    
  int m6() {
    // Cannot throw values of type ...
    throw new Integer(5);
  }

  int m61() {
    // Exception must be caught by an enclosing try or throws clause
    throw new Exception();
  }

  void m62() {
    char b=0;
    switch( b ) {
      // Non-constant value in switch label
    case b:

      // Case label value (...) not assignable to the switch expression type ...
    case -1:

      // Duplicate case label ... in switch statement
    case 0: 
    case 0:

      // Duplicate default label in switch statement
    default:
    default:
    }

    //Declared type of a catch formal parameter must be a subclass of Throwable
    try {}
    catch (Object o) {}

    // Unexpected array value in initializer
    int i = { 1, 2 };

    int i=0;		// error: duplicate declaration of i
    
    // Attempt to index an non-array value
    int j = i[0];

    // Array index is not an integer
    int[] a = { 1, 2 };
    j = a[false];
    
    // The type in a new expression must be a class
    new I();
    
    // Cannot create an instance of an abstract class
    new AbstractClass();
    
    // Incompatible arguments to the ?: operator
    int k = i==0 ? 0 : false;
    
    // Non-reference type in instanceof operation
    while( i instanceof int );
    
    // Invalid instanceof operation: A value of type ... can never be an instance of ...
    while( i instanceof Object );
    
    // Bad cast from ... to ...
    Object o = (Object)i;
    
    // Invalid types (... and ...) in equality comparison
    boolean b1 = ( i == false );
    
    // Left hand side of assignment operator is not a variable
    false = false;
    
    // Value of type ... operator cannot be assigned to location of type ...
    i = false;
    
    // Incompatible types for assignment operation 
    byte b2=0;
    b2+=1;
    
    // Cannot convert ... to ...
    while( i );
    
    // Cannot convert ... to an integral type
    Double asd = ~0.0;
    
    // Cannot convert ... to a numeric type
    i = +false;

  }
  
}


interface I {

  // Can't use keyword super in an interface
  int i = super.x;

}

abstract class AbstractClass {
}

class testor {

    // Test new expressions with array types:

    void foo() {

	I[] test1 = new I[10];

	AbstractClass[] test2 = new AbstractClass[10];

	Object[] test3 = new Object[10];

    }
}
