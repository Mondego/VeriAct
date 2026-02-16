// Test no superclass listed:
public class JavaLang {}

// Test java.lang.Object explicitly as our parent:
class ObjectChild extends java.lang.Object {}

// Test java.lang.Object explicitly as our parent, using short name:
class ObjectChild2 extends Object {}

// Test java.lang.Exception explicitly as our parent:
class ExceptionChild extends java.lang.Exception {}

// Test java.lang.Exception explicitly as our parent, using short name:
class ExceptionChild2 extends Exception {}
