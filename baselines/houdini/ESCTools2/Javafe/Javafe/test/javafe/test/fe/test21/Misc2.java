package java.lang;

/**
 ** Ensure we don't get confused when we override methods of
 ** java.lang.Object in interfaces.
 **/

class Object {
    protected Object clone() { return null;} ;
}

interface Top {
    Object clone();
}

interface Bottom extends Top {}
