package escjava.vcGeneration;

import java.lang.Class;
import java.io.PrintStream;
import java.util.Vector;
import java.util.Iterator;
import java.lang.StringBuffer;

abstract class TVariable extends TNode {

    /*
     * This method does nothing and is not abstract
     * because some of the subclasses did not need to
     * do something when this method is called.
     */
    protected void typeTree(){};

}
