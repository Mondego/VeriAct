// $Id$

package javafe.ast;

import java.util.HashMap;
import java.util.Map;

import javafe.util.Location;

/** This class is not actually ever an element of an AST.
 *  It derives from ASTNode so that it can use the decoration
 *  capability, hence the abstract methods of ASTNode are
 *  simply implemented with stubs.
 * 
 * @author David R. Cok
 */
public class IdentifierNode extends ASTNode {

    final static private /*@ non_null*/ Map map = new HashMap();
    
    /** The wrapped Identifier */
    public Identifier id;

    /**
     * Creates a IdentifierNode object given an Identifier
     * @param id The Identifier being wrapped
     * @return   The fresh IdentifierNode
     */
    //@ requires id != null;
    //@ ensures \result.id == id;
    //@ ensures \fresh(\result);
    static public IdentifierNode make(Identifier id) {
      IdentifierNode t = (IdentifierNode)map.get(id);
      if (t != null) return t;
      t = new IdentifierNode();
      t.id = id;
      map.put(id,t);
      return t;
    }
    
    //@ public represents startLoc <- Location.NULL;
    public int getStartLoc() { return Location.NULL; }
    public int childCount(){ return 0; }
    public Object childAt(int i) { return null; }
    public int getTag() { return 0; }
    public /*@non_null*/String toString() { return id.toString(); }
    public void accept(/*@ non_null */ Visitor v) { }
    public /*@ non_null */ Object accept(/*@ non_null */ VisitorArgResult v, /*@ nullable */ Object o)
        { return this; }

}
