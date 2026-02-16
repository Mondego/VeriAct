package escjava.tc;

import java.util.Iterator;
import java.util.List;
import java.util.LinkedList;
import java.util.HashMap;
import java.util.Map;
import javafe.ast.FieldDecl;
import javafe.ast.TypeDecl;
import javafe.ast.FieldAccess;
import javafe.ast.ASTDecoration;
import javafe.ast.Expr;
import javafe.ast.TypeNameVec;
import javafe.ast.TypeName;

public class Datagroups {

    /* Each TypeDecl has a decoration associated with it that contains a
       Map; this map has key type FieldDecl and value type List.  The
       FieldDecl is the model variable that has a datagroup and the List is
       the set of Expr instances in the datagroup.

       The reason that the same fd has different datagroups depending on the
       TypeDecl is that the members of a datagroup are those fields mapped into
       it in a TypeDecl and its supertypes, but not those in other branches of
       the inheritance tree.
     */

    static private final /*@ non_null */ ASTDecoration datagroups = 
                                          new ASTDecoration("datagroups");
    
    static final private /*@ non_null */ List empty = new LinkedList();
    
    /** Get the items that are in the datagroup for fd */
    //@ requires fd != null;
    static public List get(TypeDecl td, FieldDecl fd) {
        Map map = (Map)datagroups.get(td);
        if (map == null) map = initMap(td);
        List list = (List)map.get(fd);
        if (list == null) list = empty;
/*
        System.out.println("DG GET: " + td.id + " " + fd.id + ":");
        Iterator i = list.iterator();
        while (i.hasNext()) System.out.println(">>> " + i.next());
*/
        return list;
    }
    
    /** Add Expr fa to the datagroup for declaration fd in the
        context of type td; the declaration fd may be in a superclass.
        When we fetch a contents of a datagroup, we only fetch the fields that
        are in td or its superclasses and super interfaces. */
    //@ requires fd != null;
    //@ requires td != null;
    static public void add(TypeDecl td, FieldDecl fd, Expr fa) {
        Map map = (Map) datagroups.get(td);
        if (map == null) map = initMap(td);
        List list = (List)map.get(fd);
        if (list == null) {
            list = new LinkedList();
            map.put(fd,list);
        }
        list.add(fa);
//        System.out.println("ADDING TO " + td.id + " " + fd.id + " : " + fa);
    }

    static public Map initMap(TypeDecl td) {
            //System.out.println("INITIALIZING MAP FOR " + td.id);
            // If the TypeDecl does not yet have a datagroup map, we create
            // one and initialize it with all the mappings contained in its
            // super types, since the TypeDecl will see both the datagroup and
            // the field mapped into the datagroup.  (This ignores access
            // visibility).  The supertypes have already been typechecked, so
            // their maps will be complete.
            Map map = new HashMap();
            javafe.tc.TypeSig ts = TypeSig.getSig(td).superClass();
            if (ts != null) {
              //System.out.println("ADD FROM " + ts);
              addToMap(map, getMap(ts.getTypeDecl()));
            }
            TypeNameVec tv = td.superInterfaces;
            for (int i=0; i<tv.size(); ++i) {
                TypeName si = tv.elementAt(i);
                TypeDecl itd = TypeSig.getSig(si).getTypeDecl();
                //System.out.println("ADD FROM " + itd.id);
                addToMap(map, getMap(itd) );
            }
            datagroups.set(td,map);
	    return map;
    }

    //@ ensures \result != null;
    static public Map getMap(TypeDecl td) {
        Map m = (Map)datagroups.get(td);
        if (m == null) m = initMap(td);
        return m;
    }

    //@ requires map != null;
    static public void addToMap(Map map, Map m) {
        Iterator i = m.entrySet().iterator();
        while (i.hasNext()) {
            Map.Entry e = (Map.Entry)i.next();
            List list = (List) map.get( e.getKey() );
            if (list == null) {
                list = new LinkedList();
                map.put( e.getKey(), list);
            }
            list.addAll( (List) e.getValue() );
        //System.out.println("ATM: " + ((FieldDecl)e.getKey()).id);
        //Iterator ii = list.iterator();
        //while (ii.hasNext()) System.out.println(">>> " + ii.next());
        }
    }

}
