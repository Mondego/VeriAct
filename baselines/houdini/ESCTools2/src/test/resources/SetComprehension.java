// Tests set comprehension parsing
// FIXME - should support SetComprehension and should be JMLObjectSet, JMLValueSet in org.jmlspecs.lang so they are automatically imported
//@ model import org.jmlspecs.models.*;
public class SetComprehension {

	//@ ghost JMLObjectSet oo;
	//@ ghost Object o = new JMLObjectSet { Object ooo| oo.has(ooo)  && ooo.toString() == null};
	//@ ghost Object o2 = new JMLObjectSet { Object o| oo.has(o)  && o.toString() == null};
	//@ ghost boolean b = (new JMLValueSet { Integer i | oo.has(i)  && i.toString() == null}).isEmpty();

}
