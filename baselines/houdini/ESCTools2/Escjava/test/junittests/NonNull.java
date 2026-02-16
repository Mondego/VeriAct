// $Id$
// Tests the parsing of non_null in model declarations
class non_null {
}

public class NonNull {
	/*@ pure */ void qq(non_null n) {}
	/*@ pure */ void q(/*@ non_null */ Object o) {}
/*@
	model pure void m(int i);
	model pure void m(non_null Object i);
	model pure void m(non_null i);

	model pure NonNull(int i);
	model pure NonNull(non_null Object i);
	model pure NonNull(non_null i);

	model void p() {
		Object o = null;
		m(o);
	}
	model void p2() {
		Object o = null;
		new NonNull(o);
	}
	model void p3() {
		new NonNull(new non_null());
		m(new non_null());
		m(5);
		new NonNull(5);
        }
	model void p4() {
		Object o = null;
		q(o);
	}
*/
	
	void r() {
		Object o = null;
		q(o);
	}
}

class nullable {
}

class Nullable {
    /*@ pure */ void q2(/*@ nullable */ Object o) {}

    /*@
      model pure void m2(int i);
      model pure void m2(nullable Object i);
      model pure void m2(nullable i);

      model pure Nullable(int i);
      model pure Nullable(nullable Object i);
      model pure Nullable(nullable i);

      model void pp() {
        Object o = null;
	m2(o);
      }
        model void pp2() {
        Object o = null;
	new Nullable(o);
      }
      model void pp3() {
        new Nullable(new nullable());
	m2(new nullable());
	m2(5);
	new Nullable(5);
      }
      model void pp4() {
        Object o = null;
	q2(o);
      }
      @*/

    void r2() {
	Object o = null;
	q2(o);
    }

}

