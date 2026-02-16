//#FLAGS: -quiet
import java.util.*;
public class TestMap extends LocalTestCase {

    public Map map;
    public Object o;
    public Map.Entry e,ee;

    public void setUp() {
        map = new HashMap();
    } //@ nowarn;
    public void tearDown() {
        map = null;
    }

    //@ requires map != null && map.isEmpty();
    public void testEmpty() {
      try {
        assertTrue (map.isEmpty());
        assertTrue ( map.get(o) == null);
        Object oo = map.get(o);
        assertTrue ( oo == null);
        //@ assert (\forall Object o; !map.containsKey(o));
        //@ assert (\forall Object o; !map.containsValue(o));
        assertTrue( !map.containsKey(null) );
        assertTrue( !map.containsValue(null) );
        //@ assert !map.content.hasMapObject(o);
        //@ assert !map.content.hasMap(o);
//@ assert false;  // TEST FOR CONSISTENCY
      } catch (Exception e) {}
//@ assert false;  // TEST FOR CONSISTENCY
    }

    //@ requires map != null && map.isEmpty();
    public void test1() {
      try {
        assertTrue (map.size() == 0);
        Object o1 = new Object();
        Object o2 = new Object();
        assertTrue (map.isEmpty());
        assertTrue ( map.get(o) == null);
        //@ assert (\forall Object o; !map.content.hasMapObject(o));
        map.put(o1,o2);
        //@ assert map.get(o1) == o2;
        assertTrue (map.size() != 0);
        assertTrue (map.containsKey(o1));
        assertTrue (map.containsValue(o2));
        //@ assert (\forall Object o; map.content.hasMapObject(o) ==> o1 == o);
        //@ assume !o1.equals(o);
        //@ assume !o.equals(o1);
        //@ assert (\forall Object oo; map.containsKey(oo) ==> oo.equals(o1));  // FIXME
        assertTrue ( map.get(o) == null); // FIXME
        assertTrue (map.get(o1) == o2);

        map.remove(o1);
        assertTrue ( map.get(o1) == null);
//@ assert false;  // TEST FOR CONSISTENCY
      } catch (Exception e) {}
 //@ assert false;  // TEST FOR CONSISTENCY
    }

    //@ requires map != null && map.isEmpty();
    public void test2() {
        assertTrue (map.size() == 0);
        Object o1 = new Object();
        Object o2 = new Object();
        assertTrue ( map.get(o1) == null);
        Object o3 = map.put(o1,o2);
        assertTrue ( map.get(o1) == o2);

        assertTrue ( o3 == null);
        o3 = map.put(o1,o1);
        assertTrue ( o3 == o2);
        //@ assert map.get(o1) == o1;

//@ assert false;  // TEST FOR CONSISTENCY
    }


    //@ requires map != null;
    public void testClear() {

        map.clear();
        assertTrue( map.isEmpty() );
        assertTrue (map.size() == 0);
        assertTrue ( map.get(o) == null);
//@ assert false;  // TEST FOR CONSISTENCY
         
    }

    //@ requires map != null;
    public void testString() {
        String sone = "one";
        String stwo = "two";
        Integer one = new Integer(1);
        Integer two = new Integer(2);
        //@ assert !sone.equals(stwo);
        map.put(sone,one);
        assertTrue ( map.get(sone) == one);

        Object o = map.get(sone);
        assertTrue ( o == one);

        map.put(stwo,two);
        //@ assert map.get(stwo) == two;
//@ assert false;  // TEST FOR CONSISTENCY

    }
    //@ requires map != null && map.isEmpty();
    public void testStringE() {
        String sone = "one";
        String stwo = "two";
        Integer one = new Integer(1);
        Integer two = new Integer(2);
        //@ assert !sone.equals(stwo);
        map.put(sone,one);
        assertTrue ( map.get(sone) == one);

        Object o = map.get(sone);
        assertTrue ( o == one);
        //@ assert map.content.mapsObject(sone) == one;

        map.put(stwo,two);
        //@ assert map.get(stwo) == two;
        //@ assert map.content.mapsObject(sone) == one;  // FIXME
        //@ assert map.get(sone) == one;    // FIXME
        o = map.get(sone);
        assertTrue ( o == one);
        //@ assert map.get(stwo) == two;
        //@ assert map.get(sone) == one;
//@ assert false;  // TEST FOR CONSISTENCY

    }

    //@ requires map != null;
    public void testString2() {
        Integer one = new Integer(1);

        map.put("one",one);
        assertTrue ( map.get("one") == one);
        o = map.get("one");
        assertTrue ( o == one);

        Integer two = new Integer(2);
        map.put("two",two);
        //@ assert map.get("two") == two;
        //@ assert map.get("one") == one;   // FIXME
        o = map.get("one");
        assertTrue ( o == one);
        //@ assert map.get("two") == two;
        //@ assert map.get("one") == one;
        o = map.get("two");
        assertTrue (o == two);

//@ assert false;  // TEST FOR CONSISTENCY
    }


    //@ requires e != null;
    //@ requires ee != null;
    //@ requires e != ee;
    //@ signals_only RuntimeException;
    public void NoExec_testEntry() {
        //@ assume e.equalsObjects(ee);
        //@ assert e.equals(e);
        Object key = e.getKey();
        Object value = e.getValue();
        //@ assert Map.nullequals(e.getKey(),ee.getKey());
        //@ assert Map.nullequals(e.getValue(),ee.getValue());
        Object ov = e.setValue(o);
        //@ assert ov == value;
        //@ assert key == e.getKey();
        //@ assert Map.nullequals(value,o)  ==> e.equals( ee );
        //@ assert Map.nullequals(value,o) <==  e.equals( ee );
//@ assert false;  // TEST FOR CONSISTENCY
    }

    //@ requires map != null;
    public void testPutAll() {
        try {
          map.putAll(null);
          assertTrue (false);
        } catch (Exception e) {
          assertTrue (e instanceof NullPointerException);
       }
//@ assert false;  // TEST FOR CONSISTENCY
    }
}

// FIXME - more tests: return value of remove, put; more tests of putAll
// FIXME - tests of keySet, values, entrySet, equals, hashCode, clone
