import java.util.BitSet;

public class TestBitSet extends LocalTestCase {

  public void testConstructor() {
    BitSet b = new BitSet();
    //assertTrue ( b.isEmpty() );
    //assertTrue ( b.length() == 0 );
    //assertTrue ( b.size() >= 0 );
    b.set(1000);
    b = new BitSet(100);
    assertTrue ( b.isEmpty() );
    assertTrue ( b.length() == 0 );
    assertTrue ( b.size() >= 0 );
    b.set(1000);
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testSet() {
    BitSet b = new BitSet();
    b.set(10);
    assertTrue( b.get(10) );
    assertTrue( b.length() == 11);
    assertTrue( !b.isEmpty() );
    b.flip(10);
    assertTrue( !b.get(10) );
    assertTrue( b.length() == 0);
    assertTrue( b.isEmpty() );
    b.set(5,true);
    assertTrue( b.get(5) );
    assertTrue( b.length() == 6);
    assertTrue( !b.isEmpty() );
    b.clear(5);
    assertTrue( !b.get(5) );
    assertTrue( b.length() == 0);
    b.set(5,true);
    b.clear();
    assertTrue( b.length() == 0);
    assertTrue( b.isEmpty() );

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testGroup() {
    BitSet b = new BitSet();
    b.set(10,20);
    assertTrue( b.get(10) );
    assertTrue( !b.get(20) );
    b.flip(6,15);
    assertTrue( b.get(6) );
    assertTrue( !b.get(10) );
    assertTrue( !b.get(14) );
    assertTrue( b.get(15) );
    assertTrue( !b.get(20) );
    b.clear(12,25);
    assertTrue( b.get(6) );
    assertTrue( b.get(9) );
    assertTrue( !b.get(10) );
    assertTrue( !b.get(15) );

    b.set(5,20,false);
//@ assert !b.get(19);  // Lemma 
                        // The above seems obvious given the statement above it,
                        // but it must trigger other deductions that enable the 
                        // following to be proved - FIXME
    assertTrue( b.isEmpty() );

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testToString() {
    BitSet b = new BitSet();
    String s = b.toString();
    assertTrueNP ( s.equals("{}") );
    //@ assert s.charAt(0) == '{';
    //@ assert s.charAt(s.length()-1) == '}';
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testClone() {
    BitSet b = new BitSet();
    b.set(10,20);
    BitSet bb = (BitSet)b.clone();
    assertTrue (b.equals(bb));
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testAnd() {
    BitSet b = new BitSet();
    BitSet bb = new BitSet();
    b.set(10,20);
    bb.set(5,15);
    b.and(bb);
    assertTrue( !b.get(9) ); 
    assertTrue( b.get(10) ); 
    assertTrue( b.get(14) ); 
    assertTrue( !b.get(15) ); 
    bb.clear();
    bb.set(7,13);
    b.or(bb);
    assertTrue( !b.get(6) ); 
    assertTrue( b.get(7) ); 
    assertTrue( b.get(14) ); 
    assertTrue( !b.get(15) ); 
    bb.clear();
    bb.set(5,10);
    b.andNot(bb);
    assertTrue( !b.get(9) ); 
    assertTrue( b.get(10) ); 
    assertTrue( b.get(14) ); 
    assertTrue( !b.get(15) ); 
    bb.clear();
    bb.set(13,16);
//@ assert b.get(13) && bb.get(13); // Need this as a lemma
    b.xor(bb);
    assertTrue( !b.get(9) ); 
    assertTrue( b.get(10) ); 
    assertTrue( b.get(12) ); 
    assertTrue( !b.get(13) ); 
    assertTrue( b.get(15) ); 
    assertTrue( !b.get(16) ); 
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testAnd2() {
    BitSet b = new BitSet();
    BitSet bb = new BitSet();
    b.set(10,20);
    bb.set(10,20);
    bb.and(bb);
    assertTrue ( bb.equals(b) );
    bb.or(bb);
    assertTrue ( bb.equals(b) );
    bb.xor(bb);
    assertTrue ( bb.isEmpty() );
    b.andNot(b);
    assertTrue ( b.isEmpty() );

    

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testIntersects() {
    BitSet b = new BitSet();
    BitSet bb = new BitSet();
    b.set(10,20);
    bb.set(5,15);
//@ assert b.get(10) && bb.get(10); // Need this as a lemma
    assertTrue( b.intersects(bb) );
    assertTrue( b.intersects(b) );
    bb.clear();
    assertTrue( !bb.intersects(bb) );
    bb.set( 30,40);
    assertTrue( !b.intersects(bb) );
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testEquals() {
    BitSet b = new BitSet();
    BitSet bb = new BitSet();
    b.set(10,20);
    bb.set(5,15);
    //@ assert !b.get(5) && bb.get(5); // lemma needs for the following line
    assertTrue ( !b.equals(bb) );
    assertTrue ( !b.equals(null) );
    assertTrue ( !b.equals(new Object() ) );
    assertTrue ( b.equals(b) );
    //@ assert b.get(10);  // lemma needed for the following line
    assertTrue ( !b.equals( new BitSet() ) );
    assertTrue ( (new BitSet()).equals( new BitSet() ) );
    b.clear();
    bb.clear();
    b.set(7);
    bb.flip(7);
    assertTrue ( b.equals(bb) );

//@ assert false; // TEST FOR CONSISTENCY
  }

  // Not executed because there is a bug in BitSet.get (in Java 1.5 as well).
  // The bug is reported to Sun
  public void NEtestGet() {
    BitSet b = new BitSet();
    assertTrue( b.get(5,10).isEmpty() );
    b.set(10,20);
    BitSet bb = b.get(5,12);
    assertTrue ( !bb.get(9) );
    assertTrue ( bb.get(10) );
    assertTrue ( bb.get(11) );
    assertTrue ( !bb.get(12) );
    bb = b.get(0,7);
    assertTrue( bb.isEmpty() );
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testNext() {
    BitSet b = new BitSet();
    b.set(7);
    b.set(10);
    int i = b.nextSetBit(0);
    assertTrue ( i == 7);
    i = b.nextSetBit(i);
    assertTrue ( i == 7);
    i = b.nextSetBit( i+1);
    assertTrue( i == 10);
    i = b.nextSetBit( i+1);
    assertTrue( i == -1);

    i = 0;
    i = b.nextClearBit(i+1);
//@ assert !b.get(1); // Lemma
    assertTrue( i == 1);
    i = b.nextClearBit(1000);
    assertTrue( i == 1000 );

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testCardinality() {
    BitSet b = new BitSet();
    assertTrue( b.cardinality() == 0);
    b.set(10);
    assertTrue( b.cardinality() == 1);
    b.set(10,20);
    assertTrueNP( b.cardinality() == 10);
    b.clear();
    assertTrue( b.cardinality() == 0);

//@ assert false; // TEST FOR CONSISTENCY
  }

/* No attempts to prove things about size
  public void testSize() {
    BitSet b = new BitSet();
    System.out.println("SIZE " + b.size());
    b = new BitSet(40);
    System.out.println("SIZE " + b.size());
    b.clear(0);
    System.out.println("SIZE " + b.size());
    b.set(70);
    System.out.println("SIZE " + b.size());
    BitSet bb = new BitSet(100);
    System.out.println("SIZE " + bb.size());
    b.or(bb);
    System.out.println("SIZE " + b.size());
    b.clear();
    System.out.println("SIZE " + b.size());
  }
*/
/*
  public static void testBug() {
    BitSet b = new BitSet();
    b.set(10,20);
    System.out.println("BITS " + b);
    System.out.println("b.get(5,12) = " + b.get(5,12));
    System.out.println("b.get(30,40) = " + b.get(30,40));
    System.out.println("b.get(15,25) = " + b.get(15,25));
  }
*/
}
