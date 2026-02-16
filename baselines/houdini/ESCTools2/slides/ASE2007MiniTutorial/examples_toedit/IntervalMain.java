class IntervalMain {

  public static void main(String [] argv) {
    Interval i3_10 = new Interval(3,10);
    System.out.println(
       "i3_10.contains(7) == "
        + i3_10.contains(7));
    //@ assert i3_10.contains(3);
    //@ assert i3_10.contains(7);
    //@ assert i3_10.contains(10);
    //@ assert !i3_10.contains(2);
    //@ assert !i3_10.contains(11);
    Interval i2_1 = new Interval(2,1);
    //@ assert !i2_1.contains(8);
  }
}
