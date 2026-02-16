class BagOfIntMain {

  public static void main(String [] argv) {
    int[] mine
       = new int[] {0, 10, 20, 30, 40, 10};
    BagOfInt b = new BagOfInt(mine);
    System.out.println(
       "b.occurrences(10) == "
        + b.occurrences(10));
    //@ assert b.occurrences(10) == 2;
    //@ assert b.occurrences(5) == 0;
    int em1 = b.extractMin();
    //@ assert em1 == 0;
    int em2 = b.extractMin();
    //@ assert em2 == 10;
    int em3 = b.extractMin();
    //@ assert em2 == 10;
  }
}
