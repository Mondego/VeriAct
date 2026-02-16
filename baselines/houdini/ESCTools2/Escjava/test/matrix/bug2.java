class bug2
{
    //@ requires len >= 0;
    //@ requires len <= a.length;
    //@ requires len <= b.length;
    static void copy(long[] a, long[] b, int len)
    {
	System.arraycopy(a, 0, b, 0, len);
    }
}
