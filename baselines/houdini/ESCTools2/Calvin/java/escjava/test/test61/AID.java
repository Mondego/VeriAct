/* Copyright Hewlett-Packard, 2002 */

class AID {

    /*@ spec_public */ byte[] theAID;

    /*@ invariant theAID != null &&
                  5 <= theAID.length &&
                  theAID.length <= 16;
    */


    public byte getBytes (byte[] dest, short offset)
    /*@ requires dest != null &&
                 offset >= 0 &&
                 offset + theAID.length <= dest.length;
        modifies dest[*];
        ensures (\forall int i;
                         i < offset || i >= offset + theAID.length ==>
                         dest[i] == \old(dest[i]));
    */
    {
      System.arraycopy(theAID, (short)0, dest, offset, (short)theAID.length);
      return (byte)theAID.length;
    }

    //@ requires 5 <= len & len <= 16;
    public AID(int len) {
      theAID = new byte[len];
    }

}
