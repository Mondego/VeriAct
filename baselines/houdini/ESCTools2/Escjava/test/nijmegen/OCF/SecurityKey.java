//import javacard.framework.*;

class SecurityKey
{
    public static final byte MAX_KEY_LENGTH = (byte)16;

    private /*@ spec_public @*/ byte[] theKey; 
    private /*@ spec_public @*/ byte keyLen; 

    /*@
      @ invariant theKey != null &&
      @           theKey.length == MAX_KEY_LENGTH &&
      @           0 <= keyLen && keyLen <= MAX_KEY_LENGTH; 
      @*/

    /*@ normal_behavior
      @   requires true;
      @   modifiable theKey, keyLen;
      @   ensures \fresh(theKey);
      @*/
    public SecurityKey() {
        theKey = new byte[MAX_KEY_LENGTH]; 
        keyLen = MAX_KEY_LENGTH;
    }    

    /*@ behavior
      @   requires bArray.length <= MAX_KEY_LENGTH;
      @   modifiable keyLen, theKey[*];
      @   ensures bArray != null && 
      @            bArray.length <= MAX_KEY_LENGTH &&
      @            keyLen == bArray.length &&
      @            (\forall short i ; 0 <= i && i < keyLen
      @                   ==> theKey[i] == bArray[i]);
      @    signals_only NullPointerException, ArrayIndexOutOfBoundsException;
      @    signals (ArrayIndexOutOfBoundsException) false;
      @    signals (NullPointerException) false;
      @*/
    public void setKey(/*@ non_null @*/ byte[] bArray) {        
        keyLen = (byte)bArray.length;
        arrayCopy(bArray, (short)0, theKey, (short)0, keyLen);
    } //@ nowarn Exception;

    /*@ behavior
      @     requires src != null &&  srcOff >= 0 &&  
      @              srcOff+length <= src.length &&
      @            dest != null && destOff >= 0 && 
      @              destOff+length <= dest.length &&
      @            length >= 0;
      @     modifiable dest[destOff..destOff+length-1];
      @     ensures (\forall short i ; 0 <= i && i < length 
      @                   ==> dest[destOff+i] == \old(src[srcOff+i]));
      @ also
      @ behavior
      @     modifiable dest[destOff..destOff+length-1];
      @     signals_only NullPointerException, ArrayIndexOutOfBoundsException;
      @     signals (NullPointerException) src == null || dest == null;
      @     signals (ArrayIndexOutOfBoundsException) 
      @                 (src != null &&
      @                  (0 > srcOff || srcOff + length > src.length))
      @                 ||
      @                 (dest != null &&
      @                  (0 > destOff || destOff + length > dest.length));
      @*/
    public static final native short arrayCopy(byte[] src, 
                                               short srcOff, 
                                               byte[] dest, 
                                               short destOff, 
                                               short length);
}
