class KeySet
{
    public static final byte MAX_KEYS = 3;

    private /*@ spec_public @*/ SecurityKey[] keyArray;
    //@ invariant keyArray != null ==> keyArray.owner == this;

    /*@ 
      @ invariant \elemtype(\typeof(keyArray)) == \type(SecurityKey);
      @ invariant keyArray != null && keyArray.length == MAX_KEYS;
      @*/

    //@ invariant (\forall int i; 0<=i && i < keyArray.length; keyArray[i] == null || keyArray[i].owner == this);

    /*@ behavior
      @   requires 1 <= index && index <= MAX_KEYS;
      @   requires bArray != null;
      @   requires bArray.length <= SecurityKey.MAX_KEY_LENGTH;
      @   modifiable keyArray[index-1];
      @    ensures keyArray[index-1] != null;
      @    ensures keyArray[index-1].owner == this;
      @    ensures keyArray[index-1].keyLen == bArray.length;
      @    ensures (\forall short i ; 0 <= i && i < keyArray[index-1].keyLen
      @                          ==> keyArray[index-1].theKey[i] == bArray[i]);
      @    signals (ArrayIndexOutOfBoundsException) false;
      @    signals (NullPointerException) false;
      @*/
    public void setKeyBytes(/*@ non_null @*/ byte[] bArray,
                            byte index)
    {
        keyArray[--index] = new SecurityKey();
        //@ set keyArray[index].owner = this;
        keyArray[index].setKey(bArray);
    }

    //@ modifies keyArray;
    //@ ensures (\forall int i; 0<=i && i < MAX_KEYS; keyArray[i] == null);
    public KeySet() {
        keyArray = new SecurityKey[MAX_KEYS];
	//@ set keyArray.owner = this;
    }
}
