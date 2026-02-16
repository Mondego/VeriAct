// Decompiled by Jad v1.5.8e. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/kpdus/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   CardException.java


public class CardException2 extends Exception {

    public CardException2(short reason)
    {
        this.reason = reason;
        if(systemInstance == null)
            systemInstance = this;
    }

    public short getReason()
    {
        return reason;
    }

    public void setReason(short reason)
    {
        this.reason = reason;
    }

    public static void throwIt(short reason)
            throws CardException2
    {
        systemInstance.setReason(reason);
        throw systemInstance;
    }

    private static CardException2 systemInstance;
    private short reason;
}
