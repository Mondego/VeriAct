// Decompiled by Jad v1.5.8e. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.geocities.com/kpdus/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   CardException.java

//@ refine "CardException.jml";

public class CardException extends Exception {

    public CardException(short reason)
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
            throws CardException
    {
        systemInstance.setReason(reason);
        throw systemInstance;
    }

    private static CardException systemInstance;
    private short reason;
}
