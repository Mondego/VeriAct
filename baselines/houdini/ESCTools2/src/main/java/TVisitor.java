package escjava.vcGeneration;

import java.io.*;

abstract public class TVisitor {

    /**
     * Default pretty printer that may be used by implementing vistor patterns.
     */
    protected PrettyPrinter lib = null;

    protected/*@ non_null @*/Writer out = null;

    protected/*@ non_null @*/StringBuffer indentation = null;

    public TVisitor(Writer out) {
        this(out, "  ", "(", ")", "\n");
    }

    public TVisitor(Writer out, String tab, String lbr, String rbr, String nl) {
        this.out = out;
        lib = new PrettyPrinter(out, tab, lbr, rbr, nl);
        indentation = new StringBuffer();
    }

    //non automatic division
    abstract public void visitTName(/*@ non_null @*/TName n);
    
    abstract public void visitTRoot(/*@ non_null @*/TRoot n);

    // class created using the perl script
    abstract public void visitTBoolImplies(/*@ non_null @*/TBoolImplies n);

    abstract public void visitTBoolAnd(/*@ non_null @*/TBoolAnd n);

    abstract public void visitTBoolOr(/*@ non_null @*/TBoolOr n);

    abstract public void visitTBoolNot(/*@ non_null @*/TBoolNot n);

    abstract public void visitTBoolEQ(/*@ non_null @*/TBoolEQ n);

    abstract public void visitTBoolNE(/*@ non_null @*/TBoolNE n);

    abstract public void visitTAllocLT(/*@ non_null @*/TAllocLT n);

    abstract public void visitTAllocLE(/*@ non_null @*/TAllocLE n);

    abstract public void visitTAnyEQ(/*@ non_null @*/TAnyEQ n);

    abstract public void visitTAnyNE(/*@ non_null @*/TAnyNE n);

    abstract public void visitTIntegralEQ(/*@ non_null @*/TIntegralEQ n);

    abstract public void visitTIntegralGE(/*@ non_null @*/TIntegralGE n);

    abstract public void visitTIntegralGT(/*@ non_null @*/TIntegralGT n);

    abstract public void visitTIntegralLE(/*@ non_null @*/TIntegralLE n);

    abstract public void visitTIntegralLT(/*@ non_null @*/TIntegralLT n);

    abstract public void visitTIntegralNE(/*@ non_null @*/TIntegralNE n);

    abstract public void visitTIntegralAdd(/*@ non_null @*/TIntegralAdd n);

    abstract public void visitTIntegralDiv(/*@ non_null @*/TIntegralDiv n);

    abstract public void visitTIntegralMod(/*@ non_null @*/TIntegralMod n);

    abstract public void visitTIntegralMul(/*@ non_null @*/TIntegralMul n);

    abstract public void visitTFloatEQ(/*@ non_null @*/TFloatEQ n);

    abstract public void visitTFloatGE(/*@ non_null @*/TFloatGE n);

    abstract public void visitTFloatGT(/*@ non_null @*/TFloatGT n);

    abstract public void visitTFloatLE(/*@ non_null @*/TFloatLE n);

    abstract public void visitTFloatLT(/*@ non_null @*/TFloatLT n);

    abstract public void visitTFloatNE(/*@ non_null @*/TFloatNE n);

    abstract public void visitTFloatAdd(/*@ non_null @*/TFloatAdd n);

    abstract public void visitTFloatDiv(/*@ non_null @*/TFloatDiv n);

    abstract public void visitTFloatMod(/*@ non_null @*/TFloatMod n);

    abstract public void visitTFloatMul(/*@ non_null @*/TFloatMul n);

    abstract public void visitTLockLE(/*@ non_null @*/TLockLE n);

    abstract public void visitTLockLT(/*@ non_null @*/TLockLT n);

    abstract public void visitTRefEQ(/*@ non_null @*/TRefEQ n);

    abstract public void visitTRefNE(/*@ non_null @*/TRefNE n);

    abstract public void visitTTypeEQ(/*@ non_null @*/TTypeEQ n);

    abstract public void visitTTypeNE(/*@ non_null @*/TTypeNE n);

    abstract public void visitTTypeLE(/*@ non_null @*/TTypeLE n);

    abstract public void visitTCast(/*@ non_null @*/TCast n);

    abstract public void visitTIs(/*@ non_null @*/TIs n);

    abstract public void visitTSelect(/*@ non_null @*/TSelect n);

    abstract public void visitTStore(/*@ non_null @*/TStore n);

    abstract public void visitTTypeOf(/*@ non_null @*/TTypeOf n);

    abstract public void visitTForAll(/*@ non_null @*/TForAll n);

    abstract public void visitTExist(/*@ non_null @*/TExist n);

    abstract public void visitTIsAllocated(/*@ non_null @*/TIsAllocated n);

    abstract public void visitTEClosedTime(/*@ non_null @*/TEClosedTime n);

    abstract public void visitTFClosedTime(/*@ non_null @*/TFClosedTime n);

    abstract public void visitTAsElems(/*@ non_null @*/TAsElems n);

    abstract public void visitTAsField(/*@ non_null @*/TAsField n);

    abstract public void visitTAsLockSet(/*@ non_null @*/TAsLockSet n);

    abstract public void visitTArrayLength(/*@ non_null @*/TArrayLength n);

    abstract public void visitTArrayFresh(/*@ non_null @*/TArrayFresh n);

    abstract public void visitTArrayShapeOne(/*@ non_null @*/TArrayShapeOne n);

    abstract public void visitTArrayShapeMore(/*@ non_null @*/TArrayShapeMore n);

    abstract public void visitTIsNewArray(/*@ non_null @*/TIsNewArray n);

    abstract public void visitTString(/*@ non_null @*/TString n);

    abstract public void visitTBoolean(/*@ non_null @*/TBoolean n);

    abstract public void visitTChar(/*@ non_null @*/TChar n);

    abstract public void visitTInt(/*@ non_null @*/TInt n);

    abstract public void visitTFloat(/*@ non_null @*/TFloat n);

    abstract public void visitTDouble(/*@ non_null @*/TDouble n);

    abstract public void visitTNull(/*@ non_null @*/TNull n);

    // added by me
    abstract public void visitTUnset(/*@ non_null @*/ TUnset n);

    abstract public void visitTMethodCall(/*@ non_null @*/ TMethodCall call);

    abstract public void visitTIntegralSub(/*@ non_null @*/ TIntegralSub sub);
    
    abstract public void visitTSum (/*@ non_null @*/  TSum s);
}
