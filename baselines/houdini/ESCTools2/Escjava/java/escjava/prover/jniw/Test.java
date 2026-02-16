package escjava.prover.jniw;

public class Test {

  public static void main (String[] args) {
   try {
     String s;
     Cvc3Wrapper vc = new Cvc3Wrapper();
     vc.startSolver();
     vc.setFlags("-tcc");
     vc.declFun("f:(INT,INT)->INT;");
     vc.declFun("x,y,z:INT;");
     vc.declPred("a,b,c:BOOLEAN;");
     vc.assertFormula("ASSERT a AND b AND NOT c;");
     vc.assertFormula("ASSERT x=y;");
     vc.assertFormula("ASSERT f(x,x)=y;");
     vc.assertFormula("ASSERT f(x,x)=z;");
     s = vc.queryFormula("QUERY x=z;");
     System.out.println("1 (valid) "+s);
     vc.undoAssert(2);
     s = vc.queryFormula("QUERY f(x+1,x+1)=y;");
     System.out.println("2 (invalid) "+s);
     vc.stopSolver();
     vc.setFlags("-lang presentation -output-lang smtlib");
     System.out.println("Reset flags, new solver");
     vc.startSolver();
     vc.declFun("f:(INT,INT)->INT;");
     vc.declFun("x,y,z:INT;");
     vc.declPred("a,b,c:BOOLEAN;");
     vc.assertFormula("ASSERT NOT(a AND NOT b AND c);");
     vc.assertFormula("ASSERT x/=y;");
     vc.assertFormula("ASSERT f(x+1,x+1)=y+1;");
     vc.assertFormula("ASSERT f(x,x)=z;");
     s = vc.queryFormula("QUERY NOT x=z;");
     System.out.println("3 (invalid) "+s);
     s = vc.queryFormula("QUERY f(x,y)=y;");
     System.out.println("4 (invalid) "+s);
     s = vc.queryFormula("QUERY EXISTS(w:INT):f(x,w)=y;");
     System.out.println("5 (unknown) "+s);
     vc.stopSolver();
   } catch (Exception e) {
     System.out.println(e);
   }
  }


}
