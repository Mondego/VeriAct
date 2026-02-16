/* Copyright 2000, 2001, Compaq Computer Corporation */

package instrumenter;

import java.io.OutputStream;
import escjava.ast.EscPrettyPrint;
import javafe.ast.*;
import escjava.ast.TagConstants;


public class InstrumenterPrettyPrint extends EscPrettyPrint {
  final static String RES = "InStRuMeNtEr_RES";

  public InstrumenterPrettyPrint() { }

  public InstrumenterPrettyPrint(PrettyPrint self, PrettyPrint del) {
    super(self, del);
  }

  public void print(OutputStream o, int ind, VarInit e) {
    if (e.getTag() == TagConstants.RESEXPR) {
      write(o, RES);
    } else {
      super.print(o, ind, e);
    }
  }
}
