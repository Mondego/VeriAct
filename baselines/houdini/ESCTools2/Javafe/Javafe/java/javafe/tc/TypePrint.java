/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;
import java.io.OutputStream;

public class TypePrint extends DelegatingPrettyPrint
{
  // Caller must establish del != null!
  //@ requires false;
  public TypePrint() { }

  //@ requires self != null && del != null;
  public TypePrint(PrettyPrint self, PrettyPrint del) {
    super(self, del);
  }

  //@ also
  //@ requires o != null;
  public void print(/*@ non_null */ OutputStream o, int ind, VarInit e) {
    if (e instanceof Expr) {
      Type t = FlowInsensitiveChecks.getTypeOrNull((Expr)e);

      write(o, "(/*");
      if (t==null)
	write(o, "UNAVAILABLE");
      else
        write(o, Types.printName(t));
      write(o, "*/ ");

      del.print(o, ind, e);
      write(o, ')');
    } else del.print(o, ind, e);
  }
} // end of class TypePrint

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */

