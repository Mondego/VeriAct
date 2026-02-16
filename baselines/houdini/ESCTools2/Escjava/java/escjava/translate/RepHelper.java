package escjava.translate;

import javafe.ast.*;

public class RepHelper {

    public /*@ nullable */ TypeDecl td;
    public /*@ nullable */ ASTNode a;

    public RepHelper(/*@ nullable */ TypeDecl td, /*@ nullable */ FieldDecl fd) {
	this.td = td;
	this.a = fd;
    }

    public RepHelper(/*@ nullable */ TypeDecl td, /*@ nullable */ RoutineDecl rd) {
	this.td = td;
	this.a = rd;
    }

    public RepHelper(/*@ non_null @*/ FieldAccess fa) {
	this.a = fa.decl;
	ObjectDesignator od = fa.od;
	Type t = null;
	if (od instanceof ExprObjectDesignator) {
	    t = javafe.tc.FlowInsensitiveChecks.getType(
			((ExprObjectDesignator)od).expr );
	} else if (od instanceof TypeObjectDesignator) {
	    t = ((TypeObjectDesignator)od).type;
	}
	if (t instanceof javafe.tc.TypeSig) 
		this.td = ((javafe.tc.TypeSig)t).getTypeDecl();
    }

    public boolean equals(/*@ nullable */ Object o) {
	if (!(o instanceof RepHelper)) return false;
	RepHelper r = (RepHelper)o;
	return td == r.td && a == r.a;
    }

    public int hashCode() {
	return (td==null?0:td.hashCode()) + (a==null?0:a.hashCode());
    }
}
