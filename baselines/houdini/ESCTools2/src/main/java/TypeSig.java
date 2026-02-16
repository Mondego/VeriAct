package escjava.tc;
import javafe.ast.TypeDecl;
import javafe.ast.CompilationUnit;
import javafe.ast.FieldDeclVec;
import javafe.ast.Identifier;
import javafe.ast.FieldDecl;
import javafe.tc.LookupException;

public class TypeSig extends javafe.tc.TypeSig {

    //@ requires \nonnullelements(packageName);
    public TypeSig(/*@ non_null @*/ String[] packageName,
		   /*@ non_null */ String simpleName,
		   /*@ nullable */ javafe.tc.TypeSig enclosingType,
		   /*@ nullable */ TypeDecl decl,
		   /*@ nullable */ CompilationUnit CU) {
	super(packageName,simpleName,enclosingType,decl,CU);
    }

    public TypeSig(/*@ nullable */ String simpleName,
		/*@ non_null */ javafe.tc.Env enclosingEnv,
		/*@ non_null */ TypeDecl decl) {
	super(simpleName,enclosingEnv,decl);
    }

    public /*@ nullable */ FieldDeclVec jmlFields;
    public /*@ non_null */ FieldDeclVec jmlHiddenFields;
    public /*@ non_null */ FieldDeclVec jmlDupFields;

    public boolean hasField(/*@ non_null */ Identifier id) {
	// FIXME: jmlFIelds can be null for a JMLDataGroup
	prep();
	if (FlowInsensitiveChecks.inAnnotation && jmlFields != null) {
	    for (int i=0; i<jmlFields.size(); ++i) {
		if (jmlFields.elementAt(i).id.equals(id)) return true;
	    }
	}
	return super.hasField(id);
    }

    	//@ also
    	//@   requires caller != null;
    public /*@non_null*/ FieldDecl lookupField(/*@non_null*/Identifier id, 
                                 /*@non_null*/javafe.tc.TypeSig caller) 
    		throws LookupException {
	FieldDecl r = null;
	prep();
	// FIXME: jmlFIelds can be null for a JMLDataGroup
	if (FlowInsensitiveChecks.inAnnotation && jmlFields != null) {
	    for (int i=0; i<jmlFields.size(); ++i) {
		if (jmlFields.elementAt(i).id.equals(id)) {
		    if (r == null) r = jmlFields.elementAt(i);
		    else throw new LookupException( LookupException.AMBIGUOUS );
		}
	    }
	    for (int i=0; i<jmlDupFields.size(); ++i) {
		if (jmlDupFields.elementAt(i).id.equals(id)) {
		    throw new LookupException( LookupException.AMBIGUOUS );
		}
	    }
	}
	if (r != null) return r;
	return super.lookupField(id,caller);
    }
}
