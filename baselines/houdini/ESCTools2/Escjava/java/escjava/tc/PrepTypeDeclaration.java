package escjava.tc;

import javafe.ast.TypeDeclElem;
import javafe.tc.TypeSig;
import javafe.ast.Expr;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.FieldDeclVec;
import javafe.ast.MethodDecl;
import javafe.ast.TypeDecl;
import javafe.ast.ClassDecl;
import javafe.ast.InterfaceDecl;
import javafe.ast.Identifier;
import javafe.ast.TypeNameVec;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.ModifierPragmaVec;
import javafe.util.ErrorSet;
import javafe.util.StackVector;
import javafe.util.Location;
import escjava.ast.GhostDeclPragma;
import escjava.ast.ModelDeclPragma;
import escjava.ast.ModelMethodDeclPragma;
import escjava.ast.ModelConstructorDeclPragma;
import escjava.ast.ModelTypePragma;
import escjava.ast.SimpleModifierPragma;
import escjava.ast.Modifiers;
import escjava.ast.TagConstants;
import escjava.ast.Utils;
import escjava.ast.NamedExprDeclPragma;

public class PrepTypeDeclaration extends javafe.tc.PrepTypeDeclaration {

    public PrepTypeDeclaration() {
	inst = this;
    }

    protected /*@ non_null */ StackVector jmlFieldsSeq = new StackVector();
    protected /*@ non_null */ StackVector jmlHiddenFieldsSeq= new StackVector();
    protected /*@ non_null */ StackVector jmlDupFieldsSeq = new StackVector();

    private int numJmlFields = -1;
    private java.util.LinkedList numJmlList = new java.util.LinkedList();

    public void prepStart(TypeSig sig, TypeDecl decl) {
	super.prepStart(sig,decl);
	if (sig instanceof escjava.tc.TypeSig) {
	    jmlFieldsSeq.push();
	    jmlHiddenFieldsSeq.push();
	    jmlDupFieldsSeq.push();
	    Utils.representsDecoration.set(decl, TypeDeclElemVec.make());
	    numJmlList.addFirst(new Integer(numJmlFields));
	    numJmlFields = -1;
	}
    }

    public void prepEnd(TypeSig sig, TypeDecl decl) {
	super.prepEnd(sig,decl);
	if (! (sig instanceof escjava.tc.TypeSig)) return;
	escjava.tc.TypeSig s = (escjava.tc.TypeSig)sig;
	s.jmlFields = FieldDeclVec.popFromStackVector(jmlFieldsSeq);
	s.jmlHiddenFields = FieldDeclVec.popFromStackVector(jmlHiddenFieldsSeq);
	s.jmlDupFields = FieldDeclVec.popFromStackVector(jmlDupFieldsSeq);
	numJmlFields = ((Integer)numJmlList.removeFirst()).intValue();
	
	if (!escjava.Main.options().showFields) return;
	System.out.println("DUMP " + sig );
	print("Java fields",sig.getFieldsRaw());
	print("Hidden Java fields",sig.getHiddenFields());
	print("Jml fields",s.jmlFields);
	print("Jml hidden fields",s.jmlHiddenFields);
	if (s.jmlDupFields.size() != 0) print("Jml dup fields",s.jmlDupFields);
	System.out.println(" DUMP-END");
    }

    public void print(String s, FieldDeclVec fdv ) {
	System.out.println(" " + s);
	for (int i=0; i<fdv.size(); ++i) {
		System.out.println("    "+ fdv.elementAt(i).id + " " + Location.toString(fdv.elementAt(i).getStartLoc()));
	}
    }

    protected void startSupers() {
	numJmlFields = jmlFieldsSeq.size();
	super.startSupers();
    }

    protected void checkSuperInterfaces(javafe.tc.TypeSig sig,
					TypeNameVec superInterfaces) {
	super.checkSuperInterfaces(sig,superInterfaces);
    }

    protected void visitTypeDeclElem(/*@ non_null @*/ TypeDeclElem e,
				 /*@ non_null @*/ TypeSig currentSig,
				 boolean abstractMethodsOK,
				 boolean inFinalClass,
				 boolean inInterface ) {
	if (e instanceof GhostDeclPragma ||
	    e instanceof ModelDeclPragma) {
	    boolean isModel = e instanceof ModelDeclPragma;
	    String jmltype;
	    FieldDecl x;
	    if (isModel) { 
		ModelDeclPragma g = (ModelDeclPragma)e;
		x = g.decl;
		jmltype = "model";
	    } else {
		GhostDeclPragma g = (GhostDeclPragma)e;
		x = g.decl;
		jmltype = "ghost";
	    }

/* DOne in FLowInsensitiveChecks
	    FieldDecl ad = alreadyDeclared(x.id);
	    if (ad != null) {
		ErrorSet.error(x.getStartLoc(),
		    "Identifier has already been declared",
		    ad.getStartLoc());
	    }
*/

	    jmlFieldsSeq.addElement(x);

	    if (Modifiers.isStatic(x.modifiers) && !inInterface
		&& !currentSig.isTopLevelType() && !Modifiers.isFinal(x.modifiers))
		ErrorSet.error(x.locId,
		  "Inner classes may not declare static members, unless they" +
		  " are compile-time constant fields");

	    // ghost fields differ from regular fields in that transient and
	    // volatile do not apply to them
	    // We have a special case of field decls in an interface that are
	    // inherited from java.lang.Object - they might be protected
	    boolean fromClass = x.parent instanceof ClassDecl;
	    checkModifiers( x.modifiers,
		(!inInterface ? Modifiers.ACCESS_MODIFIERS : 
		 !fromClass ? Modifiers.ACC_PUBLIC : 
		   Modifiers.ACC_PUBLIC|Modifiers.ACC_PROTECTED)
				 | Modifiers.ACC_FINAL | Modifiers.ACC_STATIC ,
		x.locId, jmltype + (inInterface ? " interface field" : " field"));

	    // ghost fields in an interface are implicitly public
	    // they are not implicitly final as Java fields are
	    // they are implicitly static only if they are not declared instance
	    if (inInterface) {
		x.modifiers |= Modifiers.ACC_PUBLIC;
		if (Utils.findModifierPragma(
			x.pmodifiers, TagConstants.INSTANCE) == null) {
		    x.modifiers |= Modifiers.ACC_STATIC;
		}
	    }

	    // FIXME - Java fields check for duplicate type names here
	    // where is this done for ghost fields; also I don't think there
	    // is a check anywhere that a ghost field is hiding a 
	    // super class Java field (or enclosing class Java field)

	    getEnvForCurrentSig(currentSig, true).resolveType( currentSig, x.type);

	} else if (e instanceof MethodDecl) {

	    MethodDecl md = (MethodDecl)e;
	    boolean isAbstract = Modifiers.isAbstract(md.modifiers);

	    visitMethodDecl(md,currentSig,abstractMethodsOK, 
			inFinalClass, inInterface);

	    if (!isAbstract && md.body != null
		&& Utils.findModifierPragma(md.pmodifiers,TagConstants.MODEL) != null) 
		md.modifiers = md.modifiers & ~ Modifiers.ACC_ABSTRACT;

/*
	These are not needed at present because routines and types are 
	converted into regular Java routines and types prior to any
	type prepping or type checking.  However, this causes some scope
	problems.  When those are fixed, we may need to prep these types
	here in a manner similar to that done in javafe.tc.PrepTypeDeclaration

	} else if (e instanceof ModelMethodDeclPragma) {
	    ModelMethodDeclPragma g = (ModelMethodDeclPragma)e;
	    MethodDecl x = g.decl;

	    if (Modifiers.isStatic(x.modifiers) && !inInterface 
		&& !currentSig.isTopLevelType())
		ErrorSet.error(x.locId,
			"Only methods of top-level classes may be static");

	    if (inInterface)
		checkModifiers( x.modifiers, 
			Modifiers.ACC_PUBLIC | Modifiers.ACC_ABSTRACT,
			x.loc, "model interface method");
	    else
		checkModifiers( x.modifiers, Modifiers.ACCESS_MODIFIERS |
			Modifiers.ACC_ABSTRACT | Modifiers.ACC_FINAL |
			Modifiers.ACC_SYNCHRONIZED | Modifiers.ACC_STATIC |
			Modifiers.ACC_STRICT,
			x.loc, "model method");
		// Model methods may not be native

	    if (inInterface)
		x.modifiers |= Modifiers.ACC_ABSTRACT | Modifiers.ACC_PUBLIC;

	    if (Modifiers.isPrivate(x.modifiers) || inFinalClass)
		x.modifiers |= Modifiers.ACC_FINAL;

	    if (Modifiers.isAbstract(x.modifiers) &&
		(Modifiers.isPrivate(x.modifiers) |
		 Modifiers.isStatic(x.modifiers)  |
		 Modifiers.isFinal(x.modifiers)  |
		 Modifiers.isNative(x.modifiers)  |
		 Modifiers.isSynchronized(x.modifiers)) )
		ErrorSet.error (x.locId,
		    "Incompatible modifiers for an abstract method");

		// FIXME - resolve types
		// FIXME - check for duplicate method ids
		System.out.println("MODEL METHOD PRAGMA");
		
	} else if (e instanceof ModelConstructorDeclPragma) {
		// FIXME - stuff from non-model constructor
		System.out.println("MODEL CONSTRUCTOR PRAGMA");
	} else if (e instanceof ModelTypePragma) {
		System.out.println("MODEL TYPE PRAGMA");
*/
	} else if (e instanceof NamedExprDeclPragma) {
            TypeDecl decl = currentSig.getTypeDecl();
            TypeDeclElemVec elemvec = (TypeDeclElemVec)Utils.representsDecoration.get(decl);
            // REVIEW: I added robustness in case elemvec is null, but is it allowed to be null
            //   here in the first place?
            if (elemvec == null)
            {
              elemvec = TypeDeclElemVec.make();
              Utils.representsDecoration.set(decl, elemvec);
            }
            elemvec.addElement(e);
	} else 
	    super.visitTypeDeclElem(e,currentSig,abstractMethodsOK, 
			inFinalClass, inInterface);
    }

    public void visitMethodDecl(MethodDecl x, TypeSig currentSig,
			boolean abstractMethodsOK,
                        boolean inFinalClass,
                        boolean inInterface) {
        if (Utils.findModifierPragma(x.pmodifiers,TagConstants.MODEL) == null) {
	    super.visitMethodDecl(x,currentSig,abstractMethodsOK,inFinalClass,inInterface);
        } else {

        // Careful - the following is almost, but not quite, a duplicate of super.visitMethodDecl 

    if (Modifiers.isStatic(x.modifiers) && !inInterface
        && !currentSig.isTopLevelType())
        ErrorSet.error(x.locId,
                       "Only methods of top-level classes may be static");

    // Modifiers can only be:
    //   public protected private abstract static final synchronized native
    //   strictfp
    checkModifiers( x.modifiers,
                   inInterface
                   ? (Modifiers.ACC_PUBLIC | Modifiers.ACC_ABSTRACT | Modifiers.ACC_STATIC)
                   : (Modifiers.ACCESS_MODIFIERS | Modifiers.ACC_ABSTRACT
                   | Modifiers.ACC_FINAL | Modifiers.ACC_SYNCHRONIZED
                   | Modifiers.ACC_STATIC | Modifiers.ACC_NATIVE
                   | Modifiers.ACC_STRICT),
                   x.loc, inInterface ? "interface method" : "method" );

    // If in interface, method is implicitly public
    // But model methods are not implicitly abstract
    if( inInterface )
        x.modifiers |= Modifiers.ACC_PUBLIC;

    // private methods implicitly final
    // members of final class are implicitly final
    if( Modifiers.isPrivate(x.modifiers) || inFinalClass )
      x.modifiers |= Modifiers.ACC_FINAL;

    // Error if an abstract method is also
    // private, static, final, native, or synchronized.
    if( Modifiers.isAbstract(x.modifiers) &&
       (Modifiers.isPrivate(x.modifiers)
        | Modifiers.isStatic(x.modifiers)
        | Modifiers.isFinal(x.modifiers)
        | Modifiers.isNative(x.modifiers)
        | Modifiers.isSynchronized(x.modifiers)) )
      ErrorSet.error( x.locId,
        "Incompatible modifiers for abstract method");

    // resolve types
     getEnvForCurrentSig(currentSig, true).resolveType( currentSig, x.returnType
 );
    for( int i=0; i<x.raises.size(); i++ )
      getEnvForCurrentSig(currentSig, true).resolveType( currentSig, x.raises.elementAt(i) );
    for( int i=0; i<x.args.size(); i++ )
      getEnvForCurrentSig(currentSig, true).resolveType( currentSig, x.args.elementAt(i).type );

    // Error if two methods in type body with same signature
    for( int i=0; i<methodSeq.size(); i++ )
      {
        if(Types.isSameMethodSig( x, (MethodDecl)methodSeq.elementAt(i) ) )
          ErrorSet.error(x.loc,
                         "Duplicate declaration of method "
                         +"with same signature");
      }

    methodSeq.addElement(x);

    }
    }

    protected void addInheritedMembers(javafe.tc.TypeSig type,
				       javafe.tc.TypeSig superType) {

	super.addInheritedMembers(type,superType);
	if (!(superType instanceof escjava.tc.TypeSig) ) return;

	escjava.tc.TypeSig st = (escjava.tc.TypeSig)superType;
        for (int i=0; i<st.jmlFields.size(); ++i) {
	    FieldDecl fd = st.jmlFields.elementAt(i);
	    if (jmlFieldsSeq.contains(fd) ||
		jmlHiddenFieldsSeq.contains(fd)) {
		// skip
	    } else if (!superMemberAccessible(type,superType,fd.modifiers,
						fd.pmodifiers)) {
		jmlHiddenFieldsSeq.addElement(fd);
	    } else if (declaresField(type,fd.id,numJmlFields)) {
		// If the field is declared in this class (not inherited) as
		// either a java or jml field, then the super field is hidden
		jmlHiddenFieldsSeq.addElement(fd);
	    } else {
		// If the field is somehow inherited as a java field, but is
		// also inherited as a jml field, then for jml name lookup
		// it is considered an ambiguous reference.
		boolean found = false;
		for (int ii=0; ii<fieldSeq.size(); ++ii) {
		    FieldDecl fdj = (FieldDecl)fieldSeq.elementAt(ii);
		    if (fdj.id.equals(fd.id)) { found = true; break; }
		}
		if (!found) jmlFieldsSeq.addElement(fd);
		else        jmlDupFieldsSeq.addElement(fd);
		// FIXME - but this only checks the java fields that have
		// been added via addInheritedMembers so far - there might
		// be more interfaces to come. ???
	    }
	}
	for (int i=0; i<st.jmlHiddenFields.size(); ++i) {
	    jmlHiddenFieldsSeq.addElement(st.jmlHiddenFields.elementAt(i));
	}
	TypeDeclElemVec mpv = (TypeDeclElemVec)Utils.representsDecoration.get(type.getTypeDecl());
	TypeDeclElemVec mpvsuper = (TypeDeclElemVec)Utils.representsDecoration.get(st.getTypeDecl());

        // REVIEW: I added robustness in case mpv is null, but is it allowed to be null
        //   in the first place?
	if (st.getTypeDecl() instanceof ClassDecl) {
	    if (mpv != null)
                mpv.append(mpvsuper);
            else
                Utils.representsDecoration.set(type.getTypeDecl(), TypeDeclElemVec.make (mpvsuper.toArray()));
	} else {
            if (mpv != null)
	        mpv.append(mpvsuper);
            else
                Utils.representsDecoration.set(type.getTypeDecl(), TypeDeclElemVec.make (mpvsuper.toArray()));
		//interfaces get them from Object as well ?
		// FIXME - no dups
	    //for (int i=0; i<mpvsuper.size(); ++i) 
	}
    }

    protected boolean superMemberAccessible(javafe.tc.TypeSig type,
					javafe.tc.TypeSig superType,
					int modifiers,
					ModifierPragmaVec pmodifiers) {
	if (Utils.findModifierPragma(pmodifiers,TagConstants.SPEC_PUBLIC)
		!= null) return true;
	if (Utils.findModifierPragma(pmodifiers,TagConstants.SPEC_PROTECTED)
		!= null) return true;
	return super.superMemberAccessible(type,superType,modifiers,pmodifiers);
    }

/*
    protected FieldDecl alreadyDeclared(Identifier id) {
	for (int i=0; i<jmlFieldsSeq.size(); ++i) {
	    if (id.equals(((FieldDecl)jmlFieldsSeq.elementAt(i)).id)) 
		return (FieldDecl)jmlFieldsSeq.elementAt(i);
	}
	for (int i=0; i<fieldSeq.size(); ++i) {
	    if (id.equals(((FieldDecl)fieldSeq.elementAt(i)).id)) 
		return (FieldDecl)fieldSeq.elementAt(i);
	}
	return null;
    }
*/

    protected boolean declaresField(javafe.tc.TypeSig sig, Identifier id) {
	return declaresField(sig,id,jmlFieldsSeq.size());
    }

    private boolean declaresField(javafe.tc.TypeSig sig, Identifier id, int n) {
	    // The following returns true iff sig declares the field as a
	    // java field and does not inherit it from a parent 
	boolean b = declaresFieldJava(sig,id);
	if (b || !(sig instanceof escjava.tc.TypeSig)) return b;
	for (int i=0; i<n; ++i) {
	    if (id.equals(((FieldDecl)jmlFieldsSeq.elementAt(i)).id)) return true;
	}
	for (int i=0; i<jmlHiddenFieldsSeq.size(); ++i) {
	    if (id.equals(((FieldDecl)jmlHiddenFieldsSeq.elementAt(i)).id)) return true;
	}
	for (int i=0; i<jmlDupFieldsSeq.size(); ++i) {
	    if (id.equals(((FieldDecl)jmlDupFieldsSeq.elementAt(i)).id)) return true;
	}
	return false;
    }

    public javafe.tc.TypeSig getRootInterface() {
	if (_rootCache != null) return _rootCache;
	javafe.tc.TypeSig ts = super.getRootInterface();
	InterfaceDecl td = (InterfaceDecl) ts.getTypeDecl();
	// Add in any ghost or model fields

	TypeDecl object = Types.javaLangObject().getTypeDecl();
	for (int i=0; i<object.elems.size(); ++i) {
	    TypeDeclElem e = object.elems.elementAt(i);
// FIXME - should really only do inherited elements
// FIXME - what about inherited model methods, types?
	    if ((e instanceof ModelDeclPragma)
	    	|| (e instanceof GhostDeclPragma)
	    	|| (e instanceof FieldDecl)) {
		if (!Modifiers.isStatic(e.getModifiers())) {
		    e.getPModifiers().addElement(
			SimpleModifierPragma.make(TagConstants.INSTANCE,Location.NULL));
		}
		td.elems.addElement(e);
	    }
	}

        _rootCache = ts;
	return _rootCache;


    }

}
