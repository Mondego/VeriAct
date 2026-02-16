/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;
import javafe.util.*;
import java.util.Hashtable;
import java.util.Enumeration;

/**
 * Does type name resolution and type checking at signature level of a type
 * declaration, and infers the members of the declaration. Also resolves type
 * names at the signature level. Assumes the TypeSig was previously in the
 * "supertype links resolved" state.
 */

public class PrepTypeDeclaration {

	public static final int INTERFACE_MODIFIERS = Modifiers.ACC_PUBLIC
					| Modifiers.ACC_FINAL | Modifiers.ACC_STATIC;

	public static final int ALLOWED_MODIFIERS = Modifiers.ACCESS_MODIFIERS | Modifiers.ACC_FINAL
							| Modifiers.ACC_STATIC | Modifiers.ACC_TRANSIENT
							| Modifiers.ACC_ENUM
							| Modifiers.ACC_VOLATILE;

	/** A (possibly extended) instance of PrepTypeDeclaration. */
	/*@spec_public */protected static PrepTypeDeclaration inst;

	//@ public invariant inst!=null; //this is not meant to be a static
	// invariant

	public static/*@non_null */PrepTypeDeclaration getInst() {
		return inst;
	}

	/** */

	public PrepTypeDeclaration() {
		inst = this;
		//@ set methodSeq.elementType = \type(MethodDecl);
		//@ set methodSeq.owner = this;

		//@ set fieldSeq.elementType = \type(FieldDecl);
		//@ set fieldSeq.owner = this;

		//@ set hiddenfieldSeq.elementType = \type(FieldDecl);
		//@ set hiddenfieldSeq.owner = this;

		//@ set constructorSeq.elementType = \type(ConstructorDecl);
		//@ set constructorSeq.owner = this;

		//@ set overridesDecoration.decorationType = \type(Set);
	}

	/**
	 * Does type name resolution and type checking at signature level of a type
	 * declaration, and infers the members of the declaration. Sets the
	 * "methods" and "fields" fields of the TypeSig appropriately.
	 */
	public void prepTypeSignature(/*@ non_null @*/TypeSig sig) {

		TypeDecl decl = sig.getTypeDecl();

		prepStart(sig, decl);
		prepDo(sig, decl);
		prepEnd(sig, decl);
	}

	protected void prepStart(TypeSig sig, TypeDecl decl) {
		/*
		 * Check for same simple name as an enclosing type:
		 */
		for (TypeSig enclosing = sig.enclosingType; enclosing != null; enclosing = enclosing.enclosingType) {
			if (decl.id.equals(enclosing.getTypeDecl().id))
				ErrorSet.error(decl.locId,
						"A type may not have the same simple "
								+ "name as any of its enclosing types.");
		}

		fieldSeq.push();
		hiddenfieldSeq.push();
		methodSeq.push();
		constructorSeq.push();
		numList.addFirst(new Integer(numFields));
		numFields = -1;
	}

	protected void prepDo(TypeSig sig, /*@ non_null @*/ TypeDecl decl) {
		//@ assume (decl instanceof ClassDecl) || (decl instanceof InterfaceDecl);
		if (decl instanceof ClassDecl)
			visitClassDecl((ClassDecl) decl, sig);
		else
			visitInterfaceDecl((InterfaceDecl) decl, sig);
	}

	protected void prepEnd(TypeSig sig, TypeDecl decl) {
		sig.fields = FieldDeclVec.popFromStackVector(fieldSeq);
		sig.hiddenfields = FieldDeclVec.popFromStackVector(hiddenfieldSeq);
		sig.methods = MethodDeclVec.popFromStackVector(methodSeq);
		constructorSeq.pop();
		numFields = ((Integer) numList.removeFirst()).intValue();
	}

	// ----------------------------------------------------------------------

	/**
	 * Decorates MethodDecl of prepped TypeDecls with a Set of methods that decl
	 * overrides or hides. This is *NOT* transtitive!
	 */

	//@ requires md != null && overrides != null;
	private void addOverride(MethodDecl md, MethodDecl overrides) {
		Set overridesSet = getOverrides(md);
		overridesSet.add(overrides);
	}

	/**
	 * Returns the set of all methods that <code>md</code> overrides, where
	 * <code>md</code> is considered to appear in those prepped subtypes of
	 * <code>md.parent</code> that inherit <code>md</code>.
	 * 
	 * Warning: This set may expand as additional subtypes of
	 * <code>md.parent</code> are prepped.
	 * 
	 * Warning: If you want the set of methods that <code>md</code> overrides,
	 * with <code>md</code> considered to appear in a particular type
	 * <code>td</code>, use getOverrides(TypeDecl, MethodDecl) instead!
	 */
	//@ requires md != null;
	//@ ensures \result != null;
	//@ ensures \result.elementType == \type(MethodDecl);
	public Set getOverrides(MethodDecl md) {
		Set overrides = (Set) overridesDecoration.get(md);
		//@ assume overrides.elementType == \type(MethodDecl);
		if (overrides == null) {
			overrides = new Set();
			//@ assume overrides.elementType == \type(MethodDecl);
			overridesDecoration.set(md, overrides);
		}
		return overrides;
	}

	/**
	 * Returns the set of methods that <code>md</code> overrides, with
	 * <code>md</code> considered to appear in a particular type
	 * <code>td</code>.
	 * <p>
	 * 
	 * This routine may result in <code>td</code> being prepped.
	 */
	//@ requires td != null;
	//@ requires md != null;
	//@ ensures \result != null;
	//@ ensures \result.elementType == \type(MethodDecl);
	public Set getOverrides(TypeDecl td, MethodDecl md) {
		TypeSig sig = TypeSig.getSig(td);
		sig.prep();

		Set overrides = getOverrides(md);
		Set actualOverrides = new Set();
		//@ assume actualOverrides.elementType == \type(MethodDecl);

		Enumeration overridden_methods = overrides.elements();
		while (overridden_methods.hasMoreElements()) {
			MethodDecl smd = (MethodDecl) overridden_methods.nextElement();
			//@ assume smd.hasParent;
			if (sig.isSubtypeOf(TypeSig.getSig(smd.parent))) {
				actualOverrides.add(smd);
			}
		}

		return actualOverrides;
	}

	//@ invariant overridesDecoration.decorationType == \type(Set);
	//@ spec_public
	private static/*@ non_null @*/ASTDecoration overridesDecoration = new ASTDecoration(
			"overridesDecoration");

	// ----------------------------------------------------------------------

	// Stacks up the members of the type
	//@ invariant fieldSeq.elementType == \type(FieldDecl);
	//@ invariant fieldSeq.owner == this;
	protected/*@ non_null @*/StackVector fieldSeq = new StackVector();

	//@ invariant hiddenfieldSeq.elementType == \type(FieldDecl);
	//@ invariant hiddenfieldSeq.owner == this;
	protected/*@ non_null @*/StackVector hiddenfieldSeq = new StackVector();

	// "invariant" <elements>.hasParent
	//@ invariant methodSeq.elementType == \type(MethodDecl);
	//@ invariant methodSeq.owner == this;
	protected/*@ non_null @*/StackVector methodSeq = new StackVector();

	//@ invariant constructorSeq.elementType == \type(ConstructorDecl);
	//@ invariant constructorSeq.owner == this;
	protected/*@non_null*/StackVector constructorSeq = new StackVector();

	private int numFields = -1;

	private java.util.LinkedList numList = new java.util.LinkedList();

	// ----------------------------------------------------------------------

	/**
	 * Does signature-level checking and adds type members to fieldSeq and
	 * methodSeq.
	 */

	//@ requires decl != null && currentSig != null;
	protected void visitClassDecl(ClassDecl decl, TypeSig currentSig) {

		// Check that the modifiers are ok
		checkTypeModifiers(decl, currentSig, true);

		// Visit all enclosed member declarations
		// They will add themselves to fieldSeq and methodSeq

		for (int i = 0; i < decl.elems.size(); i++) {
			TypeDeclElem elem = decl.elems.elementAt(i);
			//@ assume elem.hasParent; // "invariant"
			visitTypeDeclElem(elem, currentSig, Modifiers
					.isAbstract(decl.modifiers), Modifiers
					.isFinal(decl.modifiers), false);
		}

		startSupers();

		// Add members of direct super interfaces

		// checkSuperInterfaces( currentSig, decl.superInterfaces );

		// Add members of direct superclass, if any
		// superclass may be null, or may name an interface

		TypeName superClassName = decl.superClass;
		TypeSig superClassSig = superClassName == null ? null : TypeSig
				.getSig(superClassName);

		if (superClassSig != null) {
			if (superClassSig.getTypeDecl() instanceof ClassDecl) {
				// check superclass is not final
				if (Modifiers.isFinal(superClassSig.getTypeDecl().modifiers))
					ErrorSet.error(superClassName.getStartLoc(),
							"Can't subclass final classes: class "
									+ superClassSig.getExternalName());
				else
					addInheritedMembers(currentSig, superClassSig);
				checkSuperTypeAccessible(currentSig, superClassSig,
						superClassName == null ? decl.getStartLoc()
								: superClassName.getStartLoc());
			} else {
				ErrorSet.error(superClassName.getStartLoc(),
						"Can't subclass interfaces: interface "
								+ superClassSig.getExternalName());
			}
		}

		// Add members of direct super interfaces

		checkSuperInterfaces(currentSig, decl.superInterfaces);

		// Check no two abstract methods with same method signature
		// and different return types

		for (int i = 0; i < methodSeq.size(); i++) {
			MethodDecl mdi = (MethodDecl) methodSeq.elementAt(i);
			for (int j = 0; j < i; j++) {
				MethodDecl mdj = (MethodDecl) methodSeq.elementAt(j);

				// Check if mdi and mdj are abstract methods
				// with same signature and different return types

				if (Modifiers.isAbstract(mdi.modifiers)
						&& Modifiers.isAbstract(mdj.modifiers)
						&& Types.isSameMethodSig(mdi, mdj)
						&& !Types.isSameType(mdi.returnType, mdj.returnType)) {
					ErrorSet.error(decl.loc, "Class " + decl.id
							+ " contains two abstract methods"
							+ " with same signature"
							+ " but different return types");
				}
			}
		}
		// All done
	}

	/**
	 * Check that the modifiers of a type are ok.
	 * <p>
	 * 
	 * decl is the TypeDecl for the type, and currentSig its TypeSig.
	 * <p>
	 * 
	 * isClass should be true iff the TypeDecl is a ClassDecl.
	 * <p>
	 */
	public void checkTypeModifiers(/*@ non_null @*/TypeDecl decl,
	/*@ non_null @*/TypeSig currentSig, boolean isClass) {
		/*
		 * Check abstract modifier:
		 * 
		 * allowed unless is a final class
		 */
		if (isClass && Modifiers.isAbstract(decl.modifiers)
				&& Modifiers.isFinal(decl.modifiers))
			ErrorSet.error(decl.loc,
					"A class cannot be declared both final and abstract");

		/*
		 * Check final modifier:
		 * 
		 * allowed on any non-interface
		 */
		if (!isClass)
			checkModifiers(decl.modifiers, ~(Modifiers.ACC_FINAL), decl.loc,
					"interface");

		/*
		 * Handle rest of non-member types: only abstract, final, and strictfp
		 * and allowed as their modifiers.
		 */
		if (!currentSig.member) {
			checkModifiers(decl.modifiers, Modifiers.ACC_FINAL
					| Modifiers.ACC_ABSTRACT | Modifiers.ACC_STRICT, decl.loc,
					"non-member type");
			return;
		}

		// Only have to worry about member types from now on:

		/*
		 * Check access modifiers (public,protected,private):
		 * 
		 * They are not allowed on non-member types; only public is allowed for
		 * package-member types; fields of interfaces may not be protected or
		 * private
		 */
		boolean isInterfaceMember = false;
		if (currentSig.enclosingType == null) {
			checkModifiers(decl.modifiers,
					~(Modifiers.ACC_PROTECTED | Modifiers.ACC_PRIVATE),
					decl.loc, "package-member type");
		} else {
			isInterfaceMember = (currentSig.enclosingType.getTypeDecl()) instanceof InterfaceDecl;
			if (isInterfaceMember)
				checkModifiers(decl.modifiers,
						~(Modifiers.ACC_PROTECTED | Modifiers.ACC_PRIVATE),
						decl.loc, "interface member");
		}

		/*
		 * Check static modifier:
		 * 
		 * Not allowed on package-member types. Otherwise only allowed for
		 * members of top-level types and interfaces.
		 */
		if (currentSig.enclosingType == null) {
			if ((decl.modifiers & Modifiers.ACC_STATIC) != 0)
				ErrorSet.error(decl.loc,
						"Package-member types cannot be declared static");
		} else {
			boolean staticAllowed = isInterfaceMember;
			if (currentSig.enclosingType.isTopLevelType())
				staticAllowed = true;
			if (Modifiers.isStatic(decl.modifiers) && !staticAllowed)
				ErrorSet.error(decl.loc, "Nested type " + currentSig
						+ " cannot be static;"
						+ " static types can be members of only"
						+ " interfaces and top-level classes.");

			// Interfaces are implicitly static:
			if (!isClass) {
				if (!Modifiers.isStatic(decl.modifiers) && !staticAllowed)
					ErrorSet
							.error(
									decl.loc,
									"Static types can be members of only interfaces "
											+ "and top-level classes.  (Interfaces are "
											+ "implicitly static.)");

				decl.modifiers |= Modifiers.ACC_STATIC;
			}
		}

		/*
		 * The following modifiers are disallowed for all types:
		 */
		checkModifiers(decl.modifiers,
				~(Modifiers.ACC_SYNCHRONIZED | Modifiers.ACC_VOLATILE
						| Modifiers.ACC_TRANSIENT | Modifiers.ACC_NATIVE),
				decl.loc, "type");
	}

	// ----------------------------------------------------------------------

	/**
	 * Does signature-level checking and adds type members to fieldSeq and
	 * methodSeq.
	 */

	//@ requires decl != null && currentSig != null;
	protected void visitInterfaceDecl(InterfaceDecl decl, TypeSig currentSig) {
		Info.out("[prepping enclosed interface " + decl.id + "]");

		// Check that the modifiers are ok
		checkTypeModifiers(decl, currentSig, false);

		// Visit all enclosed member declarations
		// They will add themselves to fieldSeq and methodSeq

		for (int i = 0; i < decl.elems.size(); i++) {
			TypeDeclElem elem = decl.elems.elementAt(i);
			//@ assume elem.hasParent; // "invariant"
			visitTypeDeclElem(elem, currentSig, true, false, true);
		}

		checkSuperInterfaces(currentSig, decl.superInterfaces);

		/*
		 * Interfaces with no superinterface fields inherit from the root
		 * interface unless they are the root interface itself...
		 */
		if (decl.superInterfaces.size() == 0) {
			TypeSig root = getRootInterface();
			if (root != currentSig)
				addInheritedMembers(currentSig, root);
		}

		// ### STILL NEED TO CHECK NO DUPLICATE METHOD SIGNATURES ???

		// All done
	}

	// ----------------------------------------------------------------------

	protected void startSupers() {
		numFields = fieldSeq.size();
	}

	/**
	 * Check superinterfaces and add their members to fieldSeq and methodSeq.
	 */

	// Precondition: all the types in superinterfaces have been resolved
	//@ requires sig != null && superInterfaces != null;
	protected void checkSuperInterfaces(TypeSig sig, TypeNameVec superInterfaces) {

		// Check no duplicates in superinterfaces
		// Check all TypeNames are interfaces
		// Add inherited methods to fieldSeq and methodSeq

		Hashtable t = new Hashtable(10);
		for (int i = 0; i < superInterfaces.size(); i++) {
			TypeName superInterfaceName = superInterfaces.elementAt(i);
			TypeSig superInterface = TypeSig.getSig(superInterfaceName);
			int loc = superInterfaceName.name.getStartLoc();

			checkSuperTypeAccessible(sig, superInterface, loc);

			if (t.containsKey(superInterface))
				ErrorSet.error(loc, "Duplicate interface "
						+ superInterfaceName.name.printName());
			else if (!(superInterface.getTypeDecl() instanceof InterfaceDecl))
				ErrorSet.error(loc, "Interfaces may not extend classes: class "
						+ superInterface.getExternalName());
			else {
				t.put(superInterface, superInterface);
				addInheritedMembers(sig, superInterface);
			}
		}
	}

	// ----------------------------------------------------------------------

	/**
	 * Visit a TypeDeclElem, check it and add it to fieldSeq or methodSeq, if
	 * appropriate
	 */

	//@ requires e.hasParent;
	protected void visitTypeDeclElem(/*@ non_null @*/TypeDeclElem e,
	/*@ non_null @*/TypeSig currentSig, boolean abstractMethodsOK,
			boolean inFinalClass, boolean inInterface) {

		if (e instanceof FieldDecl)
			visitFieldDecl((FieldDecl) e, currentSig, abstractMethodsOK,
					inFinalClass, inInterface);

		else if (e instanceof MethodDecl)
			visitMethodDecl((MethodDecl) e, currentSig, abstractMethodsOK,
					inFinalClass, inInterface);

		else if (e instanceof ConstructorDecl)
			visitConstructorDecl((ConstructorDecl) e, currentSig,
					abstractMethodsOK, inFinalClass, inInterface);
		else if (e instanceof TypeDecl) {
			/*
			 * Check for duplicates; only complain about 2nd and later ones
			 */
			TypeDecl decl = (TypeDecl) e;
			TypeDecl parent = e.getParent();

			// Find first declaration with same id:
			TypeDecl first = decl;
			for (int i = 0; i < parent.elems.size(); i++) {
				TypeDeclElem tde = parent.elems.elementAt(i);
				if (tde instanceof TypeDecl && ((TypeDecl) tde).id == first.id) {
					first = (TypeDecl) tde;
					break;
				}
			}

			if (first != decl)
				ErrorSet
						.error(decl.locId,
								"Duplicate nested-type declaration: the type "
										+ TypeSig.getSig(decl)
										+ " is already declared", first.locId);
		} else if (e instanceof InitBlock) {
			InitBlock x = (InitBlock) e;

			if (Modifiers.isStatic(x.modifiers) && !inInterface
					&& !currentSig.isTopLevelType())
				ErrorSet.error(x.getStartLoc(), "Only top-level classes may"
						+ " contain static initializer blocks");
		} else if (!(e instanceof TypeDeclElemPragma))
			Assert.fail("Unexpected TypeDeclElem");

	}

	// ----------------------------------------------------------------------

	/** Visit a FieldDecl, check it and add it to fieldSeq. */

	//@ requires x != null && currentSig != null;
	protected void visitFieldDecl(FieldDecl x, TypeSig currentSig,
			boolean abstractMethodsOK, boolean inFinalClass, boolean inInterface) {

		if (Modifiers.isStatic(x.modifiers) && !inInterface
				&& !currentSig.isTopLevelType()
				&& !Modifiers.isFinal(x.modifiers))
			ErrorSet.error(x.locId,
					"Inner classes may not declare static members, unless they are"
							+ " compile-time constant fields");

		checkModifiers(x.modifiers, inInterface ? INTERFACE_MODIFIERS
				: ALLOWED_MODIFIERS, x.locId,
				inInterface ? "interface field" : "field");

		// If in an interface, field declaration is implicitly
		// public static final

		if (inInterface)
			x.modifiers |= INTERFACE_MODIFIERS;

		if (Modifiers.isFinal(x.modifiers) && Modifiers.isVolatile(x.modifiers))
			ErrorSet.error(x.locId, "final fields cannot be volatile");

		// Error if two fields in type body with same name
		for (int i = 0; i < fieldSeq.size(); i++) {
			FieldDecl e = (FieldDecl) fieldSeq.elementAt(i);
			if (e.id == x.id)
				ErrorSet.error(x.locId, "Duplicate field with same identifier",
						e.locId);
		}

		getEnvForCurrentSig(currentSig, true).resolveType(currentSig, x.type);

		fieldSeq.addElement(x);
	}

	// ----------------------------------------------------------------------

	/** Visit a MethodDecl, check it and add it to methodSeq. */

	//@ requires x != null && currentSig != null;
	protected void visitMethodDecl(MethodDecl x, TypeSig currentSig,
			boolean abstractMethodsOK, boolean inFinalClass, boolean inInterface) {

		if (Modifiers.isStatic(x.modifiers) && !inInterface
				&& !currentSig.isTopLevelType())
			ErrorSet.error(x.locId,
					"Only methods of top-level classes may be static");

		// Modifiers can only be:
		// public protected private abstract static final synchronized native
		// strictfp
		checkModifiers(
				x.modifiers,
				inInterface ? (Modifiers.ACC_PUBLIC | Modifiers.ACC_ABSTRACT)
						: (Modifiers.ACCESS_MODIFIERS | Modifiers.ACC_ABSTRACT
								| Modifiers.ACC_FINAL
								| Modifiers.ACC_SYNCHRONIZED
								| Modifiers.ACC_STATIC | Modifiers.ACC_NATIVE | Modifiers.ACC_STRICT),
				x.loc, inInterface ? "interface method" : "method");

		// If in interface, method is implicitly abstract and public
		if (inInterface)
			x.modifiers |= Modifiers.ACC_ABSTRACT | Modifiers.ACC_PUBLIC;

		// private methods implicitly final
		// members of final class are implicitly final
		if (Modifiers.isPrivate(x.modifiers) || inFinalClass)
			x.modifiers |= Modifiers.ACC_FINAL;

		// Error if an abstract method is also
		// private, static, final, native, or synchronized.
		if (Modifiers.isAbstract(x.modifiers)
				&& (Modifiers.isPrivate(x.modifiers)
						| Modifiers.isStatic(x.modifiers)
						| Modifiers.isFinal(x.modifiers)
						| Modifiers.isNative(x.modifiers) | Modifiers
						.isSynchronized(x.modifiers)))
			ErrorSet.error(x.locId,
					"Incompatible modifiers for abstract method");

		// resolve types
		getEnvForCurrentSig(currentSig, true).resolveType(currentSig,
				x.returnType);
		for (int i = 0; i < x.raises.size(); i++)
			getEnvForCurrentSig(currentSig, true).resolveType(currentSig,
					x.raises.elementAt(i));
		for (int i = 0; i < x.args.size(); i++)
			getEnvForCurrentSig(currentSig, true).resolveType(currentSig,
					x.args.elementAt(i).type);

		// Error if two methods in type body with same signature
		for (int i = 0; i < methodSeq.size(); i++) {
			if (Types.isSameMethodSig(x, (MethodDecl) methodSeq.elementAt(i)))
				ErrorSet.error(x.loc, "Duplicate declaration of method "
						+ "with same signature");
		}

		methodSeq.addElement(x);
	}

	// ----------------------------------------------------------------------

	/** Visit ConstructorDecl, check it, and add it to constructorSeq. */

	//@ requires x != null && currentSig != null;
	protected void visitConstructorDecl(ConstructorDecl x, TypeSig currentSig,
			boolean abstractMethodsOK, boolean inFinalClass, boolean inInterface) {

		if (inInterface)
			ErrorSet.error("Interfaces cannot include a constructor");

		// Modifiers can only be: public protected private strictfp
		checkModifiers(x.modifiers, Modifiers.ACCESS_MODIFIERS
				| Modifiers.ACC_STRICT, x.loc, "constructor");

		// resolve types
		for (int i = 0; i < x.raises.size(); i++)
			getEnvForCurrentSig(currentSig, true).resolveType(currentSig,
					x.raises.elementAt(i));
		for (int i = 0; i < x.args.size(); i++) {
			getEnvForCurrentSig(currentSig, true).resolveType(currentSig,
					x.args.elementAt(i).type);
		}

		// Error if two constructors in type body with same signature
		for (int i = 0; i < constructorSeq.size(); i++) {
			ConstructorDecl cd = (ConstructorDecl) constructorSeq.elementAt(i);
			if (Types.isSameFormalParaDeclVec(x.args, cd.args))
				ErrorSet.error(x.loc, "Duplicate declaration of constructor "
						+ "with same signature", cd.loc);
		}

		constructorSeq.addElement(x);
	}

	// *********************************************************************

	/**
	 * Find all members of a supertype inherited by a type. Adds these members
	 * to fieldSeq and methodSeq.
	 * 
	 * The order in which superTypes are added is crucial. See the comment below
	 * marked by a <<>>
	 */

	//@ requires type != null && superType != null;
	protected void addInheritedMembers(TypeSig type, TypeSig superType) {

		TypeDecl jLOTypeDecl = Types.javaLangObject().getTypeDecl();

		FieldDeclVec superFields = superType.getFields(false);
		for (int i = 0; i < superFields.size(); i++) {
			FieldDecl superField = superFields.elementAt(i);
			//@ assume superField.hasParent; // "ensures"

			// A type inherits from its direct supertypes all fields
			// that are accessible and not hidden ( JLS 8.3 and 9.2)
			// If multiple paths by which same field declaration
			// could be inherited, then it is inherited only once.

			if (fieldSeq.contains(superField)
					|| hiddenfieldSeq.contains(superField)) {
				// field already included by inheritance through another
				// interface
			} else if (superMemberAccessible(type, superType,
					superField.modifiers, superField.pmodifiers)
					&& !declaresFieldJava(type, superField.id)) {
				fieldSeq.addElement(superField);
			} else {
				hiddenfieldSeq.addElement(superField);
			}
		}
		superFields = superType.getHiddenFields();
		for (int i = 0; i < superFields.size(); i++) {
			hiddenfieldSeq.addElement(superFields.elementAt(i));
		}

		MethodDeclVec superMethods = superType.getMethods();
		for (int i = 0; i < superMethods.size(); i++) {
			MethodDecl superMethod = superMethods.elementAt(i);
			//@ assume superMethod.hasParent; // "ensures"

			/*
			 * <<>>
			 * 
			 * If superType is an interface, ignore any method's it may have
			 * inherited from java.lang.Object; The methods of java.lang.Object
			 * will either already been added via a superclass (if type is a
			 * class) or will be added separately at the end (if type is an
			 * interface).
			 * 
			 * This ensures that the methods of java.lang.Object inherited from
			 * an interface never override anything.
			 */
			if (superMethod.parent == jLOTypeDecl
					&& (superType.getTypeDecl() instanceof InterfaceDecl))
				continue;

			// a class inherits from its direct supertypes all methods
			// that are accessible and not overridden nor hidden

			// Override (8.4.6.1) if declare instance method with same signature
			// Hide (8.4.6.2) if declare static method with same signature

			// If inheriting an abstract method,
			// and class already has inherited non-abstract method
			// with same sig, then the abstract method is overridden
			// by the non-abstract method

			if (superMemberAccessible(type, superType, superMethod.modifiers,
					superMethod.pmodifiers)) {

				// Extract signature to check if overridden or hidden
				Type[] argTypes = Types.getFormalParaTypes(superMethod.args);

				MethodDecl overridingMethod = declaresMethod(type,
						superMethod.id, argTypes);

				if (overridingMethod == null) {
					// The decl does not declare such a method

					if (Modifiers.isAbstract(superMethod.modifiers)) {

						// Check if overridden by concrete method in Vec
						for (int k = 0; k < methodSeq.size(); k++) {
							MethodDecl md = (MethodDecl) methodSeq.elementAt(k);
							//@ assume md.hasParent; // "invariant"

							if (!Modifiers.isAbstract(md.modifiers)
									&& Types.isSameMethodSig(md, superMethod)) {
								// This abstract method is overridden
								overridingMethod = md;
							}
						}
					}
				}

				if (overridingMethod == null) {
					// Not overridden
					methodSeq.addElement(superMethod);
				} else {
					// Variety of checks between overridden and overriding
					// methods
					if (Modifiers.isFinal(superMethod.modifiers))
						ErrorSet.error(overridingMethod.loc,
								"Attempt to override or hide a final method");

					if (Modifiers.isStatic(overridingMethod.modifiers)
							&& !Modifiers.isStatic(superMethod.modifiers))
						ErrorSet.error(overridingMethod.loc,
								"Static method hides an instance method");

					if (!Modifiers.isStatic(overridingMethod.modifiers)
							&& Modifiers.isStatic(superMethod.modifiers))
						ErrorSet.error(overridingMethod.loc,
								"Instance method overrides a static method");

					if (!Types.isSameType(superMethod.returnType,
							overridingMethod.returnType)) {

						TypeSig typeSig = TypeSig
								.getSig((TypeName) overridingMethod.returnType);

						// Allow covariant inheritance in Java 1.5
						if (!typeSig.isSubtypeOf(TypeSig
								.getSig((TypeName) superMethod.returnType))
								|| !isRecentJava()) {

							ErrorSet.error(overridingMethod.loc,
									"Different or incompatible return types"
											+ " on overridden(hidden) and "
											+ "overriding(hiding) methods");
						}
					}

					if (!Types.isCompatibleRaises(overridingMethod.raises,
							superMethod.raises)
							&& !(superMethod.parent.isBinary() || overridingMethod.parent
									.isBinary()))
						ErrorSet
								.error(
										overridingMethod.loc,
										"Incompatible throws clauses between "
												+ "overridden(hidden) and overriding(hiding) methods");

					if (!Types.isCompatibleAccess(overridingMethod.modifiers
							& Modifiers.ACCESS_MODIFIERS, superMethod.modifiers
							& Modifiers.ACCESS_MODIFIERS)) {
						ErrorSet
								.error(
										overridingMethod.loc,
										"Incompatible access modifiers between "
												+ "overridden(hidden) ["
												+ Modifiers
														.toString(superMethod.modifiers
																& Modifiers.ACCESS_MODIFIERS)
												+ ", "
												+ Location
														.toString(superMethod.loc)
												+ "] and overriding(hiding) methods ["
												+ Modifiers
														.toString(overridingMethod.modifiers
																& Modifiers.ACCESS_MODIFIERS)
												+ "]");
					}

					// Record that the method is overridden if it is not hidden:
					if (!Modifiers.isStatic(overridingMethod.modifiers)) {
						addOverride(overridingMethod, superMethod);
					}

				}
			}
		}
	}// end addInheritedMembers

	/** Check if a member is accessible from a direct subclass. */

	//@ requires type != null && superType != null;
	protected boolean superMemberAccessible(TypeSig type, TypeSig superType,
			int modifiers, ModifierPragmaVec pmodifiers) {

		// if private then not inherited by subclass
		// Only protected and public members are inherited by subclasses
		// in a different package

		return (!Modifiers.isPrivate(modifiers) && (type
				.inSamePackageAs(superType)
				|| Modifiers.isProtected(modifiers) || Modifiers
				.isPublic(modifiers)));
	}

	/** Check if a type declares a field. */

	//@ requires sig != null;
	protected boolean declaresFieldJava(TypeSig sig, Identifier id) {

		TypeDeclElemVec elems = sig.getTypeDecl().elems;
		for (int i = 0; i < elems.size(); i++) {
			TypeDeclElem elem = elems.elementAt(i);
			if (elem.getTag() == TagConstants.FIELDDECL
					&& ((FieldDecl) elem).id.equals(id))
				return true;
		}
		// Not found
		return false;
	}

	/** Check if a type declares a method. */

	//@ requires sig != null && \nonnullelements(argTypes);
	//@ ensures \result != null ==> \result.hasParent;
	private MethodDecl declaresMethod(TypeSig sig, Identifier id,
			Type[] argTypes) {
		TypeDeclElemVec elems = sig.getTypeDecl().elems;
		search: for (int i = 0; i < elems.size(); i++) {
			TypeDeclElem elem = elems.elementAt(i);
			//@ assume elem.hasParent; // "invariant"
			if (elem.getTag() == TagConstants.METHODDECL) {
				MethodDecl md = (MethodDecl) elem;
				if (md.id == id && md.args.size() == argTypes.length) {
					for (int j = 0; j < argTypes.length; j++) {
						if (!Types.isSameType(md.args.elementAt(j).type,
								argTypes[j]))
							// Not same sig
							continue search;
					}
					// Same sig
					return md;
				}
			}
		}

		// Not found
		return null;
	}

	// *********************************************************************

	//@ requires loc != Location.NULL;
	public void checkModifiers(int modifiers, int allowed, int loc, String decl) {
		for (int i = 0; i < Modifiers.SIZE_MODIFIER_BITSET; i++) {
			int bit = 1 << i;
			if ((modifiers & bit) != 0 && (allowed & bit) == 0)
				ErrorSet.error(loc, "Modifier '" + Modifiers.name(i)
						+ "' not allowed on " + decl + " declarations");
		}
	}

	/**
	 * Check to make sure a supertype is accessible; reports an error to
	 * ErrorSet if not.
	 * <p>
	 * 
	 * Here, supertype is a supertype of currentSig; this fact is declared at
	 * loc. E.g., loc is the location of the supertype name in the extends or
	 * implements clause of currentSig.
	 * <p>
	 */
	//@ requires loc != Location.NULL;
	public void checkSuperTypeAccessible(/*@non_null*/TypeSig currentSig,
	/*@ non_null @*/TypeSig supertype, int loc) {
		/*
		 * FIXME - this error is commented out because it incorrectly disallows
		 * using a protected nested class of a public supertype that is not in
		 * the same package. There are presumably situations this should check,
		 * but that will take some research, and in any case, we could just
		 * leave them to java. // fix for 1.1: !!!! if (!
		 * Modifiers.isPublic(supertype.getTypeDecl().modifiers) && !
		 * currentSig.inSamePackageAs(supertype)) ErrorSet.error(loc, "Supertype
		 * is not accessible.");
		 */
	}

	/**
	 * This routine constructs and returns the interface that all interfaces are
	 * de-facto subinterfaces of.
	 * <p>
	 * 
	 * This interface is not an actual Java interface, but rather a made up one.
	 * Its locations will be valid, but misleading.
	 * <p>
	 * 
	 * 
	 * The root interface is composed of all the public methods of
	 * java.lang.Object turned into abstract methods.
	 * <p>
	 */
	//@ ensures \result != null;
	public TypeSig getRootInterface() {
		if (_rootCache != null)
			return _rootCache;

		TypeSig objSig = Types.javaLangObject();
		TypeDecl object = objSig.getTypeDecl();
		objSig.prep();

		TypeDeclElemVec newElems = TypeDeclElemVec.make();
		for (int i = 0; i < object.elems.size(); i++) {
			TypeDeclElem e = object.elems.elementAt(i);
			if (!(e instanceof MethodDecl))
				continue;
			MethodDecl m = (MethodDecl) e;
			if (!Modifiers.isPublic(m.modifiers)
					|| Modifiers.isStatic(m.modifiers))
				continue;

			MethodDecl newM = MethodDecl.make(Modifiers.ACC_PUBLIC
					| Modifiers.ACC_ABSTRACT, m.pmodifiers, m.tmodifiers,
					m.args, m.raises, null /* body */, Location.NULL, m.loc,
					m.locId, m.locThrowsKeyword, m.id, m.returnType, m.locType);
			newElems.addElement(newM);
		}

		TypeDecl i = InterfaceDecl.make(Modifiers.ACC_PUBLIC,
				null /* pmodifiers */, object.id, TypeNameVec.make(), null,
				newElems, object.loc, object.locId, object.locOpenBrace,
				object.locCloseBrace);
		_rootCache = Types.makeTypeSig(objSig.packageName,
				"<the root interface>", null, i, objSig.getCompilationUnit());

		// PrettyPrint.inst.print(System.out, 0, i);

		return _rootCache;
	}

	protected TypeSig _rootCache = null;

	//@ ensures \result != null;
	public javafe.tc.TypeSig processTypeNameAnnotations(
	/*@ non_null @*/TypeName tn,
	/*@ non_null @*/javafe.tc.TypeSig sig, Env env) {
		return sig;
	}

	//@ ensures \result != null;
	protected EnvForTypeSig getEnvForCurrentSig(/*@non_null*/TypeSig sig,
			boolean isStatic) {
		return sig.getEnv(isStatic);
	}

	public static boolean isRecentJava() {
	  String version = System.getProperty("java.version");
	  return (version.indexOf("1.5") != -1) || 
            (version.indexOf("1.6") != -1) || (version.indexOf("1.7") != -1);
	}
}
