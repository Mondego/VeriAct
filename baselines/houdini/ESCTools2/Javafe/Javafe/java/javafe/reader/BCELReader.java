// $Id$
/* Copyright 2007, Systems Research Group, University College Dublin, Ireland */

/**
 * ========================================================================= 
 * BCELReader.java - derived from BinReader and ASTClassFileParser
 * =========================================================================
 */

package javafe.reader;

import java.io.IOException;
import java.io.InputStream;
import java.util.StringTokenizer;
import java.util.Vector;

import javafe.ast.BlockStmt;
import javafe.ast.ClassDecl;
import javafe.ast.CompilationUnit;
import javafe.ast.ConstructorDecl;
import javafe.ast.FieldDecl;
import javafe.ast.FormalParaDecl;
import javafe.ast.FormalParaDeclVec;
import javafe.ast.Identifier;
import javafe.ast.IdentifierVec;
import javafe.ast.ImportDeclVec;
import javafe.ast.InterfaceDecl;
import javafe.ast.JavafePrimitiveType;
import javafe.ast.LiteralExpr;
import javafe.ast.MethodDecl;
import javafe.ast.Modifiers;
import javafe.ast.Name;
import javafe.ast.PrettyPrint;
import javafe.ast.RoutineDecl;
import javafe.ast.StmtVec;
import javafe.ast.ASTTagConstants;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElem;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.TypeDeclVec;
import javafe.ast.TypeName;
import javafe.ast.TypeNameVec;
import javafe.genericfile.GenericFile;
import javafe.genericfile.NormalGenericFile;
import javafe.util.ErrorSet;
import javafe.util.Info;
import javafe.util.Location;

import org.apache.bcel.Constants;
import org.apache.bcel.classfile.Attribute;
import org.apache.bcel.classfile.ClassFormatException;
import org.apache.bcel.classfile.ClassParser;
import org.apache.bcel.classfile.Constant;
import org.apache.bcel.classfile.ConstantDouble;
import org.apache.bcel.classfile.ConstantFloat;
import org.apache.bcel.classfile.ConstantInteger;
import org.apache.bcel.classfile.ConstantLong;
import org.apache.bcel.classfile.ConstantPool;
import org.apache.bcel.classfile.ConstantString;
import org.apache.bcel.classfile.ConstantUtf8;
import org.apache.bcel.classfile.ConstantValue;
import org.apache.bcel.classfile.ExceptionTable;
import org.apache.bcel.classfile.Field;
import org.apache.bcel.classfile.InnerClass;
import org.apache.bcel.classfile.InnerClasses;
import org.apache.bcel.classfile.JavaClass;
import org.apache.bcel.classfile.Method;
import org.apache.bcel.generic.Type;

/**
 * -------------------------------------------------------------------------
 * BCELReader
 * -------------------------------------------------------------------------
 */

/**
 * Parses the contents of a class file into an AST for the purpose of type
 * checking. Ignores components of the class file that have no relevance to type
 * checking (e.g. method bodies).
 * 
 * @author Dermot Cochran
 */

class BCELReader extends Reader {

	/** Access flags which we are not parsing at the moment */
	public static final int HIDDEN_ACCESS_FLAGS = 
		~Constants.ACC_TRANSIENT & 
		~Constants.ACC_VOLATILE & 
		~Constants.ACC_ANNOTATION & 
		~Constants.ACC_SYNCHRONIZED;

	/** The package name of the class being parsed. */
	public Name classPackageName;

	/**
	 * A dummy location representing the class being parsed. Initialized by
	 * constructor.
	 */
	//@ invariant binaryClassLocation != Location.NULL;
	public int binaryClassLocation;

	/**
	 * The BCEL representation of the bytecode
	 */
	protected JavaClass javaClass;

	/**
	 * Add only AST nodes that are not synthetic decls to v. nodes should be an
	 * array of TypeDeclElems. A synthetic decl is one that had the synthetic
	 * attribute, or is a static method decl for an interface.
	 */
	/*@
	  @ protected normal_behavior
	  @	  ensures \old(typeDeclElemVec.size()) <= typeDeclElemVec.size();
	  @   ensures typeDeclElemVec.size() <= typeDeclElems.length + \old(typeDeclElemVec.size());
	  @*/
	protected void addElements(
	/*@ non_null */ TypeDeclElemVec typeDeclElemVec,
	/*@ non_null */ TypeDeclElem[] typeDeclElems) {
		for (int i = 0; i < typeDeclElems.length; i++) {
			if ((javaClass.isInterface())
					&& typeDeclElems[i] instanceof RoutineDecl) {
				RoutineDecl rd = (RoutineDecl) typeDeclElems[i];

				if (Modifiers.isStatic(rd.modifiers)) {
					continue;
				}
			}
			if (omitPrivateFields && typeDeclElems[i] instanceof FieldDecl) {
				if (Modifiers
						.isPrivate(((FieldDecl) typeDeclElems[i]).modifiers)) {
					continue;
				}
			}
			typeDeclElemVec.addElement(typeDeclElems[i]); //@ nowarn Pre;
		}
	}

	/**
	 * Vector of methods and fields with Synthetic attributes. Use this to weed
	 * out synthetic while constructing TypeDecl.
	 */
	protected /*@ non_null */ Vector synthetics;

	/**
	 * Flag indicating whether the class being parsed has the synthetic
	 * attribute.
	 */
	/*@spec_public*/ private boolean syntheticClass;

	/**
	 * Flag indicates that class bodies are currently being parsed, otherwise 
   * only the type interface is parsed.
	 */
	protected boolean includeBodies;

	/**
	 * A flag indicating that parsed type declaration should ignore/omit 
   * private class fields.
	 */
	protected boolean omitPrivateFields;

	/**
	 * Default constructor - initialises some instance variables
	 */
	public BCELReader() {
		syntheticClass = false;
		synthetics = new Vector();
		omitPrivateFields = true;
	}

	/**
	 * Binary inner class constructors have an extra initial argument to their
	 * constructors (the enclosing class). This is not present in the source
	 * file. To make the AST generated by reading the binary correspond to that
	 * obtained from a source file, we remove that extra argument for each inner
	 * (non-static) class. Each nested class does this for its own direct inner
	 * classes. - DRCok
	 * 
	 * @param typeDecl Type declaration of the bytecode
	 */
	protected void removeExtraArg(TypeDecl typeDecl) {
		TypeDeclElemVec typeDeclElemVec = typeDecl.elems;
		for (int loopVar = 0; loopVar < typeDeclElemVec.size(); ++loopVar) {
			if (!(typeDeclElemVec.elementAt(loopVar) instanceof ClassDecl))
				continue;
			ClassDecl classDecl = (ClassDecl) typeDeclElemVec
					.elementAt(loopVar);
			if (Modifiers.isStatic(classDecl.modifiers))
				continue;
			TypeDeclElemVec classDeclElemVec = classDecl.elems;
			for (int classDeclLoopVar = 0; classDeclLoopVar < classDeclElemVec
					.size(); ++classDeclLoopVar) {
				if (!(classDeclElemVec.elementAt(classDeclLoopVar) instanceof ConstructorDecl))
					continue;
				ConstructorDecl constructorDecl = (ConstructorDecl) classDeclElemVec
						.elementAt(classDeclLoopVar);
				constructorDecl.args.removeElementAt(0);
			}
		}
	}

	/**
	 * Convert the BCEL JavaClass format into an abstract syntax tree of a
	 * compilation unit suitable for extended static checking.
	 * 
	 * @return An abstract syntax tree of a compilation unit.
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	protected CompilationUnit getCompilationUnit()
			throws ClassNotFoundException, IOException {

		// Make the type declaration
		TypeDecl typeDecl = makeTypeDecl();
		
		// Make the compilation unit
		TypeDeclVec types = TypeDeclVec.make(new TypeDecl[] { typeDecl });
		TypeDeclElemVec emptyTypeDeclElemVec = TypeDeclElemVec.make();
		ImportDeclVec emptyImportDeclVec = ImportDeclVec.make();
		CompilationUnit result = CompilationUnit.make(classPackageName, null,
				emptyImportDeclVec, types, binaryClassLocation, emptyTypeDeclElemVec);

		return result;
	}

	/**
	 * Construct the type declaration
	 * 
	 * @throws ClassNotFoundException
	 * @throws IOException 
	 * @throws ClassFormatError
	 */
	/*@ protected normal_behavior
	  @   ensures \result.isBinary();
	  @*/
	protected TypeDecl makeTypeDecl() throws IOException, ClassNotFoundException {

		constantPool = javaClass.getConstantPool();
		// N.B. This gets the class index not the class name index
		classIndex = javaClass.getClassNameIndex(); 
		setClassNameAndLocation(binaryClassLocation);
		readClassAttributes();
		setSuperClassName();
		String packageName = javaClass.getPackageName();
		if (packageName.length() > 0) {
			classPackageName = Name.make(packageName, binaryClassLocation);
		}

		// Read interface names
		TypeNameVec interfaceVector = null;
		JavaClass[] interfaces;
		try {
			interfaces = javaClass.getInterfaces();
			int numberOfInterfaces = interfaces.length;
			TypeName[] interfaceTypeNames = new TypeName[interfaces.length];
		
			for (int interfaceLoopVar = 0; interfaceLoopVar < numberOfInterfaces; interfaceLoopVar++) {
				JavaClass interfaceInst = interfaces[interfaceLoopVar];
				String interfaceName = interfaceInst.getClassName();
				interfaceTypeNames[interfaceLoopVar] = getCompoundTypeName(interfaceName);
			}
			
			interfaceVector =
			    TypeNameVec.make(interfaceTypeNames);
			
		} catch (ClassNotFoundException e) {
			// Missing classpath entry
			ErrorSet.fatal(binaryClassLocation, "Incomplete classpath: " + e.getLocalizedMessage() + 
					" CLASSPATH is " + System.getenv("CLASSPATH")+ ":"+ System.getenv("ESC_CLASSPATH"));
		}
		 
		
		Method[] methods = javaClass.getMethods();
		RoutineDecl[] routineDecl = readMethods(methods);

		Field[] fields = javaClass.getFields();
		FieldDecl[] fieldDecl = getFieldDecl(fields);

		int predict = classMembers.size() + methods.length + fields.length;
		TypeDeclElemVec elementVec = TypeDeclElemVec.make(predict);

		elementVec.append(classMembers);

		// only add routines and fields that are not synthetic.
		addElements(elementVec, routineDecl);
		addElements(elementVec, fieldDecl);

		// The synchronized bit for classes is used for other purposes
		int accessModifiers = javaClass.getAccessFlags()
				& ~Constants.ACC_SYNCHRONIZED;

		TypeDecl typeDecl;
		if (javaClass.isInterface()) {
			typeDecl = (TypeDecl) InterfaceDecl.make(accessModifiers, null,
					canonicalClassIdentifier, interfaceVector, null, elementVec,
					binaryClassLocation, binaryClassLocation, binaryClassLocation, binaryClassLocation);
		} else {
			typeDecl = (TypeDecl) ClassDecl.make(accessModifiers, null,
					canonicalClassIdentifier, interfaceVector, null, elementVec,
					binaryClassLocation, binaryClassLocation, binaryClassLocation, binaryClassLocation,
					super_class);
		}
		typeDecl.specOnly = true;
		
		// Remove extra constructor used for binary inner classes
		removeExtraArg(typeDecl);

		return typeDecl;
	}

	/**
	 * Read field declaration
	 * 
	 * @param fields
	 *            Array of fields in BCEL
	 * @return Field declaration
	 */
	protected FieldDecl[] getFieldDecl(Field[] fields) {
		int numberOfFieldsInClass = fields.length;
		FieldDecl[] fieldDecl = new FieldDecl[numberOfFieldsInClass];

		for (int loopIndex = 0; loopIndex < numberOfFieldsInClass; loopIndex++) {

			Field field = fields[loopIndex];
			String fieldName = field.getName();
			javafe.ast.Type fieldType = getFieldType(field);

			int fieldModifiers = field.getAccessFlags()
					& ~Constants.ACC_SYNTHETIC & ~Constants.ACC_ANNOTATION;

			fieldDecl[loopIndex] = //@ nowarn IndexTooBig;
			FieldDecl.make(fieldModifiers, null, Identifier.intern(fieldName),
					fieldType, binaryClassLocation, null, binaryClassLocation);

			// Look for attributes which indicate a constant value
			int fieldTypeTag = fieldDecl[loopIndex].type.getTag();
			Attribute[] attributes = field.getAttributes();
			for (int attrLoopVar = 0; attrLoopVar < attributes.length; attrLoopVar++) {

				// Read attribute name tag
				byte attributeTag = getAttributeTag(attributes, attrLoopVar);

				if (attributeTag == Constants.ATTR_CONSTANT_VALUE) {

					// Set field initializer
					ConstantValue constantValue = (ConstantValue) attributes[attrLoopVar];
					LiteralExpr fieldInitializer = readFieldInitializer(
							fieldTypeTag, constantValue);
					fieldDecl[loopIndex].init = fieldInitializer;
				}
			}
		}
		return fieldDecl;
	}

	/**
	 * Read the constant value of the field initializer
	 * 
	 * @param fieldTypeTag
	 * @param constantValue
	 * 
	 * @return A literal expression for the field identifier
	 */
	protected LiteralExpr readFieldInitializer(int fieldTypeTag,
			ConstantValue constantValue) {
		int tag;
		Object literal;
		int constantValueIndex = constantValue.getConstantValueIndex();

		switch (fieldTypeTag) {
		case ASTTagConstants.BOOLEANTYPE:
			tag = ASTTagConstants.BOOLEANLIT;
			ConstantInteger constantForBoolean = (ConstantInteger) constantPool
					.getConstant(constantValueIndex);
			literal = Boolean.valueOf(constantForBoolean.getBytes() != 0);
			break;

		case ASTTagConstants.INTTYPE:
		case ASTTagConstants.BYTETYPE:
		case ASTTagConstants.SHORTTYPE:
			tag = ASTTagConstants.INTLIT;
			ConstantInteger constantInteger = (ConstantInteger) constantPool
					.getConstant(constantValueIndex);
			literal = new Integer(constantInteger.getBytes());
			break;

		case ASTTagConstants.LONGTYPE:
			tag = ASTTagConstants.LONGLIT;
			ConstantLong constantLong = (ConstantLong) constantPool
					.getConstant(constantValueIndex);
			literal = new Long(constantLong.getBytes());
			break;

		case ASTTagConstants.CHARTYPE:
			tag = ASTTagConstants.CHARLIT;
			ConstantInteger constantForChar = (ConstantInteger) constantPool
					.getConstant(constantValueIndex);
			literal = new Integer(constantForChar.getBytes());
			break;

		case ASTTagConstants.FLOATTYPE:
			tag = ASTTagConstants.FLOATLIT;
			ConstantFloat constantFloat = (ConstantFloat) constantPool
					.getConstant(constantValueIndex);
			literal = new Float(constantFloat.getBytes());
			break;

		case ASTTagConstants.DOUBLETYPE:
			tag = ASTTagConstants.DOUBLELIT;
			ConstantDouble constantDouble = (ConstantDouble) constantPool
					.getConstant(constantValueIndex);
			literal = new Double(constantDouble.getBytes());
			break;

		default:
			tag = ASTTagConstants.STRINGLIT;
			ConstantString constantString = (ConstantString) constantPool
					.getConstant(constantValueIndex);
			int constantStringIndex = constantString.getStringIndex();
			ConstantUtf8 constantUtf8 = (ConstantUtf8) constantPool
					.getConstant(constantStringIndex);

			literal = String.valueOf(constantUtf8.getBytes());
			break;
		}

		LiteralExpr fieldInitializer = LiteralExpr.make(tag, literal,
				binaryClassLocation);
		return fieldInitializer;
	}

	/**
	 * Add class methods to abstract syntax tree
	 * 
	 * @param methods array of methods in BCEL.
	 * @throws ClassNotFoundException.
	 */
	protected RoutineDecl[] readMethods(Method[] methods) throws ClassNotFoundException {

		// Get total number of methods including synthetics
		int numberOfMethodsInClass = methods.length;
		RoutineDecl[] tempRoutineDecl = new RoutineDecl[numberOfMethodsInClass];
		int numberOfRoutines = 0;

		for (int loopVar = 0; loopVar < numberOfMethodsInClass; loopVar++) {

			Method method = methods[loopVar];
			RoutineDecl routineDeclElement = makeRoutineDecl(method);

			// put in a dummy body
			if (includeBodies) {
				routineDeclElement.body = //@ nowarn Null, IndexTooBig;
				BlockStmt.make(StmtVec.make(), binaryClassLocation, binaryClassLocation);
				routineDeclElement.locOpenBrace = binaryClassLocation;
			}

			// Read method attributes and check for exceptions
			Attribute[] attributes = method.getAttributes();
			for (int attributeLoopVar = 0; attributeLoopVar < attributes.length; attributeLoopVar++) {

				// Read attribute name tag
				byte attributeTag = getAttributeTag(attributes,
						attributeLoopVar);

				if (attributeTag == Constants.ATTR_EXCEPTIONS) {
					ExceptionTable exceptionTable = (ExceptionTable) attributes[attributeLoopVar];

					routineDeclElement.raises = readExceptionTypeNames(exceptionTable);
				}
			}

			if (method.isSynthetic()) {
				synthetics.addElement(routineDeclElement);
			} else {
				// Add non synthetic methods to routineDecl
				tempRoutineDecl[numberOfRoutines] = routineDeclElement;
				numberOfRoutines++;
			}
		}

		// Copy and resize the array
		RoutineDecl[] routineDecl = new RoutineDecl[numberOfRoutines];
		for (int routineLoopVar = 0; routineLoopVar < numberOfRoutines; routineLoopVar++) {
			routineDecl[routineLoopVar] = tempRoutineDecl[routineLoopVar];
		}
		
		return routineDecl;
	}

	/**
	 * Get the tag of an attribute
	 * 
	 * @param attributes
	 * @param attributeLoopVar
	 * @return tag of the attribute.
	 */
	protected byte getAttributeTag(Attribute[] attributes, int attributeLoopVar) {
		Attribute attribute = attributes[attributeLoopVar];
		return attribute.getTag();
	}

	/* -- protected instance methods ----------------------------------------- */
	/**
	 * Read the class attributes
	 * 
	 * @throws ClassNotFoundException
	 * @throws IOException 
	 */
	protected void readClassAttributes() throws ClassNotFoundException, IOException {

		Attribute[] classAttributes = javaClass.getAttributes();

		for (int loopVar = 0; loopVar < classAttributes.length; loopVar++) {

			// Get attribute Name
			byte attributeTag = getAttributeTag(classAttributes, loopVar);

			if (attributeTag == Constants.ATTR_SYNTHETIC) {
				syntheticClass = true;
			} else if (attributeTag == Constants.ATTR_INNER_CLASSES) {

				InnerClasses innerClassesAttribute = (InnerClasses) classAttributes[loopVar];
				InnerClass[] innerClasses = innerClassesAttribute
						.getInnerClasses();

				for (int j = 0; j < innerClasses.length; j++) {
					InnerClass innerClass = innerClasses[j];

					// Parse the inner class
					addInnerClass(innerClass);

				}
			}
		}
	}

	/**
	 * Add the inner class to the abstract syntax tree, unless synthetic
	 * 
	 * @param innerClass
	 *            BCEL representation of the inner class
	 * @throws ClassNotFoundException
	 * @throws IOException 
	 */
	protected void addInnerClass(InnerClass innerClass)
			throws ClassNotFoundException, IOException {

		// Get inner class name
		int innerNameIndex = innerClass.getInnerNameIndex();

		int innerClassIndex = innerClass.getInnerClassIndex();
		int outerClassIndex = innerClass.getOuterClassIndex();

		// Check that this inner class is enclosed by the class being parsed
		if (outerClassIndex == this.classIndex) {

			Constant innerClassNameConstant = constantPool
					.getConstant(innerNameIndex);

			if (innerClassNameConstant != null) {
				String innerClassName = constantPool
						.constantToString(innerClassNameConstant);

				String outerClassFullName = javaClass.getClassName();
				int subStringIndex = 1 + outerClassFullName.lastIndexOf('.');

				String outerClassName = outerClassFullName
						.substring(subStringIndex);
				String innerClassNamePath = makeInnerClassFullName(
						innerClassName, outerClassName);

				GenericFile sibling = genericFile
						.getSibling(innerClassNamePath);
				BCELReader innerClassReader = readInnerClass(sibling, true);
				
				TypeDecl innerClassTypeDecl = innerClassReader.makeTypeDecl();
				
				// Set access modifier flags for inner class
				int innerAccessFlags = innerClass.getInnerAccessFlags();
				innerClassTypeDecl.modifiers = filterAccessFlags(innerAccessFlags);
				
				// Only add inner classes that are not synthetic and are not anonymous inner classes,
				// which are identified by names that start with a number
				if (!innerClassReader.isSyntheticClass()) {
				    if (!Character.isDigit(innerClassTypeDecl.id.toString().charAt(0))) {
					classMembers.addElement(innerClassTypeDecl);
					}
				}
			}
		}

	}

	/**
	 * Filter the list of access flags that will be used
	 * 
	 * @param accessFlags The original access flags
	 * @return The filtered access flags
	 */
	protected int filterAccessFlags(int accessFlags) {
		return accessFlags & HIDDEN_ACCESS_FLAGS;
	}

	/**
	 * Construct the full name for the inner class e.g. AbstractList$Itr
	 * 
	 * @param innerClassName The name of the inner class
	 * @param enclosingClassName The name of the enclosing class
	 * 
	 * @return The filename of the inner class
	 */
	protected String makeInnerClassFullName(String innerClassName,
			String enclosingClassName) {
		StringBuffer classNameBuffer = new StringBuffer(enclosingClassName);
		classNameBuffer.append("$");
		classNameBuffer.append(innerClassName);
		classNameBuffer.append(".class");
		String innerClassFileName = classNameBuffer.toString();
		return innerClassFileName;
	}

	/**
	 * Recursively read details of inner class
	 * 
	 * @param sibling
	 * @param avoidSpec
	 * 
	 * @return A reader for the inner class
	 */
	protected BCELReader readInnerClass(GenericFile sibling, boolean avoidSpec) {
		BCELReader parser = new BCELReader();
		parser.read(sibling, avoidSpec);
		return parser;
	}

	/**
	 * Parse canonical class name, package name and location of binary classfile
	 * 
	 * @param locationIndex
	 *            Location index for the binary class file
	 * 
	 */
	protected void setClassNameAndLocation(int locationIndex) {
		 
		// Get the binary class name e.g. java.lang.Character$Subset
		String binaryClassName = javaClass.getClassName();
		Info.out("Binary class name:    " + binaryClassName);
		
		// Get the canonical class name e.g. java.util.Map.Entry
		String canonicalClassName = binaryClassName.replace('$', '.');
		Info.out("Canonical class name: " + canonicalClassName);
		
		// Set class package name and location
		Name className = Name.make(canonicalClassName, locationIndex);
		classPackageName = getNameQualifier(className);
		canonicalClassIdentifier = getNameTerminal(className);
		DescriptorParser.classLocation = binaryClassLocation;
	}

	/**
	 * Set super class name
	 * 
	 */
	protected void setSuperClassName() {

		// Get the canonical super class name e.g. java.util.Map.Entry
		String superClassName = javaClass.getSuperclassName().replace('$', '.');

		Name name = Name.make(superClassName, binaryClassLocation);
		super_class = TypeName.make(name);
	}

	/**
	 * Convert BCEL field type to AST field type
	 * 
	 * @param field
	 *            BCEL representation of class or instance field
	 * @return JavaFE type of the field
	 */
	protected javafe.ast.Type getFieldType(Field field) {

		int typeTag;
		javafe.ast.Type astType;
		Type bcelFieldType = field.getType();

		switch (bcelFieldType.getType()) {
		case Constants.T_BOOLEAN:
			typeTag = ASTTagConstants.BOOLEANTYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;

		case Constants.T_BYTE:
			typeTag = ASTTagConstants.BYTETYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;

		case Constants.T_INT:
			typeTag = ASTTagConstants.INTTYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;

		case Constants.T_LONG:
			typeTag = ASTTagConstants.LONGTYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;

		case Constants.T_VOID:
			typeTag = ASTTagConstants.VOIDTYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;

		case Constants.T_DOUBLE:
			typeTag = ASTTagConstants.DOUBLETYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;

		case Constants.T_ARRAY:
			typeTag = ASTTagConstants.ARRAYTYPE;
			astType = DescriptorParser.parseField(bcelFieldType.getSignature());
			break;

		case Constants.T_FLOAT:
			typeTag = ASTTagConstants.FLOATTYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;

		case Constants.T_SHORT:
			typeTag = ASTTagConstants.SHORTTYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;
			
		case Constants.T_CHAR:
			typeTag = ASTTagConstants.CHARTYPE;
			astType = JavafePrimitiveType.make(typeTag, binaryClassLocation);
			break;
			
		// Non primitive types
		default:
			astType = DescriptorParser.parseField(bcelFieldType.getSignature());
			break;
		}

		return astType;

	}

	/**
	 * Make a method or constructor declaration
	 * 
	 * @param method
	 *            BCEL representation of the method or constructor
	 * 
	 * @return The routine declaration
	 */
	protected RoutineDecl makeRoutineDecl(Method method) {

		String methodName = method.getName();
		String methodSignature = method.getSignature();

		// Translate access modifier flags from BCEL
		int accessModifiers = method.getAccessFlags() & HIDDEN_ACCESS_FLAGS;

		MethodSignature signature = DescriptorParser
				.parseMethod(methodSignature);
		FormalParaDeclVec formalVec = FormalParaDeclVec
				.make(makeFormals(signature));
		BlockStmt body = null;

		if (methodName.equals("<init>")) {
			RoutineDecl constructor = (RoutineDecl) ConstructorDecl.make(
					accessModifiers, null, null, formalVec, emptyTypeNameVec,
					body, Location.NULL, binaryClassLocation, binaryClassLocation,
					binaryClassLocation);
			return constructor;
		} else {

			javafe.ast.Type returnType = signature
					.getReturn();
			RoutineDecl otherMethod = (RoutineDecl) MethodDecl.make(
					accessModifiers, null, null, formalVec, emptyTypeNameVec,
					body, Location.NULL, binaryClassLocation, binaryClassLocation,
					binaryClassLocation, Identifier.intern(methodName), returnType, binaryClassLocation);
			return otherMethod;
		}
	}

	/**
	 * The type name of the superclass of the class being parsed. Initialized by
	 * set_super_class.
	 */
	protected TypeName super_class;

	/**
	 * The type names of the interfaces implemented by the class being parsed.
	 * Initialized by set_num_interfaces. Elements initialized by set_interface.
	 */
	protected TypeName[] typeNames;

	/**
	 * The class members of the class being parsed. Initialized by set_field,
	 * set_method, and set_class_attributes.
	 */
	protected /*@ non_null*/ TypeDeclElemVec classMembers = TypeDeclElemVec.make(0);

	/**
	 * The identifier of the class being parsed. Initialized by set_this_class.
	 */
	//@ spec_public
	protected Identifier canonicalClassIdentifier;

	/**
	 * Parse the table of throwable exceptions
	 * 
	 * @param exceptionTable
	 *            Table of exceptions throwable
	 * 
	 * @return TypeNameVec of compomnd name form, without dots
	 * @throws ClassNotFoundException
	 */
	protected /*@non_null*/ TypeNameVec readExceptionTypeNames(ExceptionTable exceptionTable)
			throws ClassNotFoundException {

		int numberOfExceptionsThrown = exceptionTable.getNumberOfExceptions();
		String[] exceptionNames = exceptionTable.getExceptionNames();
		TypeName[] exceptionTypeNames = new TypeName[numberOfExceptionsThrown];

		for (int loopVar = 0; loopVar < numberOfExceptionsThrown; loopVar++) {

			String exceptionClassName = exceptionNames[loopVar];
			TypeName exceptionTypeName = getCompoundTypeName(exceptionClassName);
			exceptionTypeNames[loopVar] = exceptionTypeName;

		}

		TypeNameVec typeNameVec = TypeNameVec.make(exceptionTypeNames);
		return typeNameVec;
	}

	/**
	 * Get the type name as a compound name without dots
	 * 
	 * @param className
	 * 
	 * @return The type name in compound form
	 * @throws ClassFormatError
	 */
	protected TypeName getCompoundTypeName(/*@ non_null @*/ String className)
			throws ClassFormatError {

		StringTokenizer tokenizer = new StringTokenizer(className,".$");

		int count = tokenizer.countTokens();
		javafe.util.Assert.notFalse(count > 0); //@ nowarn Pre;

		/*@ non_null */ Identifier[] identifiers = new Identifier[count];
		int[] locations1 = new int[count];
		int[] locations2 = new int[count - 1];

		for (int loopVar = 0; tokenizer.hasMoreTokens(); loopVar++) {
			identifiers[loopVar] = Identifier.intern(tokenizer.nextToken());
			locations1[loopVar] = binaryClassLocation;

			if (loopVar < count - 1)
				locations2[loopVar] = binaryClassLocation;
		}
		//@ assume \nonnullelements(identifiers);
		/*
		 *@ assume (\forall int i; (0<=i && i<locations1.length) ==>
		 * locations1[i] != Location.NULL);
		 */
		/*
		 *@ assume (\forall int i; (0<=i && i<locations2.length) ==>
		 * locations2[i] != Location.NULL);
		 */

		return TypeName.make(Name.make(locations1, locations2, IdentifierVec
				.make(identifiers)));
	}

	/**
	 * Construct a vector of formal parameters from a method signature.
	 * 
	 * @param signature
	 *            the method signature to make the formal parameters from
	 * @return the formal parameters
	 */
	//@ ensures \nonnullelements(\result);
	//@ ensures \typeof(\result) == \type(FormalParaDecl[]);
	protected FormalParaDecl[] makeFormals(/*@ non_null */MethodSignature  signature) {
		int length = signature.countParameters();
		FormalParaDecl[] formals = new FormalParaDecl[length];

		for (int loopIndex = 0; loopIndex < length; loopIndex++) {
			StringBuffer identifierStringBuffer = new StringBuffer("arg");
			identifierStringBuffer.append(loopIndex);
			Identifier id = Identifier
					.intern(identifierStringBuffer.toString());
			formals[loopIndex] = FormalParaDecl.make(0, null, id, signature
					.parameterAt(loopIndex), binaryClassLocation);
		}

		return formals;
	}

	/* -- class methods ---------------------------------------------- */

	/**
	 * Return the package qualifier of a given name.
	 * 
	 * @param name
	 *            the name to return the package qualifier of
	 * @return the package qualifier of name
	 */
	protected static Name getNameQualifier(/*@ non_null @*/ Name   name) {
		int size = name.size();

		return size > 1 ? name.prefix(size - 1) : null;
		// using null for the unnamed package ???
	}

	/**
	 * Return the terminal identifier of a given name.
	 * 
	 * @param name
	 *            the name to return the terminal identifier of
	 * @return the terminal identifier of name
	 */
	protected static Identifier getNameTerminal(/*@ non_null @*/ Name   name) {
		return name.identifierAt(name.size() - 1);
	}

	/* -- class variables -------------------------------------------- */

	/**
	 * An empty type name vector.
	 */
	//@ spec_public
	protected static final /*@non_null*/ TypeNameVec emptyTypeNameVec = TypeNameVec.make();

	protected ConstantPool constantPool;

	protected int classIndex;

	protected GenericFile genericFile;

	/**
	 * Main test method
	 * 
	 * @param args
	 */
	//@ requires \nonnullelements(args);
	public static void main(/*@non_null*/ String[] args) {
		if (args.length < 1) {
			System.err.println("BCELReader: <source filename>");
			System.exit(1);
		}

		for (int loopVar = 0; loopVar < args.length; loopVar++) {
			GenericFile target = new NormalGenericFile(args[loopVar]);
			BCELReader reader;
			try {
				reader = new BCELReader();
				CompilationUnit cu = reader.read(target, false);
				if (cu != null)
					PrettyPrint.inst.print(System.out, cu);

			} catch (ClassFormatError e) {
				e.printStackTrace();
			}
		}

	}

	/**
	 * Read and parse a binary class file
	 */
	/*@ also public normal_behavior
	  @   ensures \result != null;
	  @*/
	public CompilationUnit read(/*@non_null*/ GenericFile target, boolean avoidSpec) {

		this.binaryClassLocation = Location.createWholeFileLoc(target);
		this.genericFile = target;

		// Parse the binary class file using the BCEL parser
		ClassParser classParser;
		try {
			readJavaClass(target);

			CompilationUnit compilationUnit = getCompilationUnit();

			return compilationUnit;

		} catch (ClassFormatError e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;

	}

	/**
	 * Use BCEL to read a binary classfile
	 * 
	 * @param target
	 * @throws IOException
	 * @throws ClassFormatException
	 */
	protected void readJavaClass(/*@non_null*/ GenericFile target) throws IOException,
			ClassFormatException {
		InputStream inputStream = target.getInputStream();
		String localName = target.getLocalName();
		ClassParser classParser = new ClassParser(inputStream, localName);

		this.javaClass = classParser.parse();
	}

	//@ ensures \result == syntheticClass;
	//@ modifies \nothing;
	public boolean isSyntheticClass() {
		return syntheticClass;
	}
}
