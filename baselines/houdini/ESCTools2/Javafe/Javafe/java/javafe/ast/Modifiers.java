/* Copyright 2000, 2001, Compaq Computer Corporation */
/* Modified 2007, EU Mobius Project, University College Dublin */

package javafe.ast;

import org.apache.bcel.Constants;

import javafe.parser.ParseUtil;

/**
 * This class defines the constants used to identify the different
 * kinds of modifiers, and static methods to test for modifiers.
 */

public class Modifiers {

  public static final int NONE                  = 0X000;
  public static final int ACC_PUBLIC            = 0X001;
  public static final int ACC_PRIVATE           = 0X002;
  public static final int ACC_PROTECTED         = 0X004;
  public static final int ACC_STATIC            = 0X008;
  public static final int ACC_FINAL             = 0X010;
  public static final int ACC_SYNCHRONIZED      = 0X020;
  public static final int ACC_VOLATILE          = 0X040;
  public static final int ACC_TRANSIENT         = 0X080;
  public static final int ACC_NATIVE            = 0X100;
  public static final int ACC_ABSTRACT          = 0X400;
  public static final int ACC_STRICT            = 0X800;
  public static final int ACC_ENUM				= Constants.ACC_ENUM;

  public static final int SIZE_MODIFIER_BITSET  = 16;

  public static final int ACCESS_MODIFIERS
    = ACC_PUBLIC | ACC_PROTECTED | ACC_PRIVATE;

  // ----------------------------------------------------------------------
  
  //@ ensures \result != null;
  public static /*@non_null*/String toString(int modifiers) {
    String s = "";

    for (int i = 0; i < ParseUtil.modifierKeywords.length; i++) {
      if ((modifiers & (1<<i)) != 0) {
	s += name(i) + " ";
      }
    }
    
    return s;
  }

  /** Takes a modifier index (that is, the index of the bit that
    * the "modifiers" field represents) and returns the corresponding
    * Java keyword as a String.
    */
    
  //@ requires 0 <= modifierIndex;
  //@ requires modifierIndex < ParseUtil.modifierKeywords.length;
  //@ ensures \result != null;
  public static String name(int modifierIndex) {
    int tag = ParseUtil.modifierKeywords[modifierIndex];
    // Note:  The following "TagConstants" is not the one in this package
    // (javafe.ast), but the one in javafe.parser.
    return javafe.parser.ParserTagConstants.toString(tag);
  }
  
  public static boolean isStrictFP(int modifiers) {
    return (modifiers & ACC_STRICT) != 0;
  }
  
  public static boolean isAbstract(int modifiers) {
    return (modifiers & ACC_ABSTRACT) != 0;
  }
  
  public static boolean isNative(int modifiers) {
    return (modifiers & ACC_NATIVE) != 0;
  }
  
  public static boolean isFinal(int modifiers) {
    return (modifiers & ACC_FINAL) != 0;
  }
  
  public static boolean isPrivate(int modifiers) {
    return (modifiers & ACC_PRIVATE) != 0;
  }
  
  public static boolean isProtected(int modifiers) {
    return (modifiers & ACC_PROTECTED) != 0;
  }
  
  public static boolean isPublic(int modifiers) {
    return (modifiers & ACC_PUBLIC) != 0;
  }
  
  public static boolean isPackage(int modifiers) {
    return (modifiers & (ACC_PRIVATE | ACC_PROTECTED | ACC_PUBLIC)) == 0;
  }
  
  public static boolean isVolatile(int modifiers) {
    return (modifiers & ACC_VOLATILE) != 0;
  }
  
  public static boolean isStatic(int modifiers) {
    return (modifiers & ACC_STATIC) != 0;
  }
  
  public static boolean isSynchronized(int modifiers) {
    return (modifiers & ACC_SYNCHRONIZED) != 0;
  }
  
}
