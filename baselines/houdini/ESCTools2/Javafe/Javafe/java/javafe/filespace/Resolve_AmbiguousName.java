/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;

/**
 * The exceptional result type for lookup:
 */

public class Resolve_AmbiguousName extends Exception {
  private static final long serialVersionUID = 2257695944062705896L;
  
  Tree ambiguousPackage;	// The package that is also a reference type
  
  public Resolve_AmbiguousName(String message, Tree P) {
    super(message);
    ambiguousPackage = P;
  }
}
