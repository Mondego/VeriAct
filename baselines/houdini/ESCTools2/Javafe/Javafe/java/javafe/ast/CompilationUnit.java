// -*- mode: java -*-
/* Copyright 2000, 2001, Compaq Computer Corporation */

/* IF THIS IS A JAVA FILE, DO NOT EDIT IT!  

   Most Java files in this directory which are part of the Javafe AST
   are automatically generated using the astgen comment (see
   ESCTools/Javafe/astgen) from the input file 'hierarchy.h'.  If you
   wish to modify AST classes or introduce new ones, modify
   'hierarchy.j.'
 */

package javafe.ast;

import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.ErrorSet;
import javafe.tc.TagConstants;

// Convention: unless otherwise noted, integer fields named "loc" refer
// to the location of the first character of the syntactic unit


/* ---------------------------------------------------------------------- */

/** The <code>make</code> method of this class has the side effect of
pointing the <code>parent</code> pointers of the <code>TypeDecl</code>s
inside a <code>CompilationUnit</code> to point to that unit. */

public class CompilationUnit extends ASTNode
{
  public /*@nullable*/ Name pkgName;

  public /*@nullable*/ LexicalPragmaVec lexicalPragmas;

  public /*@non_null*/ ImportDeclVec imports;

  public /*@non_null*/ TypeDeclVec elems;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  public /*@non_null*/ TypeDeclElemVec otherPragmas;


  //@ public represents startLoc <- loc;

  public boolean duplicate = false;

  private void postCheck() {
    for(int i = 0; i < elems.size(); i++) {
      for(int j = i+1; j < elems.size(); j++)
	Assert.notFalse(elems.elementAt(i) != elems.elementAt(j));  //@ nowarn Pre;
    }
  }

  /**
   * @return true iff this CompilationUnit was created from a .class
   * file.
   */
  public boolean isBinary() {
    return Location.toFileName(loc).endsWith(".class");
  }

  public /*@ pure @*/ int getStartLoc() { return loc; }
  public /*@ pure @*/ int getEndLoc() { 
    if (elems == null || elems.size() < 1)
      return super.getEndLoc();

    return elems.elementAt(elems.size()-1).getEndLoc();
  }

  public /*@non_null*/ javafe.genericfile.GenericFile sourceFile() {
    return Location.toFile(loc);
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.pkgName == pkgName;
  //@ ensures this.lexicalPragmas == lexicalPragmas;
  //@ ensures this.imports == imports;
  //@ ensures this.elems == elems;
  //@ ensures this.loc == loc;
  //@ ensures this.otherPragmas == otherPragmas;
  protected CompilationUnit(/*@nullable*/Name pkgName, /*@nullable*/LexicalPragmaVec lexicalPragmas, /*@non_null*/ ImportDeclVec imports, /*@non_null*/ TypeDeclVec elems, int loc, /*@non_null*/ TypeDeclElemVec otherPragmas) {
     super();
     this.pkgName = pkgName;
     this.lexicalPragmas = lexicalPragmas;
     this.imports = imports;
     this.elems = elems;
     this.loc = loc;
     this.otherPragmas = otherPragmas;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     LexicalPragmaVec tmp_lexicalPragmas = this.lexicalPragmas;
     if (tmp_lexicalPragmas != null) sz += tmp_lexicalPragmas.size();
     ImportDeclVec tmp_imports = this.imports;
     if (tmp_imports != null) sz += tmp_imports.size();
     TypeDeclVec tmp_elems = this.elems;
     if (tmp_elems != null) sz += tmp_elems.size();
     TypeDeclElemVec tmp_otherPragmas = this.otherPragmas;
     if (tmp_otherPragmas != null) sz += tmp_otherPragmas.size();
     return sz + 1;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.pkgName;
     else index--;

     LexicalPragmaVec tmp_lexicalPragmas = this.lexicalPragmas;
     sz = (tmp_lexicalPragmas == null ? 0 : tmp_lexicalPragmas.size());
     if (0 <= index && index < sz)
        return tmp_lexicalPragmas.elementAt(index);
     else index -= sz;

     ImportDeclVec tmp_imports = this.imports;
     sz = (tmp_imports == null ? 0 : tmp_imports.size());
     if (0 <= index && index < sz)
        return tmp_imports.elementAt(index);
     else index -= sz;

     TypeDeclVec tmp_elems = this.elems;
     sz = (tmp_elems == null ? 0 : tmp_elems.size());
     if (0 <= index && index < sz)
        return tmp_elems.elementAt(index);
     else index -= sz;

     TypeDeclElemVec tmp_otherPragmas = this.otherPragmas;
     sz = (tmp_otherPragmas == null ? 0 : tmp_otherPragmas.size());
     if (0 <= index && index < sz)
        return tmp_otherPragmas.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[CompilationUnit"
        + " pkgName = " + this.pkgName
        + " lexicalPragmas = " + this.lexicalPragmas
        + " imports = " + this.imports
        + " elems = " + this.elems
        + " loc = " + this.loc
        + " otherPragmas = " + this.otherPragmas
        + "]";
  }

  public final int getTag() {
     return TagConstants.COMPILATIONUNIT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitCompilationUnit(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitCompilationUnit(this, o); }

  public void check() {
     super.check();
     if (this.pkgName != null)
        this.pkgName.check();
     LexicalPragmaVec tmp_lexicalPragmas = this.lexicalPragmas;
     if (tmp_lexicalPragmas != null)
        for(int i = 0; i < tmp_lexicalPragmas.size(); i++)
           tmp_lexicalPragmas.elementAt(i).check();
     ImportDeclVec tmp_imports = this.imports;
     for(int i = 0; i < tmp_imports.size(); i++)
        tmp_imports.elementAt(i).check();
     if (this.elems == null) throw new RuntimeException();
     if (this.otherPragmas == null) throw new RuntimeException();
     postCheck();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ CompilationUnit make(/*@nullable*/Name pkgName, /*@nullable*/LexicalPragmaVec lexicalPragmas, /*@non_null*/ ImportDeclVec imports, /*@non_null*/ TypeDeclVec elems, int loc, /*@non_null*/ TypeDeclElemVec otherPragmas) {
     CompilationUnit result = new CompilationUnit(pkgName, lexicalPragmas, imports, elems, loc, otherPragmas);
     return result;
  }
}
