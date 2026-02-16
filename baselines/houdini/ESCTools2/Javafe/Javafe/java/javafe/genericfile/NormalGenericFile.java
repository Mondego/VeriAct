/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.genericfile;

import javafe.Tool;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * A NormalGenericFile represents a normal file ({@link java.io.File})
 * as a GenericFile.
 */

public class NormalGenericFile extends File implements GenericFile {
  private static final long serialVersionUID = 4996822143630105487L;
  
  /**
   * Create a NormalGenericFile to represent an existing {@link
   * File}.
   */
  //@ requires underlyingFile != null;
  public NormalGenericFile(File underlyingFile) {
    super(underlyingFile.getPath());
  }
  
  /**
   * Create a NormalGenericFile from a filename.
   */
  //@ requires name != null;
  public NormalGenericFile(String name) {
    super(name);
  }
  
  /**
   * @return a name that uniquely identifies us to the user.
   *
   * Warning: the result may not be a conventional filename or use
   * the system separators.
   */
  public String getHumanName() {
	  String result = this.toString();
	  if (Tool.options.testMode)
		  result = result.replace('/', '|').replace('\\', '|');
	  else if (File.separatorChar == '\\') {
		  // Normalize path names under Windows (like what is done in Eclipse).
		  result = result.replace('\\', '/');
		  if (result.length() > 2 && result.charAt(1) == ':')
			  result = result.substring(2);
	  }
	  return result;
  }
  
  /**
   * @return a String that canonically represents the identity of
   * our underlying file.
   *
   * This function must be defined such that if two GenericFiles
   * return non-null canonical ID's then the IDs are the same
   * (modulo .equals) => the GenericFiles represent the same
   * underlying file.  Ideally, under normal circumstances, the =>
   * is actually a <=>.
   *
   * This function should only return null in exceptional cases,
   * such as when an I/O error in the underlying storage media
   * prevents construction of a canonical ID.
   *
   * Convention: Canonical IDs start with <X> where X is the
   * fully-qualified name of the class that mediates I/O to the
   * underlying file.  E.g., java.io.File for a normal disk file.
   */
  public String getCanonicalID() {
    try {
      return "<java.io.File>" + this.getCanonicalPath();
    } catch (IOException e) {
      return null;
    }
  }
  
  /**
   * @return our local name, the name that distinguishes us within
   * the directory that contains us.
   *
   * E.g., "/a/b/c" has local name "c", "/e/r/" has local name "r", and
   * "/" has local name "".  (assuming "/" is the separator char)
   */
  public String getLocalName() { return getName(); }
  
  /**
   * Open the file we represent as an {@link InputStream}.
   *
   * @exception IOException May be thrown for many reasons,
   * including no such file and read permission denied.
   */
  public InputStream getInputStream() throws IOException {
    return new FileInputStream(this);
  }
  
  /**
   * @return a GenericFile that describes the file in the same
   * "directory" as us that has the local name <code>n</code>.  No
   * attempt is made to verify whether or not that file exists.  In
   * cases where the notion of "containing directory" makes no sense
   * (e.g., streams or root directories), null is returned.
   */
  public GenericFile getSibling(/*@non_null*/String n) { 
    String dir = super.getParent();
    if (dir==null)
      return null;
    
    return new NormalGenericFile(dir + separator + n);
  }
}
