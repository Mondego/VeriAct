/*
  Copyright (c) 1994-1996 The Regents of the Univ. of California.
  All rights reserved. 

  Permission is hereby granted, without written agreement and without license
  or royalty fees, to use, copy, modify, and distribute this software and its
  documentation for any purpose, provided that the above copyright notice and
  the following two paragraphs appear in all copies of this software.

  IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
  DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF
  CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
  FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN
  "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO PROVIDE
  MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

package mocha.wrappers.jbdd; 

/** java wrapper to a BDD package. This defines a jbddManager and a jbdd
    class. The following functions are exported in the jbddManager
    class. See the BDD package for documentation of each of these
    functions. Since java does not support enumerated types,
    bdd_reorder_type_t and bdd_reorder_verbosity_t are handled as strings. For
    instance, if you want the default verbosity, pass the String
    "BDD_REORDER_VERBOSITY_DEFAULT" as the argument. We have not exported any
    functions that require a bdd node or return a bdd_node because we strongly
    believe that the user of a BDD package should never worry about this. We
    also have taken out some fringe functions such approximate functions and
    file I/O functions for the sake of simplicity
    jbdd jbdd_create_variable()
    jbdd jbdd_create_variable_after(int)
    jbdd jbdd_get_variable(int)
    jbdd jbdd_var_with_index(int)
    jbdd jbdd_compute_cube(int[])
    jbdd jbdd_one()
    jbdd jbdd_zero()
    jbdd jbdd_multiway_and(jbdd[])
    jbdd jbdd_multiway_or(jbdd[])
    jbdd jbdd_multiway_xor(jbdd[])
    int jbdd_read_reorderings()
    void jbdd_realign_enable()
    void jbdd_realign_disable()
    int jbdd_realignment_enabled()
    int jbdd_debug_check()
    void jbdd_dynamic_reordering(String string1, String string2)
    int jbdd_reordering_status (String string1)
    void jbdd_reorder()
    int jbdd_shuffle_heap(int[])
    void jbdd_dynamic_reordering_disable()
    int jbdd_read_reordered_field()
    int jbdd_enable_reordering_reporting()
    int jbdd_disable_reordering_reporting()
    String jbdd_reordering_reporting()
    void jbdd_set_gc_mode(boolean)
    int jbdd_num_vars()
    int jbdd_read_node_count()
    long jbdd_top_var_level(jbdd)
    int jbdd_get_id_from_level(long)
    int jbdd_get_level_from_id(int)
    int jbdd_check_zero_ref()
    void jbdd_set_reordered_field(int)
*/    

public class jbddManager {
  // the lone member of the class
  protected long jbddManager_mem;
  // internal helper functions
  protected native void jbddManagerInit(int i);
  protected native void jbddManagerFinalize();
  // exported functions
  public native /*@ non_null @*/ jbdd jbdd_create_variable();
  public native /*@ non_null @*/ jbdd jbdd_create_variable_after(int i);
  public native /*@ non_null @*/ jbdd jbdd_get_variable(int i);
  public native /*@ non_null @*/ jbdd jbdd_var_with_index(int i);
  public native /*@ non_null @*/ jbdd jbdd_compute_cube(/*@ non_null @*/ int[] ia); // not done yet
  public native /*@ non_null @*/ jbdd jbdd_one();
  public native /*@ non_null @*/ jbdd jbdd_zero();
  public native /*@ non_null @*/ jbdd jbdd_multiway_and (/*@ non_null @*/ jbdd[] bdd1);
  public native /*@ non_null @*/ jbdd jbdd_multiway_or (/*@ non_null @*/ jbdd[] bdd1);
  public native /*@ non_null @*/ jbdd jbdd_multiway_xor (/*@ non_null @*/ jbdd[] bdd1);
  public native int jbdd_read_reorderings ();
  public native void jbdd_realign_enable();
  public native void jbdd_realign_disable();
  public native int jbdd_realignment_enabled();
  public native int jbdd_debug_check();
  public native void jbdd_dynamic_reordering(/*@ non_null */ String string1, /*@ non_null */ String string2);
  public native int jbdd_reordering_status (/*@ non_null */ String string1);
  public native void jbdd_reorder();
  public native int jbdd_shuffle_heap (/*@ non_null @*/ int [] ia);
  public native void jbdd_dynamic_reordering_disable();
  public native int jbdd_read_reordered_field();
  public native int jbdd_enable_reordering_reporting();
  public native int jbdd_disable_reordering_reporting();
  public native /*@ non_null @*/ String jbdd_reordering_reporting();
  public native void jbdd_set_gc_mode (boolean truth);
  public native int jbdd_num_vars();
  public native int jbdd_read_node_count();
  public native long jbdd_top_var_level(/*@ non_null @*/ jbdd bd);
  public native int jbdd_get_id_from_level(long l);
  public native int jbdd_get_level_from_id (int i);
  public native int jbdd_check_zero_ref();
  public native void jbdd_set_reordered_field(int i);
  // the constructor
  public jbddManager(int i)
    {
      jbddManagerInit(i);
    }
  // the finalizer
  protected void finalize() 
    {
      jbddManagerFinalize();
    }

  public static void main(/*@ non_null @*/ String args[]) {
    jbddManager manager = new jbddManager(0);
    jbdd bdd1 = manager.jbdd_one();
    jbdd bdd0 = manager.jbdd_zero();
    jbdd bddand = jbdd.jbdd_and(bdd1, bdd0, true, true);
    jbdd bddor = jbdd.jbdd_or(bdd1, bdd0, true, true);
    System.out.println("BDDAND " + bddand.jbdd_is_tautology(true));
    System.out.println("BDDOR " + bddor.jbdd_is_tautology(true));

    jbdd bddx  = manager.jbdd_create_variable();
    jbdd bddy  = manager.jbdd_create_variable();
    System.out.println("BDDX " + bddx.jbdd_is_tautology(true));
    System.out.println("BDDY " + bddy.jbdd_is_tautology(true));

    jbdd bddor1 = jbdd.jbdd_or(bddx, bddx, true, false);
    System.out.println("BDDOR1 " + bddor1.jbdd_is_tautology(true));
    System.out.println("Im done");
    
  }
   
  static 
    {
      //      try {
    System.loadLibrary("glu");
    System.loadLibrary("cu");
    System.loadLibrary("jbdd");
      //      }
      //      catch (Throwable t){
      //	System.out.println( "Error in loadLibrary: " + t);
      //      }
    }

}
