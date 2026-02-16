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

import java.util.BitSet;

public class jbdd {
  // the class member
  protected long jbdd;
  // the static functions. These typically take two or more symmetric
  // arguments so it is better to define them static. Use them as follows:
  // jbdd bdd3 = jbdd.jbdd_and(bdd1, bdd2) 

    //@ requires f != null && g != null;
    //@ ensures  \result != null;
  public static	native /*@ non_null */ jbdd jbdd_and(/*@ non_null */ jbdd f, /*@ non_null */ jbdd g, boolean f_phase,
                                       boolean g_phase);

    //@ requires f != null && g != null;
    //@ requires smoothing_vars != null;
    //@ ensures  \result != null;
  public static	native /*@ non_null */ jbdd jbdd_and_smooth(/*@ non_null */ jbdd f, /*@ non_null */ jbdd g, /*@ non_null */ jbdd[]
                                              smoothing_vars);

    //@ requires f != null && g != null;
    //@ requires smoothing_vars != null;
    //@ ensures  \result != null;
  public static native /*@ non_null */ jbdd jbdd_clipping_and_smooth(/*@ non_null */ jbdd f, /*@ non_null */ jbdd g,
                                                       /*@ non_null */ jbdd[]
                                                       smoothing_vars, int
                                                       f_phase, int g_phase);

    //@ requires f != null && g != null;
    //@ requires smoothing_vars != null;
    //@ ensures  \result != null;
  public static	native /*@ non_null */ jbdd jbdd_xor_smooth(/*@ non_null */ jbdd f, /*@ non_null */ jbdd g, /*@ non_null */ jbdd[]
                                              smoothing_vars);

    //@ requires f_min != null && f_max != null;
    //@ ensures  \result != null;
  public static native /*@ non_null */ jbdd jbdd_between(/*@ non_null */ jbdd f_min, /*@ non_null */ jbdd f_max);

    //@ requires f != null && v != null && g != null;
    //@ ensures  \result != null;
  public static native /*@ non_null */ jbdd jbdd_compose(/*@ non_null */ jbdd f, /*@ non_null */ jbdd v, /*@ non_null */ jbdd g);

    //@ requires i != null && t != null && e != null;
    //@ ensures  \result != null;
  public static native /*@ non_null */ jbdd jbdd_ite(/*@ non_null */ jbdd i, /*@ non_null */ jbdd t, /*@ non_null */ jbdd e, boolean
                                       i_phase, boolean t_phase, boolean
                                       e_phase);

    //@ requires f != null && g != null;
    //@ ensures  \result != null;
  public static native /*@ non_null */ jbdd jbdd_or(/*@ non_null */ jbdd f, /*@ non_null */ jbdd g, boolean f_phase,
                                      boolean g_phase);

    //@ requires f != null && g != null;
    //@ ensures  \result != null;
  public static native /*@ non_null */ jbdd jbdd_xnor (/*@ non_null */ jbdd f, /*@ non_null */ jbdd g);

    //@ requires f != null && g != null;
    //@ ensures  \result != null;
  public static native /*@ non_null */ jbdd jbdd_xor(/*@ non_null */ jbdd f, /*@ non_null */ jbdd g);

    //@ requires f != null && g != null;
  public static native int jbdd_apa_compare_ratios(int a, /*@ non_null */ jbdd f, /*@ non_null */ jbdd g,
                                                   int j, int k);

    //@ requires f != null && g != null;
  public static native int jbdd_estimate_cofactor(/*@ non_null */ jbdd f, /*@ non_null */ jbdd g, int n);

  //--------------------------------------------------------------------------
  // and these are the nonstatic functions

  public native /*@ non_null @*/ jbdd jbdd_dup();
  public native	void jbdd_free();
  public native	/*@ non_null @*/ jbdd jbdd_cofactor(/*@ non_null @*/ jbdd g);
  public native	/*@ non_null @*/ jbdd jbdd_var_cofactor(/*@ non_null @*/ jbdd g);
  public native	/*@ non_null @*/ jbdd jbdd_consensus(/*@ non_null @*/ jbdd [] quantifying_vars);
  public native	/*@ non_null @*/ jbdd jbdd_cproject(/*@ non_null @*/ jbdd [] var_array);
  public native	/*@ non_null @*/ jbdd jbdd_else();
  public native	/*@ non_null @*/ jbdd jbdd_minimize(/*@ non_null @*/ jbdd c);
  public native	/*@ non_null @*/ jbdd jbdd_compact(/*@ non_null @*/ jbdd c);
  public native	/*@ non_null @*/ jbdd jbdd_squeeze(/*@ non_null @*/ jbdd c); // not doc'd
  public native	/*@ non_null @*/ jbdd jbdd_not();
  public native	/*@ non_null @*/ jbdd jbdd_smooth (/*@ non_null @*/ jbdd[] smoothing_vars);
  public native	/*@ non_null @*/ jbdd jbdd_substitute(/*@ non_null @*/ jbdd[] old_array,
						   /*@ non_null @*/ jbdd[] new_array);
  public native /*@ non_null @*/ jbdd jbdd_then();
  public native	/*@ non_null @*/ jbdd jbdd_top_var ();
  public native	/*@ non_null @*/ jbdd jbdd_shortest_path(/*@ non_null */ int [] weight, /*@ non_null */ int [] support, /*@ non_null */ int
                                          [] length); // check -- this is
  // trouble
  public native	boolean jbddest_unate(int var_id, int phase); // shouldn't
  // this be boolean?
  public native	/*@ non_null @*/ jbdd[] jbdd_find_essential();
  public native	/*@ non_null @*/ double[] jbdd_cof_minterm();
  public native	boolean jbdd_var_is_dependent(/*@ non_null @*/ jbdd f);
  public native	boolean jbdd_equal(/*@ non_null @*/ jbdd g);
  public native	/*@ non_null @*/ jbdd jbdd_intersects(/*@ non_null @*/ jbdd g); 
  public native	boolean jbdd_is_tautology (boolean phase);
  public native	boolean jbdd_leq(/*@ non_null @*/ jbdd g, boolean f_phase, boolean g_phase);
  public native	double jbdd_count_onset(/*@ non_null @*/ jbdd[] var_array);
  public native	double jbdd_correlation(/*@ non_null @*/ jbdd g);
  public native	int jbdd_get_free ();
  public native	long jbdd_get_manager();
  public native	/*@ non_null */ BitSet jbdd_get_support ();
  public native	void jbdd_print();
  public native	int jbdd_print_minterm();
  public native	int jbdd_size();
  public native	boolean jbdd_is_cube();
  public native	int jbdd_top_var_id();
  public jbdd(){
  }
  protected native void jbdd_finalize();

  protected void finalize () {
	jbdd_finalize();
  }
  static {
    System.loadLibrary("glu");
    System.loadLibrary("cu");
    System.loadLibrary("jbdd");
  }


}
