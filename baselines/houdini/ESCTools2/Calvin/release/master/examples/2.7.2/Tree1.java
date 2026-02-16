/* Copyright Hewlett-Packard, 2002 */

public class Tree1 { 
  public /*@ monitored */ Tree1 left, right; 
  public /*@ monitored non_null */ Thing contents; 

  //@ axiom (\forall Tree1 t; t.left != null ==> t < t.left); 
  //@ axiom (\forall Tree1 t; t.right != null ==> t < t.right); 


  Tree1(/*@ non_null */ Thing c) {
    contents = c;
  }

  //@ requires \max(\lockset) <= this;
  public synchronized void visit() { 
     contents.mungle(); 
     if (left != null) left.visit(); 
     if (right != null) right.visit(); 
  } 

  //@ requires \max(\lockset) <= this; 
  public synchronized void wiggleWoggle() { 
    // Perform a rotation on this.right (but give up and just 
    // return if this.right or this.right.left is null): 
    // 
    //   this              this 
    //   /  \              /  \ 
    // ...   x           ...   v 
    //      / \    -->        / \ 
    //     v   y             u   x 
    //    / \                   / \ 
    //   u   w                 w   y 
    // 
    { Tree1 x = this.right; 
      if (x == null) return; 
      synchronized (x) { 
        Tree1 v = x.left; 
        if (v == null) return; 
        synchronized (v) { 
          x.left = v.right; 
          v.right = x; 
          this.right = v; 
        }                       // line (a) 
      } 
    } 
    // Undo the rotation: 
    { Tree1 v = this.right; 
      synchronized (v) {            // line (b) 
        Tree1 x = v.right; 
                                    // line (c) omitted
          synchronized (x) {        // line (d) 
            v.right = x.left; 
            x.left = v; 
            this.right = x; 
          }
                                    // line (e) omitted
      } 
    } 
  }

}
