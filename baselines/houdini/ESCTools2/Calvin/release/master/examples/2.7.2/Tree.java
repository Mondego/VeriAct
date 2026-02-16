/* Copyright Hewlett-Packard, 2002 */

public class Tree { 
  public /*@ monitored */ Tree left, right; 
  public /*@ monitored non_null */ Thing contents; 

  //@ axiom (\forall Tree t; t.left != null ==> t < t.left); 
  //@ axiom (\forall Tree t; t.right != null ==> t < t.right); 


  Tree(/*@ non_null */ Thing c) {
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
    { Tree x = this.right; 
      if (x == null) return; 
      synchronized (x) { 
        Tree v = x.left; 
        if (v == null) return; 
        synchronized (v) { 
          x.left = v.right; 
          v.right = x; 
          this.right = v; 
        }                       // line (a) 
      } 
    } 
    // Undo the rotation: 
    { Tree v = this.right; 
      synchronized (v) {             // line (b) 
        Tree x = v.right; 
        if (x != null) {            // line (c) 
          synchronized (x) {        // line (d) 
            v.right = x.left; 
            x.left = v; 
            this.right = x; 
          } 
        }                           // line (e) 
      } 
    } 
  }

}
