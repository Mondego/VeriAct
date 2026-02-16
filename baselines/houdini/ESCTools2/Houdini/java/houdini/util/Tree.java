/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.util;
 
import java.util.*;
 
public class Tree {
    
    class TreeNode {
        String prefix;
        TreeNode left;
        TreeNode right;
        TreeNode before;
        TreeNode after;
        boolean isTerminal;
        TreeNode(String prefix, TreeNode left, TreeNode right, boolean isTerminal) {
            this.prefix = prefix;
            this.left = left;
            this.right = right;
            this.isTerminal = isTerminal;
            this.before = null;
            this.after = null;
        }
    }
    
    TreeNode root;
    TreeNode first;
    
    public Tree() {
        root = null;
        first = null;
    }
    
    private void put(String s, TreeNode t) {
        int n = s.compareTo(t.prefix);
        if (n < 0) {
            if (t.left == null) {
                t.left = new TreeNode(s, null, null, true);
                if (t.before != null) t.before.after = t.left;
                t.left.before = t.before;
                t.before = t.left;
                t.left.after = t;
                if (t.left.before == null) {
                    first = t.left;
                }
            } else {
                put(s, t.left);
            }
        } else if (n > 0) {
            if (t.right == null) {
                t.right = new TreeNode(s, null, null, true);
                if (t.after != null) t.after.before = t.right;
                t.right.before = t;
                t.right.after = t.after;                
                t.after = t.right;
            } else {
                put(s, t.right);
            }
        } else { 
            // already there
        }     
        
    }
    
    class TreeEnum implements Enumeration {
        TreeNode curr;
        String prefix;
        TreeEnum(TreeNode t) {
            this.curr = t;
        }
        TreeEnum(TreeNode t, String prefix) {
            this.curr = t;
            this.prefix = prefix;
        }        
        public boolean hasMoreElements() {
            return this.curr != null && 
            (prefix == null || this.curr.prefix.startsWith(prefix));
        }
        public Object nextElement() {
            if (!hasMoreElements()) return null;
            String s = this.curr.prefix;
            this.curr = this.curr.after;
            return s;
        }
    }
    
    public Enumeration elements() {
        return new TreeEnum(first);
    }

    public Enumeration elements(String prefix) {
        return new TreeEnum(findPrefix(prefix, root), prefix);
    }

    String toString(TreeNode t) {
        if (t == null) return "-";
        return "(" + toString(t.left) + " " + t.prefix + " " + toString(t.right) + ")";
    }
    
    public String toString() {
        return toString(root);
    }
    
    public void put(String s) {
        if (root == null) {
            first = root = new TreeNode(s, null, null, true);
            root.before = null;
            root.after = null;
        } else {
            put(s, root);
        }
    }

    
    private TreeNode findPrefix(String s, TreeNode t) {
        if (t == null) return null;
        int n = s.compareTo(t.prefix);
        if (n == 0) return t;
        if (n < 0) {
            TreeNode tt = findPrefix(s, t.left);
            if (tt == null && t.prefix.startsWith(s)) return t;
            return tt;
        } else {
            // n > 0
            return findPrefix(s, t.right); 
        }
    }    
    
    public String matchPrefix(String s) {
        TreeNode t =findPrefix(s, root);
        return t == null ? null : t.prefix;
    }
    
    private TreeNode findNode(String s, TreeNode t) {
        if (t == null) return null;
        int n = s.compareTo(t.prefix);
        if (n == 0) return t;
        if (n < 0) return findNode(s, t.left);
        return findNode(s, t.right);
    }
    
    public boolean contains(String s) {
        return findNode(s, root) != null;   
    }
    
    public static void main(String st[]) {
        Tree p = new Tree();
        for (int i = 0; i < 10; i++) {
            p.put("" + (char)('a' + ((i * 3) % 10)));
        }
        for (int i = 0; i < 100; i++) {
            p.put("" + (char)('a' + ((i * 3) % 13)) + (char)('a' + ((i * 37) % 11)));
        }
        for (char a = 'a'; a < 'a' + 10; a++) {
            System.out.print(a + ": ");
            for (Enumeration e = p.elements(""+a); e.hasMoreElements() ;) {
                String s = (String)e.nextElement();
                System.out.print(s+ " ");
            }
            System.out.println("");
        }
    }
    
}
