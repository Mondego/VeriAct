package tohtml;

import java.io.*;
import java.util.*;

class VectorSorter {
    static private Vector dat;

    static public void Sort(Vector data) {
        dat = data;
        sort(0, dat.size() - 1);
    }

    static private final long compare(Object a, Object b) {
        if (a instanceof Package)
            return ((Package)a).name.compareTo(((Package)b).name);
        else
            return ((JFile)a).name.compareTo(((JFile)b).name);
    }

    static private void sort(int p, int r) {
        if (p < r) {
            int q = partition(p,r);
            if (q == r) 
                q--;
            sort(p,q);
            sort(q+1,r);
        }
    }

    static private int partition(int lo, int hi) {
        Object pivot = dat.elementAt(lo);
        while (true) {
            while (compare(dat.elementAt(hi), pivot) >= 0 && lo < hi) hi--;
            while (compare(dat.elementAt(lo), pivot) < 0 && lo < hi) lo++;
            if (lo < hi) {
                Object T = dat.elementAt(lo);
                dat.setElementAt(dat.elementAt(hi), lo);
                dat.setElementAt(T, hi);
            } else
                return hi;
        }
    }
}
