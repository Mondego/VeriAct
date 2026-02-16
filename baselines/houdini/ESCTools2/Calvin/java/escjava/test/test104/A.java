/* Copyright Hewlett-Packard, 2002 */

class B {
    int x, y;

    /*@ global_invariant x < y */
    /*@ oti y < \old(x) */
    /* ti x == x */    
    /* oti y < \tid */
    /* ti x == \tid */

    int Add() {
	int z;
	z = x + y;
	return z;
    }

    int IncX() {
	x++;
	return x;
    }

    int IncX1() {
	int[] a = new int[2];

	a[0] = x;

	a[0]++;

	return a[0];
    }

    int Foo1(int i) {
	x = i;
    }

    int Foo2() {
	int [] a = new int[10];
	y = 0;
	a[y] = x;
    }

    int Foo3() {
	int [] a = new int[10];
	x = a[0];
    }

} 
