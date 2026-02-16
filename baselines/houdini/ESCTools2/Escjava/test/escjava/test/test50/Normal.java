/**
 ** This file tests parsing of normal escjava annotation comments
 **/

class Normal {

    void standard() {

	//@ assert true

	/*@ assert true*/
	/*@assert true*/

	/*@assume "/*"!="**"*/
    }

    void wizard() {
	//@ assert true
	//@() assert true
	//@(++!!++) assert true
    }
}

