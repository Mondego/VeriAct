public class Final {

    public void m() {
	//@ assert \typeof(this) <: \type(Final);  // OK
    }

    public void mm() {
	//@ assert \typeof(this) == \type(Final);  // ERROR
    }

    public void n() {
	//@ assert !(\typeof(this) <: \type(I)); // ERROR
    }

    public void nn() {
	//@ assert \typeof(this)==\type(Final) ==> !(\typeof(this) <: \type(I)); // OK
    }

}


final class F2 {
    public void p() {
	//@ assert \typeof(this) == \type(F2);  // OK
    }

    public void q() {
	//@ assert !(\typeof(this) <: \type(I)); // OK
    }
 }

interface I {}
