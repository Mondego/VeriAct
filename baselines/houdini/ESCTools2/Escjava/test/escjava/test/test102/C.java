// @see Ticket #28 : NullPointerException thrown with @pure modified comes before @modifies spec

public class C {

    /*@pure*/
    //@ modifies \nothing;
	public void foo() {
	}

 	public void bar() {
		foo();
	}
}
