// This is a simplified version of test37 showing a difference that has
// arisen between Exceptions and Throwables - we may not want to maintain
// this difference.
//#FLAGS: 
public class DefCon {


    static public int gg;


    class AE extends E {

	// This fails to establish the postcondition because the
        // implicit constructor of the super class might throw a
        // exception and it might be SE.

        //@ signals(SE se) gg == 17;
	public AE() throws Throwable {
	   if (gg == 17) throw new SE();
	}





        //@ signals(ST se) gg == 17;
	public AE(int i) throws Throwable {
	   if (gg == 17) throw new ST();
	}


    }

    class E {

	public E() throws Throwable {
	}

    }


    class ST extends Throwable {
        public ST() { }
    }

    class SE extends Exception {
        public SE() { }
    }

}

