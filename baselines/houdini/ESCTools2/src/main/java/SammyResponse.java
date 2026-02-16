package escjava.prover;

public class SammyResponse extends ProverResponse {

    // placeholder for factory for building ProverResponses
    /*@ 
      @ ensures return_code == 1 ==> \result == ProverResponse.OK;
      @ ensures return_code == -1 ==> \result == ProverResponse.FAIL;
      @ ensures return_code == -2 ==> \result == ProverResponse.SYNTAX_ERROR;
      @
      @*/
    static public /*@ pure non_null @*/ ProverResponse factory(int return_code) {
	switch(return_code){
	  
	case 1 :
	    return ProverResponse.OK;
	case -1 :
	    return ProverResponse.FAIL;
	case -2 :
	    return ProverResponse.SYNTAX_ERROR;
	default : //positive attitude
	    return ProverResponse.OK;
	}
    }

}
