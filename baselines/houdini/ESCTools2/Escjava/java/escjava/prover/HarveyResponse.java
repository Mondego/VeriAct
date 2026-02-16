package escjava.prover;

public class HarveyResponse extends ProverResponse {

    // placeholder for factory for building ProverResponses
    static public ProverResponse factory(int return_code) {
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
