public class StudentEnrollment {
    public static final int costPerCredit = 200;  
    public static final int totalCredits = 120;
    public static final int maxSemesterCredits = 20;

     private String firstName;
     private String lastName;
     private int passedCredits;  
     private int enrollmentCredits; 
    
     private int tuitionBalance;
    
     private boolean lateRegistration;
	
    
    StudentEnrollment( String firstName,
               String lastName) 
    {
        this.firstName = firstName;
        this.lastName = lastName;
    }

    
    public void setTuitionBalance(int tuitionBalance)
    {
	int maxTuitionBalance = maxSemesterCredits * costPerCredit + maxSemesterCredits * ((costPerCredit/100)*6);
	if (maxTuitionBalance < tuitionBalance) {
                throw new IllegalArgumentException();
	} else {
		this.tuitionBalance = tuitionBalance;
	}
    }
	
    
    public void setPassedCredits(int passedCredits)
    {
        if (0 <= passedCredits && passedCredits <= totalCredits) {
            this.passedCredits = passedCredits;
        } else {
            throw new IllegalArgumentException();
        }
    }

    
    public void setEnrollmentCredits(int enrollmentCredits)
    {
        if (0 <= enrollmentCredits && enrollmentCredits <= maxSemesterCredits) {
            this.enrollmentCredits = enrollmentCredits;
        } else {
            throw new IllegalArgumentException();
        }
    }

    
    
    
    public void setLateRegistration (boolean lateRegistration) 
    {
        this.lateRegistration = lateRegistration;
    }

    
    
    public  int getTuition() 
    {
        return this.tuitionBalance;
    }

    
    
    public  int getEnrollmentCredits() 
    {
        return this.enrollmentCredits;
    }

    
    
    public  int getPassedCredits() 
    {
        return this.passedCredits;
    }

    
    
    public  boolean getLateRegistrations() 
    {
        return this.lateRegistration;
    }

    
    
    public  String getFirstName()
    {
        return this.firstName;
    }

    
    
    public  String getLastName()
    {
        return this.lastName;
    } 

    
    
    public  boolean registrationPermission() 
    {
        return (this.tuitionBalance <= 0);
    }

    
    
    public  boolean validPayment(int payment)
    {
        return (0 < payment);
    }

    
    public void regularEnrollment()
    {
	if (registrationPermission()) {
        	int cost = getEnrollmentCredits() * costPerCredit;
        	tuitionBalance = getTuition() + cost;
	} else { 
		throw new IllegalArgumentException();
        }
    }

    
    public void lateEnrollment() 
    {
	if (registrationPermission()) {
        	int latePerCredit = (costPerCredit/100)*6;
        	int cost;
        	cost = getEnrollmentCredits() * costPerCredit + getEnrollmentCredits() * latePerCredit;
        	tuitionBalance = getTuition() + cost;
	} else { 
		throw new IllegalArgumentException();
        }
    }

    
    public void enroll() 
    {
        if (getLateRegistrations())
            lateEnrollment(); 
        else 
            regularEnrollment();
    }

    
    private  void payTuitionWithDebitCard(int payment) 
    {
        if (validPayment(payment)) {
            tuitionBalance -= payment;
        } else {
            throw new IllegalArgumentException();
        }
    }

    
    private  void payTuitionWithCreditCard(int payment) 
    {
        int penalty;
        penalty = (payment/100)*2;
        payment -= penalty;
        if (validPayment(payment)) {
            tuitionBalance -= payment;
	} else {
            throw new IllegalArgumentException();
        }
    }

    
    public void pay(int payment, boolean debit) 
    {
        if (debit) {
            payTuitionWithDebitCard(payment);
        } else {
            payTuitionWithCreditCard(payment);
        }
    }

    
    public void enrollmentProcess(int passedCredits, int semesterCredits, int payment, int initialBalance, boolean lateRegistration, boolean debit, boolean option) 
     {
        String familyName = getLastName();
        String firstName = getFirstName();
        setTuitionBalance(initialBalance);
        setPassedCredits(passedCredits);
        setEnrollmentCredits(semesterCredits);
	setLateRegistration(lateRegistration);

        if (option) {
		if (registrationPermission()) {
			enroll();
                }
	} else {
		pay(payment, debit);
	}
     }
}
