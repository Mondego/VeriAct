public class StudentEnrollment {
    public static final int costPerCredit = 200;  
    public static final int totalCredits = 120;
    public static final int maxSemesterCredits = 20;

    /*@ spec_public @*/ private String firstName;
    /*@ spec_public @*/ private String lastName;
    /*@ spec_public @*/ private int passedCredits;  //number of credits which are passed during previous semesters
    /*@ spec_public @*/ private int enrollmentCredits; //number of credits which will get this semester
    //@ public invariant 0 <= enrollmentCredits && enrollmentCredits <= maxSemesterCredits;
    /*@ spec_public @*/ private int tuitionBalance;
    /*@ public invariant tuitionBalance <= 
                         maxSemesterCredits * costPerCredit 
                         + (maxSemesterCredits * ((costPerCredit/100)*6)); @*/
    /*@ spec_public @*/ private boolean lateRegistration;
	
    /*@ ensures this.firstName.equals(firstName);
      @ ensures this.lastName.equals(lastName);
      @ ensures passedCredits == 0 && enrollmentCredits == 0;
      @ ensures tuitionBalance == 0; @*/
    StudentEnrollment(/*@ non_null @*/ String firstName,
              /*@ non_null @*/ String lastName) 
    {
        this.firstName = firstName;
        this.lastName = lastName;
    }

    /*@     public normal_behavior
      @   	requires tuitionBalance <= maxSemesterCredits * costPerCredit + maxSemesterCredits * ((costPerCredit/100)*6);
      @         assignable this.tuitionBalance;
      @   	ensures this.tuitionBalance == tuitionBalance; 
      @ also
      @     public exceptional_behavior
      @		requires maxSemesterCredits * costPerCredit + maxSemesterCredits * ((costPerCredit/100)*6) < tuitionBalance;
      @ 	assignable \nothing;
      @		signals_only IllegalArgumentException;
      @   	signals(IllegalArgumentException) this.tuitionBalance == \old (this.tuitionBalance); @*/
    public void setTuitionBalance(int tuitionBalance)
    {
	int maxTuitionBalance = maxSemesterCredits * costPerCredit + maxSemesterCredits * ((costPerCredit/100)*6);
	if (maxTuitionBalance < tuitionBalance) {
                throw new IllegalArgumentException();
	} else {
		this.tuitionBalance = tuitionBalance;
	}
    }
	
    /*@      public normal_behavior
      @   	requires 0 <= passedCredits && passedCredits <= totalCredits;
      @   	assignable this.passedCredits;
      @   	ensures this.passedCredits == passedCredits;
      @ also
      @      public exceptional_behavior
      @   	requires passedCredits < 0 || totalCredits < passedCredits;
      @   	assignable \nothing;
      @   	signals_only IllegalArgumentException; @*/
    public void setPassedCredits(int passedCredits)
    {
        if (0 <= passedCredits && passedCredits <= totalCredits) {
            this.passedCredits = passedCredits;
        } else {
            throw new IllegalArgumentException();
        }
    }

    /*@      public normal_behavior
      @   	requires 0 <= enrollmentCredits && enrollmentCredits <= maxSemesterCredits;
      @   	assignable this.enrollmentCredits;
      @   	ensures this.enrollmentCredits == enrollmentCredits;
      @ also
      @      public exceptional_behavior
      @   	requires enrollmentCredits < 0 || maxSemesterCredits < enrollmentCredits;
      @   	assignable \nothing;
      @   	signals_only IllegalArgumentException; @*/
    public void setEnrollmentCredits(int enrollmentCredits)
    {
        if (0 <= enrollmentCredits && enrollmentCredits <= maxSemesterCredits) {
            this.enrollmentCredits = enrollmentCredits;
        } else {
            throw new IllegalArgumentException();
        }
    }

    //@ public normal_behavior
    //@   assignable this.lateRegistration;
    //@   ensures this.lateRegistration == lateRegistration;
    public void setLateRegistration (boolean lateRegistration) 
    {
        this.lateRegistration = lateRegistration;
    }

    //@ public normal_behavior
    //@   ensures \result == this.tuitionBalance;
    public /*@ pure @*/ int getTuition() 
    {
        return this.tuitionBalance;
    }

    //@ public normal_behavior
    //@   ensures \result == this.enrollmentCredits;
    public /*@ pure @*/ int getEnrollmentCredits() 
    {
        return this.enrollmentCredits;
    }

    //@ public normal_behavior
    //@   ensures \result == this.passedCredits;
    public /*@ pure @*/ int getPassedCredits() 
    {
        return this.passedCredits;
    }

    //@ public normal_behavior
    //@   ensures \result == this.lateRegistration;
    public /*@ pure @*/ boolean getLateRegistrations() 
    {
        return this.lateRegistration;
    }

    //@ public normal_behavior
    //@   ensures \result.equals(this.firstName);
    public /*@ pure @*/ String getFirstName()
    {
        return this.firstName;
    }

    //@ public normal_behavior
    //@   ensures \result.equals(this.lastName);
    public /*@ pure @*/ String getLastName()
    {
        return this.lastName;
    } 

    //@ public normal_behavior
    //@   ensures \result <==> (tuitionBalance <= 0);
    public /*@ pure @*/ boolean registrationPermission() 
    {
        return (this.tuitionBalance <= 0);
    }

    //@ public normal_behavior
    //@   ensures \result <==> (0 < payment);
    public /*@ pure @*/ boolean validPayment(int payment)
    {
        return (0 < payment);
    }

    /*@      public normal_behavior
      @   	requires registrationPermission();
      @   	requires this.tuitionBalance + maxSemesterCredits * costPerCredit <= Integer.MAX_VALUE;
      @   	assignable this.tuitionBalance;
      @   	ensures this.tuitionBalance == \old(tuitionBalance + enrollmentCredits * costPerCredit);
      @ also
      @      public exceptional_behavior
      @   	requires !registrationPermission();
      @   	assignable \nothing;
      @   	signals_only IllegalArgumentException; @*/
    public void regularEnrollment()
    {
	if (registrationPermission()) {
        	int cost = getEnrollmentCredits() * costPerCredit;
        	tuitionBalance = getTuition() + cost;
	} else { 
		throw new IllegalArgumentException();
        }
    }

    /*@      public normal_behavior
      @   	old int latePerCredit = (costPerCredit/100)*6;
      @   	requires registrationPermission();
      @   	requires this.tuitionBalance + maxSemesterCredits * costPerCredit + maxSemesterCredits * latePerCredit <= Integer.MAX_VALUE;
      @   	assignable this.tuitionBalance;
      @   	ensures this.tuitionBalance == \old (tuitionBalance) + enrollmentCredits * costPerCredit + enrollmentCredits * latePerCredit;
      @ also
      @      public exceptional_behavior
      @   	requires !registrationPermission();
      @   	assignable \nothing;
      @   	signals_only IllegalArgumentException; @*/
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

    /*@      public normal_behavior
      @ 	requires lateRegistration;
      @ 	old int latePerCredit = (costPerCredit/100)*6;
      @ 	requires registrationPermission();
      @ 	requires this.tuitionBalance + maxSemesterCredits * costPerCredit + maxSemesterCredits * latePerCredit <= Integer.MAX_VALUE;
      @ 	assignable this.tuitionBalance;
      @ 	ensures this.tuitionBalance == \old (tuitionBalance + enrollmentCredits * costPerCredit) + enrollmentCredits * latePerCredit;
      @ also
      @      public normal_behavior
      @   	requires !lateRegistration;
      @   	requires registrationPermission();
      @   	requires this.tuitionBalance + maxSemesterCredits * costPerCredit <= Integer.MAX_VALUE;
      @   	assignable this.tuitionBalance;
      @   	ensures this.tuitionBalance == \old (tuitionBalance + enrollmentCredits * costPerCredit); @*/
    public void enroll() 
    {
        if (getLateRegistrations())
            lateEnrollment(); 
        else 
            regularEnrollment();
    }

    /*@      public normal_behavior
      @   	requires validPayment(payment);
      @  	requires Integer.MIN_VALUE <= tuitionBalance - payment;
      @   	assignable tuitionBalance;
      @   	ensures tuitionBalance == \old(tuitionBalance - payment);
      @ also
      @      public exceptional_behavior
      @   	requires !validPayment(payment);
      @   	assignable \nothing;
      @   	signals_only IllegalArgumentException; @*/
    private /*@ spec_public @*/ void payTuitionWithDebitCard(int payment) 
    {
        if (validPayment(payment)) {
            tuitionBalance -= payment;
        } else {
            throw new IllegalArgumentException();
        }
    }

    /*@      public normal_behavior
      @   	old int penalty = (payment/100)*2;
      @   	old int _payment = payment - penalty;
      @   	requires validPayment(_payment);
      @   	requires Integer.MIN_VALUE <= tuitionBalance - _payment;
      @   	assignable tuitionBalance;
      @   	ensures tuitionBalance == \old(tuitionBalance - _payment);
      @ also
      @      public exceptional_behavior
      @   	old int penalty = (payment/100)*2;
      @   	old int _payment = payment - penalty;
      @   	requires !validPayment(_payment);
      @   	assignable \nothing;
      @   	signals_only IllegalArgumentException; @*/
    private /*@ spec_public @*/ void payTuitionWithCreditCard(int payment) 
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

    /*@      public normal_behavior
      @   	requires debit;
      @   	requires validPayment(payment);
      @   	requires Integer.MIN_VALUE <= tuitionBalance - payment;
      @   	assignable this.tuitionBalance;
      @   	ensures tuitionBalance == \old (tuitionBalance - payment);
      @ also
      @      public normal_behavior
      @   	requires !debit;
      @   	old int penalty = (payment/100)*2;
      @   	old int _payment = payment - penalty;
      @   	requires validPayment(_payment);
      @   	requires Integer.MIN_VALUE <= tuitionBalance - _payment;
      @   	assignable this.tuitionBalance;
      @   	ensures tuitionBalance == \old (tuitionBalance - _payment); @*/
    public void pay(int payment, boolean debit) 
    {
        if (debit) {
            payTuitionWithDebitCard(payment);
        } else {
            payTuitionWithCreditCard(payment);
        }
    }

    /*@      public exceptional_behavior
      @   	requires !(initialBalance <= maxSemesterCredits * costPerCredit + maxSemesterCredits * ((costPerCredit/100)*6))
      @           || !(0 <= passedCredits && passedCredits <= totalCredits)
      @           || !(0 <= semesterCredits && semesterCredits <= maxSemesterCredits);
      @   	signals_only IllegalArgumentException;
      @ also
      @      public normal_behavior
      @   	assignable this.*;
      @   	old int latePerCredit = (costPerCredit/100)*6;
      @  	old int penalty = (payment/100)*2;
      @   	old int _payment = payment - penalty;
      @   	requires 0 < payment; 
      @   	requires passedCredits + semesterCredits <= totalCredits;
      @   	requires initialBalance <= maxSemesterCredits * costPerCredit + maxSemesterCredits * ((costPerCredit/100)*6);
      @   	requires 0 <= semesterCredits && semesterCredits <= maxSemesterCredits;
      @   	requires 0 <= passedCredits && passedCredits <= totalCredits;
      @   	requires lateRegistration ==> initialBalance + maxSemesterCredits * costPerCredit + maxSemesterCredits * latePerCredit <= Integer.MAX_VALUE;
      @   	requires !lateRegistration ==> initialBalance + maxSemesterCredits * costPerCredit <= Integer.MAX_VALUE;
      @   	requires debit ==> Integer.MIN_VALUE <= initialBalance - payment;
      @   	requires !debit ==> Integer.MIN_VALUE <= initialBalance - _payment;
      @   	{|
      @ 		requires !lateRegistration && option && initialBalance <= 0;
      @ 		ensures this.tuitionBalance == initialBalance + enrollmentCredits * costPerCredit;
      @ 		ensures this.enrollmentCredits == semesterCredits;
      @ 		ensures this.passedCredits == passedCredits;
      @ 	also
      @ 		requires lateRegistration && option && initialBalance <= 0;
      @ 		ensures this.tuitionBalance == initialBalance + enrollmentCredits * costPerCredit + enrollmentCredits * latePerCredit;
      @ 		ensures this.enrollmentCredits == semesterCredits;
      @ 		ensures this.passedCredits == passedCredits;
      @ 	also
      @ 		requires debit && validPayment(payment) && !option;
      @ 		ensures this.tuitionBalance == initialBalance - payment;	        
      @ 	also
      @ 		requires !debit && validPayment(_payment) && !option;
      @ 		ensures this.tuitionBalance == initialBalance - _payment;
      @         |} @*/
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
