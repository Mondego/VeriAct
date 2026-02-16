public class BankAccount 
{
	int balance;
	int previousTransaction;
        //@ invariant 0 <= balance;

	//@ assignable \everything;
	//@ ensures balance == 0; 
	//@ ensures previousTransaction == 0; 
	BankAccount()
	{
		balance = 0;
		previousTransaction = 0;
	}

	//@ assignable \everything; 
	//@ ensures (currentBalance <= 0) ==> balance == 0; 
	//@ ensures (0 < currentBalance) ==> balance == currentBalance; 
	//@ ensures previousTransaction == 0; 
	BankAccount(int currentBalance)
	{
		if (currentBalance <= 0){
			balance = 0;
		} else {		
			balance = currentBalance;
		}
		previousTransaction = 0;
	}

	//@ assignable \everything;
	//@ ensures (currentBalance <= 0) ==> balance == 0; 
	//@ ensures (0 < currentBalance) ==> balance == currentBalance; 
	//@ ensures previousTransaction == _previousTransaction; 
	BankAccount(int currentBalance, int _previousTransaction)
	{
		if (currentBalance <= 0){
			balance = 0;
		} else {		
			balance = currentBalance;
		}
		previousTransaction = _previousTransaction;
	}

	//@ ensures \result == this.balance;
	/*@ spec_public pure @*/ int getBalance() 
	{
		return this.balance;
	}

	//@ ensures \result == this.previousTransaction;
	/*@ spec_public pure @*/ int getPreviousTransaction()
	{
		return this.previousTransaction;
	}
	
	//@ requires 0 < _amount;
	//@ ensures \result;
	
	//@ also

	//@ requires _amount <= 0;
	//@ ensures !\result;
	/*@ spec_public pure @*/  boolean isValid(int _amount)
	{
		if (0 < _amount) {
			return true;
		} else {
			return false;
		}
	}

	//@ requires isValid(_amount);
	//@ requires 0 <= _balance;
	//@ {|	
	  //@ requires 0 <= _balance - _amount;
	  //@ ensures \result;
	
	  //@ also

	  //@ requires _balance - _amount < 0;
	  //@ ensures !\result;
	//@ |}
	/*@ spec_public pure @*/ boolean isValid(int _balance, int _amount)
	{
		if (0 <= _balance - _amount) {
			return true;
		} else {
			return false;
		}
	}

	//@ requires isValid(amount);
	//@ requires amount + balance <= Integer.MAX_VALUE;
	//@ assignable balance, previousTransaction;
	//@ ensures balance == \old (balance) + amount;
	//@ ensures previousTransaction == amount;

	//@ also

	//@ requires !isValid(amount);
	//@ ensures balance == \old (balance);
	//@ ensures previousTransaction == \old (previousTransaction);
	void deposit(int amount)
	{
		if (isValid(amount)) {
			balance = balance + amount;
			previousTransaction = amount;
		} 
	}

	//@ requires isValid(amount);
	//@ requires isValid(balance, amount);
	//@ assignable balance, previousTransaction;
	//@ ensures balance == \old (balance) - amount;
	//@ ensures previousTransaction == -amount;
	  
	//@ also

	//@ requires isValid(amount);
	//@ requires !isValid(balance, amount);
	//@ ensures balance == \old (balance);
	//@ ensures previousTransaction == \old (previousTransaction);

	//@ also

	//@ requires !isValid(amount);
	//@ ensures balance == \old (balance);
	//@ ensures previousTransaction == \old (previousTransaction);
	void withdraw(int amount)
	{
		if (isValid(amount)) {
			if (isValid(balance, amount)) {
				balance = balance - amount;
				previousTransaction = -amount;
			}
		}
	}

	//@ requires isValid(amount);
	//@ requires isValid(balance, amount);
	//@ assignable balance, previousTransaction;
	//@ ensures balance == \old (balance) - amount;
	//@ ensures previousTransaction == -amount;
	  
	//@ also

	//@ requires isValid(amount);
	//@ requires !isValid(balance, amount);
	//@ requires isValid(balance, 50);  
	//@ assignable balance, previousTransaction;
	//@ ensures balance == \old (balance) - 50;
	//@ ensures previousTransaction == -50;
	
	//@ also

	//@ requires isValid(amount);
	//@ requires !isValid(balance, amount);
	//@ requires !isValid(balance, 50); 
	//@ assignable balance, previousTransaction;
	//@ ensures balance == 0;
	//@ ensures previousTransaction == \old (-balance);

	//@ also

	//@ requires !isValid(amount);
	//@ ensures balance == \old (balance);
	//@ ensures previousTransaction == \old (previousTransaction);
	void checkWithdrawal(int amount)
	{
		if (isValid(amount)) {
			if (isValid(balance, amount)) {
				balance = balance - amount;
				previousTransaction = -amount;
			}
			else {
				int notEnoughMoneyPenalty;
				notEnoughMoneyPenalty = 50;
				int _balance;
				_balance = balance - notEnoughMoneyPenalty;
				if (0 <= _balance) { 
					balance = _balance;
					previousTransaction = -notEnoughMoneyPenalty;
				}
				else {
					previousTransaction = -balance;
					balance = 0;
				}	
			}
		}
	}

	//@ old int _amount =  amount + (amount/100)*5;
	//@ requires _amount <= Integer.MAX_VALUE;
	//@ {|	
	  //@ requires isValid(_amount);
	  //@ requires isValid(balance, _amount);
	  //@ assignable balance, previousTransaction;
	  //@ ensures balance == \old (balance) - _amount;
	  //@ ensures previousTransaction == -_amount;
	  
	  //@ also 

	  //@ requires isValid(_amount);
	  //@ requires !isValid(balance, _amount);
   	  //@ ensures balance == \old (balance);
	  //@ ensures previousTransaction == \old (previousTransaction);

	  //@ also

	  //@ requires !isValid(_amount);
	  //@ ensures balance == \old (balance);
	  //@ ensures previousTransaction == \old (previousTransaction);
	//@ |}
	void foreignTransfer(int amount)
	{
		int penalty;
		penalty = (amount/100)*5;
		amount = amount + penalty;
		if (isValid(amount)) {
			if (isValid(balance, amount)) {
				balance = balance - amount;
				previousTransaction = -amount;
			}
		}
	}

	//@ old int _amount = amount - (amount/100)*5;
	//@ requires isValid(_amount);
	//@ requires _amount + balance <= Integer.MAX_VALUE;
	//@ assignable balance, previousTransaction;
	//@ ensures balance == \old (balance) + _amount;
	//@ ensures previousTransaction == _amount;

	//@ also

	//@ old int _amount = amount - (amount/100)*5;
	//@ requires !isValid(_amount);
	//@ ensures balance == \old (balance);
	//@ ensures previousTransaction == \old (previousTransaction);
	void foreignDeposit(int amount) 
	{
		int penalty;
		penalty = (amount/100)*5;
		amount = amount - penalty;
		if (isValid(amount)) {
			balance = balance + amount;
			previousTransaction = amount;
		}
	}

	//@ old int _amount = amount - (amount/100)*2;
	//@ requires isValid(_amount);
	//@ requires isValid(balance, _amount);
	//@ assignable balance, previousTransaction;
	//@ ensures balance == \old (balance) - _amount;
	//@ ensures previousTransaction == -_amount;
	  
	//@ also

	//@ old int _amount =  amount - (amount/100)*2;
	//@ requires isValid(_amount);
	//@ requires !isValid(balance, _amount); 
	//@ ensures balance == \old (balance);
	//@ ensures previousTransaction == \old (previousTransaction);

	//@ also

	//@ old int _amount =  amount - (amount/100)*2;
	//@ requires !isValid(_amount);
	//@ ensures balance == \old (balance);
	//@ ensures previousTransaction == \old (previousTransaction);
	void withdrawByCashBack(int amount) 
	{
		int cashback; 
		cashback =  (amount/100)*2;
		amount = amount - cashback;
		if (isValid(amount)) {
			if (isValid(balance, amount)) {
				balance = balance - amount;
				previousTransaction = -amount;
			}
		}
	}
	//@ old int ATMpenalty = 4;
	//@ requires amount + ATMpenalty <= Integer.MAX_VALUE;
	//@ {|
	  //@ requires isValid(amount);
	  //@ requires isValid(balance, (amount + ATMpenalty));
	  //@ assignable balance, previousTransaction;
	  //@ ensures balance == \old (balance) - (amount + ATMpenalty);
	  //@ ensures previousTransaction == -(amount + ATMpenalty);

	  //@ also 

	  //@ requires isValid(amount);
	  //@ requires !isValid(balance, (amount + ATMpenalty));
	  //@ ensures balance == \old (balance);
	  //@ ensures previousTransaction == \old (previousTransaction);

	  //@ also

	  //@ requires !isValid(amount);
	  //@ ensures balance == \old (balance);
	  //@ ensures previousTransaction == \old (previousTransaction);
	//@ |}
	void ATMWithdraw(int amount)
	{
		int ATMpenalty = 4;
		if (isValid(amount)) {
			amount += ATMpenalty;
			if (isValid(balance, amount)) {
				balance = balance - amount;
				previousTransaction = -amount;
			}
		}
	}
		
	//@ requires balance <= 20000;
	//@ ensures \result == balance/100;

	//@ also

	//@ requires 20000 < balance && balance <= 160000;
	//@ ensures \result == (balance/100)*2;

	//@ also

	//@ requires 160000 < balance && balance <= 300000 ;
	//@ ensures \result == (balance/100)*3;

	//@ also

	//@ requires 300000 < balance && balance <= Integer.MAX_VALUE;
	//@ ensures \result == (balance/100)*4;
	/*@ spec_public pure @*/int interestAfterYear () 
	{
		int interest;
		interest = 0;
		if (balance <= 20000) {
			interest = balance/100;
		} 
		else if (balance <= 160000) { 
			int _interest;
			_interest = balance/100;
			interest = _interest*2;
		}
		else if (balance <= 300000) {
			int _interest;
			_interest = balance/100;
			interest = _interest*3;
		}
		else {
			int _interest;
			_interest = balance/100;
			interest = _interest*4;
		}
		return interest;
	}

	/*@ assignable \everything;
	    requires 0 <= option && option <= 9; 
	    {|
		requires option == 1 && isValid(amount);
	  	requires amount + balance <= Integer.MAX_VALUE;
	  	ensures balance == \old (balance) + amount;
	  	ensures previousTransaction == amount;
		
	 	also

	   	requires option == 2 && isValid(amount);
	   	requires isValid(balance, amount);
	   	ensures balance == \old (balance) - amount;
	 	ensures \result == balance;
	   	ensures previousTransaction == -amount;

	 	also

	 	requires option == 3 && isValid(amount);
	   	requires isValid(balance, amount);
	   	ensures balance == \old (balance) - amount;
	   	ensures previousTransaction == -amount;

	   	also

	   	requires option == 3 && isValid(amount);
	   	requires !isValid(balance, amount);
	   	requires isValid(balance, 50);  
	   	ensures balance == \old (balance) - 50;
	   	ensures previousTransaction == -50;
	
	   	also

	   	requires option == 3 && isValid(amount);
	   	requires !isValid(balance, amount);
	   	requires !isValid(balance, 50); 
	   	ensures balance == 0;
	   	ensures previousTransaction == \old (-balance);

		also 

	 	requires option == 4;
	 	ensures \result == previousTransaction;

	 	also
		
	 	old int _amount =  amount + (amount/100)*5;
	 	requires option == 5;
		requires _amount <= Integer.MAX_VALUE;
	   	requires isValid(_amount);
	   	requires isValid(balance, _amount);
	   	ensures balance == \old (balance) - _amount;
	   	ensures previousTransaction == -_amount;

	   	also
	
	   	old int _amount =  amount + (amount/100)*5;
	 	requires option == 5;
		requires _amount <= Integer.MAX_VALUE;
	   	requires isValid(_amount);
	   	requires !isValid(balance, _amount);
	   	ensures balance == \old (balance);
	        ensures previousTransaction == \old (previousTransaction);  

	 	also

	 	old int _amount =  amount - (amount/100)*2;
	   	requires option == 6 && isValid(_amount);
	   	requires isValid(balance, _amount);
	   	ensures balance == \old (balance) - _amount;
	   	ensures previousTransaction == -_amount;

	   	also

	  	old int _amount =  amount - (amount/100)*2;
	   	requires option == 6 && isValid(_amount);
	   	requires !isValid(balance, _amount);
		ensures balance == \old (balance);
		ensures previousTransaction == \old (previousTransaction);  

	 	also

		old int _amount =  amount - (amount/100)*5;
	 	requires option == 7 && isValid(_amount);
	   	requires _amount + balance <= Integer.MAX_VALUE;
	   	ensures balance == \old (balance) + _amount;
	   	ensures previousTransaction == _amount;
		
	 	also

	   	requires option == 8 && balance <= 20000;
	   	ensures \result == balance/100;

	   	also

	   	requires option == 8 && 20000 < balance && balance <= 160000;
	   	ensures \result == (balance/100)*2;

	   	also

	   	requires option == 8 && 160000 < balance && balance <= 300000 ;
	   	ensures \result == (balance/100)*3;

	   	also

	   	requires option == 8 && 300000 < balance && balance <= Integer.MAX_VALUE;
	   	ensures \result == (balance/100)*4;

		also

		requires option == 9;
		old int ATMpenalty = 4;
		requires amount + ATMpenalty <= Integer.MAX_VALUE;
		requires isValid(amount);
		requires isValid(balance, (amount + ATMpenalty));
		ensures balance == \old (balance) - (amount + ATMpenalty);
		ensures previousTransaction == -(amount + ATMpenalty);
	 
               also

	       requires option == 0;
               ensures balance == \old (balance);
	       ensures previousTransaction == \old (previousTransaction);
	    |} @*/ 
	int menu(int option, int amount)
	{
		int result;
		result = 0;	
			
		switch (option) 
		{
			case 1:
			deposit(amount);
			result = getBalance();
			break;

			case 2:
			withdraw(amount);
			result = getBalance();
			break;
			
			case 3: 
			checkWithdrawal(amount);
			result = getBalance();
			break;

			case 4:
			result = getPreviousTransaction();
			break;

			case 5: 
			foreignTransfer(amount);
			result = getBalance();
			break;

			case 6:
			withdrawByCashBack(amount); 
	 		result = getBalance();
			break;

			case 7: 
			foreignDeposit(amount);
			result = getBalance();
			break;

			case 8:
			result = interestAfterYear();
			break;

			case 9:
			ATMWithdraw(amount);
			result = getBalance();
			break;

			default:
			result = getBalance();
               		break;
		}
	        return result;
        }
}
