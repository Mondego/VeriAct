public class BankAccount 
{
	int balance;
	int previousTransaction;
        
	BankAccount()
	{
		balance = 0;
		previousTransaction = 0;
	}

	BankAccount(int currentBalance)
	{
		if (currentBalance <= 0){
			balance = 0;
		} else {		
			balance = currentBalance;
		}
		previousTransaction = 0;
	}
	
	BankAccount(int currentBalance, int _previousTransaction)
	{
		if (currentBalance <= 0){
			balance = 0;
		} else {		
			balance = currentBalance;
		}
		previousTransaction = _previousTransaction;
	}

	int getBalance() 
	{
		return this.balance;
	}

	int getPreviousTransaction()
	{
		return this.previousTransaction;
	}

	boolean isValid(int _amount)
	{
		if (0 < _amount) {
			return true;
		} else {
			return false;
		}
	}
	
	 boolean isValid(int _balance, int _amount)
	{
		if (0 <= _balance - _amount) {
			return true;
		} else {
			return false;
		}
	}

	void deposit(int amount)
	{
		if (isValid(amount)) {
			balance = balance + amount;
			previousTransaction = amount;
		} 
	}
	
	void withdraw(int amount)
	{
		if (isValid(amount)) {
			if (isValid(balance, amount)) {
				balance = balance - amount;
				previousTransaction = -amount;
			}
		}
	}

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
		

	int interestAfterYear () 
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
