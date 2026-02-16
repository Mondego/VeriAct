public class StackQueue {
	public class Stack { 
	 	public static final int MAX = 100;   
	
	   	private  int top;

	    private  int arr[] = new int[MAX];

	   	Stack() 
	   	{ 
	        top = -1; 
	    } 
	
	   	public  int getTop() 
	    {
	        return top;
	   	}

		public  boolean isEmpty() 
		{ 
			return (getTop() < 0); 
		} 
	
		public  boolean isFull() 
		{ 
			return (top == MAX - 1); 
		} 

		public void push(int x) 
		{ 
			if (!isFull()) {
					arr[++top] = x; 
			} else {
				throw new IllegalArgumentException();
			}
		} 

		public int pop() 
		{ 
			if (!isEmpty()) {
					return arr[top--]; 
				} else {
					throw new IllegalArgumentException();
				}
		}

		public  int peek() 
		{ 
			if (!isEmpty())
					return arr[top]; 
			else
					throw new IllegalArgumentException();
   	   	} 

		public  int search(int key) 
		{
			int index = top;
			
			
			while (0 <= index) {
				if (getElem(index) == key) {
					return index;
				}
				index--;
			}
			return -1;
		}	

		public  boolean isContain(int key)
		{	
			int index = top;
			
			
			while (0 <= index) {
				if (key == getElem(index)) {
					return true;
				}
				index--;
			}
			return false;
		}	

		public  int size() 
		{ 
			return getTop() + 1;
		}

		public  int getElem(int i) 
		{ 
			return arr[i]; 
		}
	};

	public class Queue { 
		public static final int MAX = 100;   
		private int front, rear; 
		private final int queue[] = new int[MAX];

		public  Queue() 
		{ 
			front = rear = 0;  
		} 

   		public void enter(int data) 
		{ 
			if (!isFull()) { 
					queue[rear] = data; 
					rear++; 
			} else { 
					throw new IllegalArgumentException(); 
			} 
		} 

		public int delete() 
		{ 
			if (!isEmpty()) { 
					int poll = queue[front];
					
					int i = 0;
					
					while (i < rear - 1) {
						
						queue[i] = queue[i + 1];
						i++;
					}
					rear--;
					return poll; 
			} else {
					throw new IllegalArgumentException();
			}
		} 

		public  int peek() 
		{ 
			if (!isEmpty()) { 
					return queue[front]; 
			} else {
					throw new IllegalArgumentException(); 
			}
		} 

		public  boolean isContain(int key)
		{
			int index = 0;
			
			
			while (index < rear) {
				if (key == queue[index]) {
						return true;
					}
				index++;
				}
			return false;
		}

		public  int search(int key)
		{
			int index = 0;
			
			
			while (index < rear) {
					if (key == queue[index]) {
						return index;
					}
				index++;
			}
			return -1;
		}

   		public  boolean isEmpty() 
   		{
   		     	return (getRear() == getFront());
		}

		public  boolean isFull() 
		{
				if (MAX == getRear()) 
					return true;
				else
					return false;
		}

		public  int size()
		{
				return rear;
		}

   		public  int getFront() 
    	{	
       			return front; 
   		}
	
		public  int getRear() 
		{ 
			return rear; 
		}

		public  int getElem(int i) 
		{ 
			return queue[i]; 
		}
	}; 

    	
	public int stackPlus(Stack stack) 
	{
	stack.push(stack.pop() + stack.pop());
	return stack.peek();
	}

	
	public int stackMinus(Stack stack) 
	{
	stack.push(stack.pop() - stack.pop());
	return stack.peek();
	}

	
	public int stackDivision(Stack stack) 
	{
	stack.push(stack.pop() / stack.pop());
	return stack.peek();
	}

	
	public int stackModulus(Stack stack) 
	{
	stack.push(stack.pop() % stack.pop());
	return stack.peek();
	}

	
	public int QPlus(Queue Q) 
	{
	Q.enter(Q.delete() + Q.delete());
	return Q.getElem(Q.getRear() - 1);
	}

	
	public int QMinus(Queue Q) 
	{
	Q.enter(Q.delete() - Q.delete());
	return Q.getElem(Q.getRear() - 1);
	}

	
	public int QDivision(Queue Q) 
	{
	Q.enter(Q.delete() / Q.delete());
	return Q.getElem(Q.getRear() - 1);
	}

	
	public int QModulus(Queue Q) 
	{
	Q.enter(Q.delete() % Q.delete());
	return Q.getElem(Q.getRear() - 1);
	}	


	

	public  int plusQStack(Queue Q, Stack stack) 
	{
	return Q.peek() + stack.peek();
	}

	
	public  int minusQStack(Queue Q, Stack stack) 
	{
	return Q.peek() - stack.peek();
	}

	
	public  int qDivideStack(Queue Q, Stack stack) 
	{
	return Q.peek() / stack.peek();
	}

	
	public  int stackDivideQ(Queue Q, Stack stack) 
	{
	return  stack.peek()/Q.peek();
	}	

	
	public  int qModulusStack(Queue Q, Stack stack) 
	{
	return Q.peek() % stack.peek();
	}

	
	public  int stackModulusQ(Queue Q, Stack stack) 
	{
	return  stack.peek() % Q.peek();
	}
	
	
    public int driverStack(Stack stack, int op, int input) 
    {
		int output = 0;
		switch (op) {
          		case 0:
            		stack.push(input);
			break;

			case 1:
            		output = stack.pop();
			break;

			case 2:
            		output = stack.search(input);
			break;

			case 3:
            		output = (stack.isContain(input)) ? 1 : 0;
			break;

			case 4:
            		output = stackPlus(stack);
			break;

			case 5:
            		output = stackMinus(stack);
               		 break;

			case 6:
            		output = stackDivision(stack);
               		break;

			case 7:
			output = stackModulus(stack);
			break;

			default:
            		output = stack.size();
			break;
	 	}
		return output;
    }

    	 
   	public int driverQueue(Queue q, int op, int input)
    {
		int output = 0;
		switch (op) {
			case 0:
           		q.enter(input);
                	break;

			case 1:
            		output = q.delete();
                	break;

			case 2:
            		output = q.search(input);
                	break;

			case 3:
            		output = q.isContain(input) ? 1 : 0;
               		break;

			case 4:
            		output = QPlus(q);
                	break;

			case 5:
            		output = QMinus(q);
                	break;

			case 6:
            		output = QDivision(q);
                	break;

			case 7:
			output = QModulus(q);
			break;

			default:
            		output = q.size();
                	break;
		}
		return output;
    }

    	
    	public int driverQStack(Stack stack, Queue q, int op) 
    	{
			StackQueue sq = new StackQueue();
			int output = 0;
			switch (op) {
				case 0:
					output = sq.plusQStack(q, stack);
						break;

				case 1:
						output = sq.minusQStack(q, stack);
						break;

				case 2:
						output = sq.qDivideStack(q, stack);
						break;

				case 3:
				output = sq.stackDivideQ(q, stack);
				break;

				case 4:
				output = sq.qModulusStack(q, stack);
				break;

				default:
				output = sq.stackModulusQ(q, stack);
				break;
			}
			return output;
    	}
}
