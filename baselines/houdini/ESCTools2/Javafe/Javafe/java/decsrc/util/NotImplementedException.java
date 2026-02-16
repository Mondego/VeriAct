package decsrc.util;

public class NotImplementedException extends RuntimeException {
  private static final long serialVersionUID = 852607024405642641L;
  
  NotImplementedException(String s) {
    super(s);
  }
}
