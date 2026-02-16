
//@ model import org.jmlspecs.lang.JMLDataGroup;

public class Timer{
  // public JMLDatagroup time, alarm;
  //@ model public JMLDatagroup time, alarm;
  int time_hrs, time_mins, time_secs; //@ in time;
  int alarm_hrs, alarm_mins, alarm_secs; //@ in alarm;

  //@ assignable time;
  public void tick() { time_secs++; }

  //@ assignable alarm;
  public void setAlarm(int hrs, int mins, int secs)  {
    alarm_hrs = hrs;
    alarm_mins = mins;
    alarm_secs = secs;
  }
}

