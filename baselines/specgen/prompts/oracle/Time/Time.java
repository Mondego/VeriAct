public class Time {
    private /*@ spec_public @*/ int second;
    private /*@ spec_public @*/ int minute;
    private /*@ spec_public @*/ int hour;

    /*@ public invariant 0 <= second && second < 60;
      public invariant 0 <= minute && minute < 60;
      public invariant 0 <= hour && hour < 24; @*/

    //@ ensures this.hour == 23;
    //@ ensures this.minute == 59;
    //@ ensures this.second == 59;
    public /*@ pure @*/ Time() 
    {
        hour = 23;
        minute = 59;
        second = 59;
    }

    //@ requires 0 <= h && h < 24;
    //@ requires 0 <= m && m < 60;
    //@ requires 0 <= s && s < 60;
    //@ ensures this.hour == h;
    //@ ensures this.minute == m;
    //@ ensures this.second == s;
    public /*@ pure @*/ Time(int h, int m, int s) 
    {
        hour = h;
        minute = m;
        second = s;
    }
    /*@      public normal_behavior
      @     	requires 0 <= s && s < 60;
      @     	assignable this.second;
      @     	ensures this.second == s;
      @ also
      @     public exceptional_behavior
      @     	requires s < 0 || 60 <= s;
      @     	assignable \nothing;
      @     	signals_only IllegalArgumentException; @*/ 
    public void setSecond(int s) 
    {
        if (s < 0 || 60 <= s) {
            throw new IllegalArgumentException();
        } else {
            this.second = s;
        }
    } 

    /*@     public normal_behavior
      @     	requires 0 <= m && m < 60;
      @     	assignable this.minute;
      @     	ensures this.minute == m;
      @ also
      @     public exceptional_behavior
      @      	requires m < 0 || 60 <= m;
      @     	assignable \nothing;
      @     	signals_only IllegalArgumentException; @*/
    public void setMinute(int m) 
    {
        if (m < 0 || 60 <= m) {
            throw new IllegalArgumentException();
        } else {
            this.minute = m;
        }
    } 

    /*@     public normal_behavior
      @  	requires 0 <= h && h < 24;
      @	  	assignable this.hour;
      @  	ensures this.hour == h;
      @ also
      @     public exceptional_behavior
      @ 	requires h < 0 || 24 <= h;
      @     	assignable \nothing;
      @  	signals_only IllegalArgumentException; @*/
    public void setHour(int h) 
    {
        if (h < 0 || 24 <= h) {
            throw new IllegalArgumentException();
        } else {
            this.hour = h;
        }
    } 

    //@ ensures this.equals(\result) && this != \result;
    public /*@ pure @*/ Time getTime()
    {
	Time t = new Time(this.hour, this.minute, this.second);
	return t;
    }

    //@ ensures \result == second;
    public /*@ pure @*/ int getSecond()
    {
        return second;
    }

    //@ ensures \result == minute;
    public /*@ pure @*/ int getMinute()
    {
        return minute;
    }


    //@ ensures \result == hour;
    public /*@ pure @*/ int getHour() 
    {
        return hour;
    }

    //@ ensures \result == hour*60*60 + minute*60 + second;
    public /*@ pure @*/ int convertToSeconds()
    {
        return (hour*60*60 + minute*60 + second);
    }

    //@ 	requires convertToSeconds() == 0;
    //@   	ensures convertToSeconds() == 0;
    //@ also
    //@   	requires convertToSeconds() != 0;
    //@   	assignable second, minute, hour;
    //@   	ensures convertToSeconds() == \old(convertToSeconds() - 1);
    public void decr()
    {
        if (isTimeZero())
            return;
        else {
            second--;
            if(second < 0) {
                second = 59;
                minute--;
                if (minute < 0 ) { 
                    minute = 59; 
                    hour--;
                }
            }
        }
    }

    //@ assignable second, minute, hour;
    //@ ensures convertToSeconds() == 0;
    public void timer()
    {
        //@ ghost boolean flag = false;
        //@ maintaining !isTimeZero() && flag ==> convertToSeconds() == \old (convertToSeconds() - 1);
        while (!isTimeZero()) {
            //@ set flag = true;
            // each time around this loop should take 1 second, ideally
            decr();
        }
    }

    
    //@ requires 0 <= h && h < 24;
    //@ requires 0 <= m && m < 60;
    //@ requires 0 <= s && s < 60;
    //@ assignable this.second, this.minute, this.hour;
    //@ ensures convertToSeconds() == 0;
    public void timer(int h, int m, int s)
    {
        setHour(h);
        setMinute(m);
        setSecond(s);
	//@ assert hour == h && minute == m && second == s;
        timer();
    }

    //@ ensures \result == (convertToSeconds() == 0);
    public /*@ pure */ boolean isTimeZero() 
    {
        return (convertToSeconds() == 0);
    }

    //@ assignable second, minute, hour;
    //@ ensures second == 0 && minute == 0 && hour == 0;
    public void reset()
    {
        second = 0;
        minute = 0; 
        hour = 0;
    }

    /*@ ensures \result == ((this.hour > start.hour) 
      @                   || (this.hour == start.hour && this.minute > start.minute) 
      @                   || (this.hour == start.hour && this.minute == start.minute && this.second > start.second)); @*/
    public /*@ pure @*/ boolean later_than(Time start) 
    {
        if (this.hour != start.hour) {
            return this.hour > start.hour;
        } else if (this.minute != start.minute) {
            return this.minute > start.minute;
        } else { 
            return this.second > start.second;
        }
    }

    //@ also
    //@    	requires !(o instanceof Time);
    //@    	ensures !\result;
    //@ also
    //@    	requires (o instanceof Time);
    /*@    	ensures \result <==> (this.hour == ((Time) o).hour)
      @                               && (this.minute == ((Time) o).minute)
      @                        	      && (this.second == ((Time) o).second); @*/
    public boolean equals(Object o) 
    {
        if (!(o instanceof Time)) {
            return false;
        }
        Time t = (Time) o;
        return this.hour == t.hour && this.minute == t.minute && this.second == t.second;
    }

    //@ requires stop.later_than(start) || stop.equals(start);		
    //@ old int _stop_minutes = (stop.second < start.second) ? (stop.minute -1): stop.minute;
    //@ old int diff_seconds = (stop.second < start.second) ? (stop.second + 60 - start.second) : (stop.second - start.second);
    //@ old int _stop_hours = (_stop_minutes < start.minute) ? (stop.hour -1): stop.hour;
    //@ old int diff_minutes = (_stop_minutes < start.minute) ? (_stop_minutes + 60 - start.minute) : (_stop_minutes - start.minute);
    //@ old int diff_hours = _stop_hours - start.hour;
    //@ ensures diff_hours == \result.hour;
    //@ ensures diff_minutes == \result.minute;
    //@ ensures diff_seconds == \result.second;
    private /*@ spec_public pure @*/ Time trustedDifference(Time start, Time stop) 
    {
        Time diff = new Time();
        int temp_second = stop.getSecond();
        int temp_minute = stop.getMinute();
        int temp_hour = stop.getHour();
       
        if (temp_second < start.getSecond()) {
            --temp_minute;
            temp_second += 60;
        }
	
        diff.second = temp_second - start.getSecond();

        if (temp_minute < start.getMinute()) {
            --temp_hour;
            temp_minute += 60;
        }

        diff.minute = temp_minute - start.getMinute();
        diff.hour = temp_hour - start.getHour();
        return(diff);
    }

    //@    	requires stop.later_than(start);
    //@ 	ensures \result.equals(trustedDifference(start,stop));
    //@ also
    //@    	requires start.later_than(stop) || stop.equals(start);
    //@    	ensures \result.equals(trustedDifference(stop,start));
    public /*@ pure @*/ Time difference(Time start, Time stop)
    {
	if (stop.later_than(start)) {
            return trustedDifference(start, stop);
	} else {
            return trustedDifference(stop, start);
	}
    }	

    /*@ requires 0 <= sel && sel < 5;
      @ {|
      @		requires 0 <= sel && sel <= 2;
      @		ensures \result.hour == 0 && \result.minute == 0 && \result.second == 0;
      @		ensures  start == \old (start);
      @		ensures  stop == \old (stop);
      @ also
      @		requires sel == 3 && !start.equals(stop);
      @		ensures \result.hour == \old (hour) && \result.minute == \old (minute) && \result.second == \old (second);
      @	also
      @		requires sel == 3 && start.equals(stop);
      @		ensures  \result.hour == 0 && \result.minute == 0 && \result.second == 0;
      @		ensures  start.hour == 0 && start.minute == 0 && start.second == 0;
      @		ensures  stop == \old (stop);
      @	also
      @		requires sel == 4;
      @		assignable \nothing;
      @		ensures \result.equals(difference(start, stop));
      @		ensures  start == \old (start);
      @		ensures  stop == \old (stop);
      @ |}
    @*/     
    public Time timeOptions(Time start, Time stop, int sel) {
	if (sel == 0) {
		reset();
	} else if (sel == 1) {
		timer(start.hour, start.minute, start.second);
	} else if (sel == 2) {
		timer();
	} else if (sel == 3) {
		if (start.equals(stop)) {
			start.reset();
			return start.getTime();
		}
	} else {
   		return difference(start, stop);
	}
	return getTime();
    }
}	
