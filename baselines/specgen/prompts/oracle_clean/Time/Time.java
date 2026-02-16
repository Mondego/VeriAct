public class Time {
    private  int second;
    private  int minute;
    private  int hour;

    public  Time() 
    {
        hour = 23;
        minute = 59;
        second = 59;
    }

    public  Time(int h, int m, int s) 
    {
        hour = h;
        minute = m;
        second = s;
    }
     
    public void setSecond(int s) 
    {
        if (s < 0 || 60 <= s) {
            throw new IllegalArgumentException();
        } else {
            this.second = s;
        }
    } 

    
    public void setMinute(int m) 
    {
        if (m < 0 || 60 <= m) {
            throw new IllegalArgumentException();
        } else {
            this.minute = m;
        }
    } 

    
    public void setHour(int h) 
    {
        if (h < 0 || 24 <= h) {
            throw new IllegalArgumentException();
        } else {
            this.hour = h;
        }
    } 

    
    public  Time getTime()
    {
	Time t = new Time(this.hour, this.minute, this.second);
	return t;
    }

    
    public  int getSecond()
    {
        return second;
    }

    
    public  int getMinute()
    {
        return minute;
    }


    
    public  int getHour() 
    {
        return hour;
    }

    
    public  int convertToSeconds()
    {
        return (hour*60*60 + minute*60 + second);
    }

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

    public void timer()
    {
        
        
        while (!isTimeZero()) {
            
            
            decr();
        }
    }

    public void timer(int h, int m, int s)
    {
        setHour(h);
        setMinute(m);
        setSecond(s);
	
        timer();
    }

    
    public  boolean isTimeZero() 
    {
        return (convertToSeconds() == 0);
    }

    
    
    public void reset()
    {
        second = 0;
        minute = 0; 
        hour = 0;
    }

    
    public  boolean later_than(Time start) 
    {
        if (this.hour != start.hour) {
            return this.hour > start.hour;
        } else if (this.minute != start.minute) {
            return this.minute > start.minute;
        } else { 
            return this.second > start.second;
        }
    }

    public boolean equals(Object o) 
    {
        if (!(o instanceof Time)) {
            return false;
        }
        Time t = (Time) o;
        return this.hour == t.hour && this.minute == t.minute && this.second == t.second;
    }

    private  Time trustedDifference(Time start, Time stop) 
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

    public  Time difference(Time start, Time stop)
    {
        if (stop.later_than(start)) {
                return trustedDifference(start, stop);
        } else {
                return trustedDifference(stop, start);
        }
    }	
   
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
