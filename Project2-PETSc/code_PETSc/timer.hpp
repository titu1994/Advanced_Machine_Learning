#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/param.h>
#include <sys/times.h>

#if defined(CLK_TCK)
#define TIMES_TICKS_PER_SEC double(CLK_TCK)
#elif defined(_SC_CLK_TCK)
#define TIMES_TICKS_PER_SEC double(sysconf(_SC_CLK_TCK))
#elif defined(HZ)
#define TIMES_TICKS_PER_SEC double(HZ)
#endif

// Class for recording CPU and wall-clock time (in seconds) of program segments.

class timer {
private:
  
  timeval wallclock;
  struct tms begin;
  struct tms end;
  long   n;                  // number of intervals 
  double cpu_cur;            // starting cpu time
  double wallclock_cur;      // starting wallclock time
  
public:
  
  double cpu_max;            // longest recorded cpu time interval
  double cpu_min;            // shortest recorded cpu time interval
  double cpu_avg;            // average cpu time interval recorded
  double cpu_total;          // total cpu time interval recorded
  
  double wallclock_max;      // longest recorded wallclock time interval
  double wallclock_min;      // shortest recorded wallclock time interval
  double wallclock_avg;      // average wallclock time interval recorded
  double wallclock_total;    // total wallclock time interval recorded
  
  timer(); 
  ~timer(){} 
  
  void   start();            
  void   stop();             
  void   reset();            // reset internal state 
};

#endif
