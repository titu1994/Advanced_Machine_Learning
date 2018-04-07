#include "timer.hpp"
#include <algorithm>

timer::timer()
  :n(0),
   cpu_cur(0),
   wallclock_cur(0),
   cpu_max(0),
   cpu_min(99999999),
   cpu_avg(0),
   cpu_total(0),
   wallclock_max(0),
   wallclock_min(99999999),
   wallclock_avg(0),
   wallclock_total(0){}

void timer::start(){
  times(&begin); 
  cpu_cur=((double)begin.tms_utime+(double)begin.tms_stime)/TIMES_TICKS_PER_SEC;
  gettimeofday(&wallclock, NULL);
  wallclock_cur=(double)wallclock.tv_sec+(double)wallclock.tv_usec/1e6;
}


void timer::stop(){  
  n++;
  
  times(&end);
  cpu_cur=(((double)end.tms_utime+(double)end.tms_stime)/TIMES_TICKS_PER_SEC)-cpu_cur;  
  cpu_total+=cpu_cur;
  cpu_max=std::max(cpu_max,cpu_cur);
  cpu_min=std::min(cpu_min,cpu_cur);
  cpu_avg=cpu_total/n;
  
  gettimeofday(&wallclock, NULL);
  wallclock_cur=((double)wallclock.tv_sec+((double)wallclock.tv_usec/1e6))-wallclock_cur;
  wallclock_total+=wallclock_cur;
  wallclock_max=std::max(wallclock_max,wallclock_cur);
  wallclock_min=std::min(wallclock_min,wallclock_cur);  
  wallclock_avg=wallclock_total/n;
}


void timer::reset(){
  n=0;
  cpu_max=0;
  cpu_min=99999999;
  cpu_avg=0;
  cpu_total=0;
  cpu_cur=0;
  
  wallclock_max=0;
  wallclock_min=99999999;
  wallclock_avg=0;
  wallclock_total=0;
  wallclock_cur=0;
}
