import bisect

import math

class CommandSequence:
    def __init__(self,data):
        self.mode=data.get('interpolation','linear')
        self.points=data['keyframes']
        if self.mode not in ('linear','step') or len(self.points)<2:
            raise ValueError('Expected linear/step sequence with at least two keyframes')
        self.times=[p['time'] for p in self.points]
        if self.times[0]!=0 or not all(math.isfinite(t) for t in self.times) or any(b<=a for a,b in zip(self.times,self.times[1:])):
            raise ValueError('Times must start at zero and increase strictly')
        for p in self.points:
            c=p['command']
            if len(c)!=3 or not all(math.isfinite(v) for v in c) or abs(c[0])>4 or abs(c[1])>4 or c[2]!=.4:
                raise ValueError('Expected bounded vx/yaw and fixed .40m height')

    def at(self,t):
        i=max(0,bisect.bisect_right(self.times,t)-1)
        c=self.points[i]['command']
        if i==len(self.points)-1 or self.mode=='step':return list(c)
        fraction=max(0.,min(1.,(t-self.times[i])/(self.times[i+1]-self.times[i])))
        return [a+(b-a)*fraction for a,b in zip(c,self.points[i+1]['command'])]

