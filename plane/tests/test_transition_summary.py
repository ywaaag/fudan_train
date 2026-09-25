import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from summarize_transitions import response_metrics


def test_stop_distance_is_invalid_after_reset():
    from summarize_transitions import summarize
    trace=[{'time':i*.1,'vx':[0.],'yaw':[0.],'height':[.4],
            'path_since_switch_m':[1.2],'wheel_contacts':[[True,True]],'roll':[0.]} for i in range(20)]
    data={'response_trace':trace,'envs_per_command':1,'switch_at':.5,'initial_commands':[[4,0,.4]],
          'results':[{'command':[0,0,.4],'failure_count':0,'timeout_count':0}],
          'checkpoint':'test','seed':1}
    assert summarize(data)['rows'][0]['stop_distance_m']==[1.2]
    data['results'][0]['failure_count']=1
    assert summarize(data)['rows'][0]['stop_distance_m']==[None]


def test_settling_rejects_short_crossing_and_late_drift():
    times=np.arange(0,4,.1)
    values=np.ones_like(times)
    values[10:16]=0
    out=response_metrics(times,values,0,1,1,.05)
    assert out['first_sustained_band_s'] == 0
    assert out['settling_time_s'] is None


def test_directional_overshoot_and_retained_settling():
    times=np.arange(0,4,.1)
    values=np.zeros_like(times)
    values[10:15]=1.2
    values[15:]=1
    out=response_metrics(times,values,1,0,1,.1)
    assert abs(out['directional_overshoot']-.2)<1e-6
    assert abs(out['settling_time_s']-.5)<1e-6
