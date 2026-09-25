"""Course aggregation must use only selected episodes and preserve signed semantics."""
import torch

from wheel_legged_gym.domain.commands.metrics_state import create_command_metrics
from wheel_legged_gym.domain.commands.curriculum_window import create_window, accumulate_window


def test_selected_episode_sums_and_fresh_window():
    metrics=create_command_metrics(3,'cpu')
    for buffer in metrics.buffers():buffer[:]=torch.tensor([1.,100.,3.])
    metrics.reverse_command_sum[:]=torch.tensor([-2.,-100.,-4.])
    window=create_window('cpu');other=create_window('cpu')
    accumulate_window(window,torch.tensor([0,2]),timeouts=torch.tensor([True,True,False]),metrics=metrics)
    assert window['timeouts'].item()==1
    assert window['reverse_command_abs_sum'].item()==6
    for key in set(window)-{'timeouts','reverse_command_abs_sum'}:assert window[key].item()==4
    assert all(value.item()==0 for value in other.values())
    assert all(value.shape==() and value.dtype==torch.float32 for value in window.values())
    accumulate_window(window,torch.tensor([],dtype=torch.long),timeouts=torch.zeros(3,dtype=torch.bool),metrics=metrics)
    assert window['timeouts'].item()==1 and window['reverse_command_abs_sum'].item()==6
    assert metrics.reverse_command_sum.tolist()==[-2.,-100.,-4.]
