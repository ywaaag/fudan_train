import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import isaacgym
import torch

from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.domain.commands.turn_envelope import TurnEnvelope
from wheel_legged_gym.domain.commands.height_skill import HeightSkill
from wheel_legged_gym.domain.commands.resampling import CommandBuffers, resample
from wheel_legged_gym.learning.modules.policy_retention import reference_loss
from wheel_legged_gym.experiments.recipes.turn_lean_long import (
    R10200, R10200_SHA, apply_turn_lean_long, validate_spec,
)
from wheel_legged_gym.app import turn_lean_review
from wheel_legged_gym.app.turn_lean_review import command as review_command, target_heights, entry_exit_pairs
from wheel_legged_gym.app.turn_lean_monitor import probe_command, height_skill_stagnant
from tools.summarize_turn_envelope import lean_aware_checks, height_skill_checks


def spec():
    return {'source_checkpoint': R10200, 'source_sha256': R10200_SHA,
            'source_iteration': 10200, 'teacher_checkpoint': R10200,
            'teacher_sha256': R10200_SHA,
            'course_plan': [{'phase':phase,'budget_max':10000} for phase in (1,2,3)],
            'stage': {'phase': 1, 'turn_height': .38, 'lean_max_deg': 2.5,
                      'turn_fraction': .5, 'pairs': [[.5,.25],[1.,.5],[2.,.75]],
                      'turn_height_reward_scale': 1.,
                      'learning_rate': 1e-5, 'freeze_encoder': True,
                      'encoder_learning_rate': None, 'entry_range': [.5,1.5],
                      'hold_range': [2.,5.], 'speed_ramp_seconds': 2.,
                      'yaw_ramp_seconds': 2., 'hypothesis': 'test'}}


def test_recipe_keeps_contract_and_retention():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    manifest = apply_turn_lean_long(cfg, train, spec=spec())
    assert cfg.commands.turn_stride == 2
    assert cfg.commands.turn_height == .38
    assert cfg.commands.ranges.height == [.4, .4]
    assert len(cfg.commands.retention_bank) == 50
    assert len(cfg.commands.turn_bank) == 12
    assert set(cfg.commands.turn_bank) == {(sv*v,sw*w) for v,w in ((.5,.25),(1.,.5),(2.,.75))
                                          for sv in (-1.,1.) for sw in (-1.,1.)}
    assert cfg.rewards.turn_lean_max_rad == math.radians(2.5)
    assert cfg.rewards.scales.orientation == -100.
    assert train.algorithm.learning_rate == 1e-5
    assert train.runner.save_interval == 250
    assert manifest['dynamic_reference_coef'] == 1.
    assert manifest['freeze_motion_encoder']


def test_height_tracks_public_yaw_and_recovers():
    scheduler = TurnEnvelope(4, 'cpu', .01, stride=2, turn_height=.38)
    commands = torch.tensor([[0.,0.,.4],[1.,0.,.4],[0.,0.,.4],[-1.,0.,.4]])
    ids = torch.tensor([0,2])
    scheduler.reset(commands, ids, torch.tensor([[1.,.5],[-1.,-.5]]))
    scheduler.entry[ids] = .5
    scheduler.hold[ids] = 2.
    for _ in range(250):
        scheduler.advance(commands)
    assert commands[0, 1] == 0. and commands[0, 2] == .4
    for _ in range(200):
        scheduler.advance(commands)
    assert commands[0, 1] == .5 and abs(commands[0, 2]-.38) < 1e-6
    assert commands[2, 1] == -.5 and abs(commands[2, 2]-.38) < 1e-6
    for _ in range(400):
        scheduler.advance(commands)
    assert commands[0, 1] == 0. and abs(commands[0, 2]-.4) < 1e-6
    torch.testing.assert_close(commands[[1,3]], torch.tensor([[1.,0.,.4],[-1.,0.,.4]]))


def test_spec_rejects_unsafe_changes():
    original = spec()
    for change in ({'turn_height': .33}, {'lean_max_deg': 11.},
                   {'learning_rate': 4e-5}, {'turn_fraction': 1.}):
        modified = copy.deepcopy(original)
        modified['stage'].update(change)
        try:
            validate_spec(modified)
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe stage accepted: ' + str(change))
    continuation = copy.deepcopy(original)
    continuation['source_iteration'] = 13250
    continuation['source_checkpoint'] = '/tmp/model_13250.pt'
    continuation['source_sha256'] = 'reviewed-checkpoint-hash'
    continuation['stage']['phase'] = 2
    assert validate_spec(continuation) is continuation


def test_lean_gate_preserves_original_safety_but_allows_reviewed_roll():
    metrics = dict(nonwheel_contact=0., left_contact=1., right_contact=1.,
                   height_mae=.005, vx_mae=.02, yaw_mae=.02, abs_yaw=.5,
                   torque_saturation=0., abs_lateral_velocity=.01, slip_rms=.02,
                   abs_pitch=.01, height=.36, height_target=.36,
                   roll_target=.14, roll_target_mae=.01, roll_overshoot=.005,
                   abs_roll=.14)
    row = {'command':[2.,.5,.36], 'metrics':metrics, 'failure_count':0,
           'timeout_count':0, 'survival_fraction':1.,
           'max_abs_metrics':{'abs_roll':.15},
           'per_env_metrics':{'vx_mae':[.02,.03], 'yaw_mae':[.02,.03],
                              'height_mae':[.005,.006], 'height':[.36,.36],
                              'roll_target_mae':[.01,.01], 'roll_overshoot':[.005,.005],
                              'abs_pitch':[.01,.01], 'abs_lateral_velocity':[.01,.01],
                              'slip_rms':[.02,.02], 'soft_joint_limit_margin':[.5,.5],
                              'left_contact':[1.,1.], 'right_contact':[1.,1.]}}
    verdict = lean_aware_checks(row)
    assert verdict['passed']
    assert not verdict['historical_strict_turn']['passed']
    row['metrics']['roll_target_mae'] = .04
    row['per_env_metrics']['roll_target_mae'][0] = .04
    assert not lean_aware_checks(row)['passed']


def test_review_entry_exit_restores_height_with_public_command():
    argv=review_command('/tmp',R10200,'/tmp/new.json',[1.], [.5],[.38],19,8,
                        spec=spec(),exit_at=12)
    assert argv[argv.index('--height-delay-seconds')+1]=='2.0'
    assert '--height-return-at-exit' in argv
    assert argv[argv.index('--trace-stride')+1]=='10'


def test_review_heights_follow_each_turn_load_and_keep_legacy_fixed_height():
    pairs = [(-1.,-.5),(-2.5,.6),(3.,-.7),(-3.,1.),(4.,1.),(0.,4.)]
    stage_spec = spec()
    assert target_heights(pairs,None) == [.4]*len(pairs)
    assert target_heights(pairs,stage_spec) == [.38]*len(pairs)
    stage_spec['stage'].update(turn_height=.4,
                               height_schedule=[[1.5,.38],[2.5,.36]])
    assert target_heights(pairs,stage_spec) == [.4,.38,.38,.36,.36,.4]
    assert len(entry_exit_pairs(stage_spec,[(1.,.5),(2.5,.6),(3.,1.)])) == 8


def test_review_cli_passes_scheduled_heights_to_evaluator(tmp_path,monkeypatch):
    job=tmp_path/'job'
    job.mkdir()
    (job/'holdout_commands.json').write_text(json.dumps({'pairs':[[3.5,.8]]}))
    checkpoint=tmp_path/'source.pt'
    checkpoint.write_bytes(b'isolated-checkpoint')
    s=spec()
    s['stage'].update(turn_height=.4,phase=3,
        pairs=[[1.,.5],[2.5,.6],[3.,1.]],
        height_schedule=[[1.5,.38],[2.5,.36]],
        cohort_plan={'fractions':{'retention':.5,'height':.2,'mid_turn':.25,'high_turn':.05},
            'height_bank':[[0.,.38],[-.5,.38],[.5,.38]],
            'mid_pairs':[[1.,.5],[2.5,.6]],'high_pairs':[[3.,1.]],
            'unclip_base_height':True})
    spec_path=tmp_path/'spec.json'
    spec_path.write_text(json.dumps(s))
    passed=[]
    def fake_run(argv,**kwargs):
        passed.append(argv)
        out=Path(argv[argv.index('--out')+1])
        out.write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
    monkeypatch.setattr(turn_lean_review.subprocess,'run',fake_run)
    monkeypatch.setattr(sys,'argv',['review_turn_lean_long.py','--job',str(job),
        '--checkpoint',str(checkpoint),'--spec',str(spec_path),
        '--review-name','scheduled_height','--model-label','source',
        '--seeds','19','--envs-per-command','1'])
    turn_lean_review.main(tmp_path)
    by_name={Path(argv[argv.index('--out')+1]).stem:argv for argv in passed}
    def heights(name):
        argv=by_name[name]
        return [float(v) for v in argv[argv.index('--height-commands')+1:argv.index('--profile')]]
    assert heights('retention') == [.4]*25
    assert heights('turn_train') == [.4]*4+[.38]*4+[.36]*4
    assert heights('turn_holdout') == [.36]*4
    assert heights('entry_exit') == [.38]*4+[.36]*4
    assert heights('height_skill') == [.38]*3+[.36]*3
    assert heights('height_entry_exit') == [.38]*3+[.36]*3
    assert '--height-return-at-exit' in by_name['height_entry_exit']


def test_height_skill_requires_every_environment_to_reach_target():
    metrics=dict(nonwheel_contact=0.,left_contact=1.,right_contact=1.,
        height_mae=.01,vx_mae=.01,yaw_mae=.01,abs_yaw=.01,
        torque_saturation=0.)
    per_env=dict(height_mae=[.01,.014],vx_mae=[.01,.02],yaw_mae=[.01,.02],
        left_contact=[1.,1.],right_contact=[1.,1.],slip_rms=[.01,.02],
        abs_pitch=[.01,.02],soft_joint_limit_margin=[.4,.5])
    row=dict(command=[0.,0.,.38],metrics=metrics,per_env_metrics=per_env,
        failure_count=0,timeout_count=0,survival_fraction=1.)
    assert height_skill_checks(row)['passed']
    per_env['height_mae'][1]=.016
    assert not height_skill_checks(row)['passed']


def test_milestone_probe_uses_stage_height_and_separate_command_ramps():
    argv=probe_command('/tmp/source.pt','/tmp/probe.json',spec()['stage'])
    assert argv[argv.index('--height-delay-seconds')+1]=='2.0'
    assert argv[argv.index('--lean-reference-max-deg')+1]=='2.5'
    heights=argv[argv.index('--height-commands')+1:argv.index('--initial-commands')]
    assert heights==['0.4']*5+['0.38']*6


def test_cornering_height_cohorts_and_reward_gradient():
    s = spec()
    s['stage'].update(phase=3, turn_height=.4, lean_max_deg=8.,
        pairs=[[1.,.5],[2.5,.6],[3.,.7],[3.,1.],[3.5,.8],[4.,1.]],
        height_schedule=[[1.5,.38],[2.5,.36]],
        cohort_plan={'fractions':{'retention':.5,'height':.2,'mid_turn':.25,'high_turn':.05},
            'height_bank':[[0.,.38],[-.5,.38],[.5,.38]],
            'mid_pairs':[[1.,.5],[2.5,.6],[3.,.7]],
            'high_pairs':[[3.,1.],[3.5,.8],[4.,1.]],
            'unclip_base_height':True})
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    manifest = apply_turn_lean_long(cfg,train,spec=s)
    assert cfg.commands.sampling_strategy == 'cornering_height_skill'
    assert cfg.commands.height_slots == (0,2,4,6)
    assert cfg.commands.mid_slots == (8,10,12,14,16)
    assert cfg.commands.high_slots == (18,)
    assert len(cfg.commands.turn_bank) == len(cfg.commands.high_turn_bank) == 12
    assert cfg.rewards.unclipped_reward_names == ('base_height',)
    assert manifest['profile'] == 'cornering_height_skill_v1'
    assert manifest['reference_scope'].endswith('odd env IDs only (50% retention)')
    ids = torch.arange(20)
    mask = ids.remainder(2) != 0
    _, fraction = reference_loss(torch.ones(20,6),torch.zeros(20,6),
        torch.ones(6),ids,2)
    assert fraction == .5 and mask.sum() == 10


def test_adjusted_height_cohort_keeps_half_retention_and_high_boundary():
    s=spec()
    s['stage'].update(phase=3,turn_height=.4,lean_max_deg=8.,
        pairs=[[1.,.5],[2.5,.6],[3.,1.]],
        cohort_plan={'fractions':{'retention':.5,'height':.3,'mid_turn':.15,'high_turn':.05},
            'height_bank':[[0.,.38],[-.5,.38],[.5,.38]],
            'mid_pairs':[[1.,.5],[2.5,.6]],'high_pairs':[[3.,1.]],
            'unclip_base_height':True})
    cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO()
    apply_turn_lean_long(cfg,train,spec=s)
    assert cfg.commands.height_slots==(0,2,4,6,8,10)
    assert cfg.commands.mid_slots==(12,14,16)
    assert cfg.commands.high_slots==(18,)
    assert cfg.commands.turn_slots==(12,14,16,18)


def test_height_skill_public_command_and_recovery():
    scheduler = HeightSkill(20,'cpu',.01,(0,2,4,6),
        entry_range=(1.,1.),hold_range=(2.,2.),ramp_seconds=2.)
    commands = torch.zeros(20,3)
    ids = scheduler.ids
    scheduler.reset(commands,ids,torch.tensor([[0.,.38],[-.5,.38],[.5,.36],[0.,.36]]))
    for _ in range(400):
        scheduler.advance(commands)
    torch.testing.assert_close(commands[ids,2],torch.tensor([.38,.38,.36,.36]))
    for _ in range(400):
        scheduler.advance(commands)
    torch.testing.assert_close(commands[ids,2],torch.full((4,),.4))
    assert commands[2,0] == -.5 and commands[4,0] == .5


def test_cornering_probe_separates_retention_height_mid_high():
    s = spec()['stage']
    s['cohort_plan'] = {'height_bank':[[0.,.38]]}
    argv = probe_command('/tmp/source.pt','/tmp/probe.json',s)
    heights = argv[argv.index('--height-commands')+1:argv.index('--initial-commands')]
    assert len(heights) == 23
    assert heights[:5] == ['0.4']*5
    assert heights[5:11] == ['0.38']*3+['0.36']*3
    assert heights[11:19] == ['0.38']*8
    assert heights[19:] == ['0.36']*4


def test_height_stagnation_requires_three_probe_windows_and_no_gain():
    def probe(error,passed=0):
        return {'height_skill_passed':passed,
                'height_rows':[{'command':[0.,0.,.38],'height_mae':error}]}
    assert not height_skill_stagnant([probe(.019),probe(.019)])
    assert height_skill_stagnant([probe(.019),probe(.0188),probe(.0185)])
    assert not height_skill_stagnant([probe(.019),probe(.017),probe(.015)])
    assert not height_skill_stagnant([probe(.019),probe(.019),probe(.019,1)])


def test_cornering_sampler_assigns_only_nonretained_cohorts():
    s = spec()
    s['stage'].update(phase=3,turn_height=.4,lean_max_deg=8.,
        pairs=[[1.,.5],[2.5,.6],[3.,1.]],
        cohort_plan={'fractions':{'retention':.5,'height':.2,'mid_turn':.25,'high_turn':.05},
            'height_bank':[[0.,.38]], 'mid_pairs':[[1.,.5],[2.5,.6]],
            'high_pairs':[[3.,1.]], 'unclip_base_height':True})
    cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO()
    apply_turn_lean_long(cfg,train,spec=s)
    ids=torch.arange(20)
    commands=torch.zeros(20,3)
    buffers=CommandBuffers(commands,torch.zeros(20,dtype=torch.long),
        torch.zeros(20,dtype=torch.long))
    ranges={key:torch.tensor([[lo,hi]]*20,dtype=torch.float)
            for key,lo,hi in (('lin_vel_x',-4.,4.),('ang_vel_yaw',-4.,4.),('height',.4,.4))}
    seen={'turn':[],'height':[]}
    def reset_turn(out,selected,target):
        seen['turn'].extend(selected.tolist())
        out[selected,:2]=target
    def reset_height(out,selected,target):
        seen['height'].extend(selected.tolist())
        out[selected,0]=target[:,0]
    resample(ids,state=buffers,ranges=ranges,config=cfg.commands,device='cpu',
        reset_start_stop=lambda *args:None,sample_heading=None,
        reset_turn=reset_turn,reset_height_skill=reset_height)
    assert seen['height']==[0,2,4,6]
    assert seen['turn']==[8,10,12,14,16,18]
    assert commands[1::2,2].eq(.4).all()
    assert commands[1::2,1].eq(0).all()
