"""Paired noise diagnostic using the common evaluator; no policy updates."""
import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plane'))
import isaacgym
from wheel_legged_gym.scripts import evaluate_policy_comparison as evaluator


def main():
    mode=sys.argv.pop(1)
    if mode not in ['sampled','sampled_noisy']:raise ValueError('Expected sampled or sampled_noisy')
    make=evaluator.task_registry.make_env
    def make_env(*args,**kwargs):
        kwargs['env_cfg'].noise.add_noise=mode=='sampled_noisy'
        return make(*args,**kwargs)
    evaluator.task_registry.make_env=make_env
    load=evaluator.load_policy
    def load_policy(*args,**kwargs):
        model=load(*args,**kwargs)
        def sample(obs,history):
            action=model.act(obs,history)
            return action,model.latent
        model.act_inference=sample
        return model
    evaluator.load_policy=load_policy
    evaluator.main()
    out=Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('--out=')))
    d=json.loads(out.read_text());d.update(diagnostic_only=True,deterministic=False,noise=mode=='sampled_noisy',
        noise_mode=mode,limitation='Noise draws also affect RNG/reset trajectory; same seed does not guarantee identical physical initial states. Read recorded states.')
    out.write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()
