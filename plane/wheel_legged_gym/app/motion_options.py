"""Legacy motion supervisor command-line arguments; no runtime side effects."""
import argparse
from pathlib import Path


def build_parser(description):
    parser=argparse.ArgumentParser(description)
    parser.add_argument('--max-rounds',type=int,default=40)
    parser.add_argument('--max-stagnant',type=int,default=3)
    parser.add_argument('--training-seed',type=int,default=23)
    parser.add_argument('--command-switch-seconds',type=int,choices=[0,5],default=0)
    parser.add_argument('--freeze-motion-encoder',action='store_true',help='Basic-motion ablation: retain encoder weights and Adam state without encoder updates')
    parser.add_argument('--geometry-weight',type=float,choices=[-.1,-.2],default=-.1)
    parser.add_argument('--start-stop-ramp-seconds',type=float,choices=[.5,1.,2.])
    parser.add_argument('--start-stop-speed',type=float,choices=[1.,2.,3.,4.],default=1.)
    parser.add_argument('--start-stop-fraction',type=float,choices=[0.,.25,.5],default=.5,
                        help='Zero is a matched static control; it does not train transitions')
    parser.add_argument('--dynamic-fixed-lr',type=float,choices=[1e-6])
    parser.add_argument('--dynamic-equivariance',type=float,choices=[.01])
    parser.add_argument('--dynamic-reference-coef',type=float,choices=[1.])
    parser.add_argument('--resume-evaluation-job',type=Path,help='Resume a dead supervisor at its completed-training evaluation boundary')
    parser.add_argument('--speed-envelope',action='store_true',help='Fixed-height practical turning envelope; preserves geometry and straight reverse4')
    parser.add_argument('--basic-motion',action='store_true',help='Prioritize parking, straight +/-4 and pure yaw +/-4; no combined turns')
    parser.add_argument('--turn-curriculum',action='store_true',help='Resume accepted6900, expand combined turns gradually at fixed .40m')
    parser.add_argument('--geometry-symmetry',action='store_true',help='One bounded turn1 geometry ablation from accepted7400; no speed promotion')
    parser.add_argument('--recover-motion',action='store_true',help='Continue geometry model7900 with unchanged recipe; preserve posture while recovering tracking')
    parser.add_argument('--recover-from',help='Explicit reviewed geometry checkpoint; only with --recover-motion')
    parser.add_argument('--high-speed-geometry-floor',type=float,choices=[.03,.035],default=.03,
                        help='Candidate retention only for |vx|>=2; final geometry gate stays .03')
    return parser
