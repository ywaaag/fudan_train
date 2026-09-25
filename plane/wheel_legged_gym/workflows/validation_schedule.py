from concurrent.futures import ThreadPoolExecutor,as_completed

def run_grid(evaluate,on_result,on_skip,workers=1,cancel=lambda:None):
    on_result(evaluate(('zero',0,0)))
    failed=set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for speed in [1,2,4]:
                pending={}
                for axis in ['vx','yaw']:
                    for sign in [1,-1]:
                        tag=f'{axis}_{sign}_{speed}'
                        if (axis,sign) in failed:on_skip(tag);continue
                        case=(tag,sign*speed if axis=='vx' else 0,sign*speed if axis=='yaw' else 0)
                        pending[pool.submit(evaluate,case)]=(axis,sign)
                for future in as_completed(pending):
                    result=future.result();on_result(result)
                    if not result['physical']:failed.add(pending[future])
        except BaseException:
            cancel()
            for future in pending:future.cancel()
            raise

