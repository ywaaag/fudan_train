"""Candidate yaw precheck; passing this alone does not accept a policy."""


def yaw_precheck_passed(raw, expected_policy_sha256):
    if raw['policy_sha256'] != expected_policy_sha256:
        raise RuntimeError('Policy changed')
    result = raw['result']
    measurements = raw['measurements']
    return bool(result['passed'] and result['completed_steps'] == 34000
                and measurements['samples'] == 20000
                and measurements['vx_mae'] <= .05
                and measurements['yaw_mae'] <= .1
                and measurements['height_mae'] <= .03)
