"""Historical low-speed relay acceptance over already-loaded audit reports."""


def low_speed_passed(reports):
    """Preserve signed-speed, contact, height and yaw thresholds, including empty all()."""
    return all(
        x['failure_count'] == 0 and x['timeout_count'] == 0
        and x['metrics']['nonwheel_contact_fraction']['mean'] == 0
        and min(x['metrics'][k]['mean'] for k in ['left_contact', 'right_contact']) >= .99
        and abs(x['metrics']['height_m']['mean'] - .4) < .03
        and x['tracking_vx_mae'] <= (.05 if x['command_vx'] == 0 else .10)
        and x['metrics']['yaw_rad_s']['mean_abs'] <= .10
        for x in reports
    )
