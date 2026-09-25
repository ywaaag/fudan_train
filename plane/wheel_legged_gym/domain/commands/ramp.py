def ramp_fraction(t, settle, duration, shape='linear'):
    if settle<0 or duration<0 or shape not in ('linear','smoothstep'):
        raise ValueError('Invalid command ramp')
    u=min(1.,max(0.,(t-settle)/duration)) if duration else float(t>=settle)
    return u if shape=='linear' else u*u*(3-2*u)

