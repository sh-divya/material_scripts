def get_relaxed_structures(oracle, structs, verbosity):
    rel_str = []
    for s in structs:
        if s is not None:
            try:
                rs = oracle.relax(s, verbosity)
            except:
                rs = None
            rel_str.append(rs)
        else:
            rel_str.append(None)
    return rel_str


def get_delta_target(oracle, structs, true_val):
    pred_vals = []
    tmp_idx = []
    for i, s in enumerate(structs):
        if s is not None:
            pred_vals.append(oracle.predict(s))
        else:
            pred_vals.append(10000)
            tmp_idx.append(i)
    if not true_val:
        true_val = min(pred_vals)
    delta = [(i, abs(val - true_val)) for i, val in enumerate(pred_vals)]
    for i in tmp_idx:
        pred_vals[i] = None
    return delta, pred_vals