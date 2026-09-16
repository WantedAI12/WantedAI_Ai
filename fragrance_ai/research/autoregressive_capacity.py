"""Function-preserving expansion of the learned recurrent optimizer.

The old hidden subspace has zero connections from the new units. Its gates
and output columns are copied exactly. New random hidden units initially
have zero output weights, so expansion itself is not a changed prediction.
Training can subsequently use the added capacity.
"""
import torch


def initialize_from_controller(model, source):
    target = {k:v.clone() for k,v in model.state_dict().items()}
    old = source['cell.weight_hh'].shape[1]
    new = model.cell.hidden_size
    if old>new or set(source)!=set(target):
        raise ValueError('controller expansion cannot discard state or parameters')
    if old==new:
        model.load_state_dict(source)
        return
    with torch.no_grad():
        for name in ('chemistry.weight','chemistry.bias','controls.bias'):
            target[name].copy_(source[name])
        target['controls.weight'].zero_()
        target['controls.weight'][:,:old].copy_(source['controls.weight'])
        for gate in range(3):
            src = slice(gate*old,(gate+1)*old)
            dst = slice(gate*new,gate*new+old)
            for name in ('cell.weight_ih','cell.bias_ih','cell.bias_hh'):
                target[name][dst].copy_(source[name][src])
            target['cell.weight_hh'][dst].zero_()
            target['cell.weight_hh'][dst,:old].copy_(source['cell.weight_hh'][src])
    model.load_state_dict(target)
