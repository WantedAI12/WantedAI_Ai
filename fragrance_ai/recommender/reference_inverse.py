"""Source-fixed inverse recipes: fractional LP seeds and smooth physical polish.

Search smoothing never changes the reported loss or the acceptance threshold.
All eligible material columns enter the nominal LP; finite support is only an
explicit request constraint, not a permanent vocabulary/material shortlist.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize

from .nonlinear_inverse import NonlinearDoseObjective, profile_loss
from .search_budget import ACTIVE, allowance, exhausted

VERSION = 'source-fixed-physical-inverse/v81'


def point_value_gradient(engine, profiles, target, avoided, weights, *, kind='profile',
                         epsilon=1e-7, tau=1e-5):
    """Per-head/per-time constraint VJPs in finite Hill coordinates.

    The derivative is of the loss on the MEAN sampled profile, not the mean
    of individual-draw losses. Heads and time windows remain separate.
    Complexity avoids a material x material x descriptor Jacobian.
    """
    p,q,w = map(lambda x:np.asarray(x,float),(profiles,target,weights))
    pred,cache = engine.predict(p,w)
    mask = np.broadcast_to(avoided,q.shape)
    if kind=='avoidance':
        values,partial = (pred*mask).sum(-1),mask
    elif kind=='profile':
        difference = pred-q
        absolute = np.sqrt(difference**2+epsilon**2)
        norm,qnorm = np.linalg.norm(pred,axis=-1),np.linalg.norm(q,axis=-1)
        cosine = (pred*q).sum(-1)/np.maximum(norm*qnorm,1e-30)
        terms = np.stack((.5*absolute.sum(-1),1-cosine,(pred*mask).sum(-1)),-1)
        values,coefficients = _softmax(terms,tau)
        partial = coefficients[...,0,None]*(.5*difference/absolute)
        partial += coefficients[...,1,None]*(cosine[...,None]*pred/np.maximum(norm[...,None]**2,1e-30)
            -q/np.maximum((norm*qnorm)[...,None],1e-30))
        partial += coefficients[...,2,None]*mask
    else:
        raise ValueError('unknown pointwise inverse constraint')
    denominator,response,divisor,_,scaled,total,nominal_raw = cache
    m = engine.model
    gradient = np.zeros((*values.shape,len(w)))
    gradient[:,0] = np.einsum('hd,hid->hi',partial[:,0],p-pred[:,0,None,:])
    gradient[:,0] *= m.gain/np.maximum(nominal_raw.sum(-1),1e-30)[:,None]
    gradient[:,0] *= (1/.55)*w**.45
    slope = .55*response*(1-response/m.transport)
    for h in range(len(p)):
        g = partial[h,1:]
        adjusted = (g[None]-(g[None]*scaled[:,:,h]).sum(-1)[...,None])/np.maximum(total[:,:,h,None],1e-30)
        adjoint = adjusted@p[h].T
        before = adjoint/divisor-m.suppression[:,None,None]*((adjoint*response/divisor**2)@m.interaction)
        log_derivative = before*slope
        direct = np.divide(log_derivative,w,out=np.zeros_like(log_derivative),where=w>0)
        jac = (direct-log_derivative.sum(-1)[...,None]*m.total_moles/denominator)*(1/.55)*w**.45
        zero = w==0
        if np.any(zero):
            jac[...,zero] = before[...,zero]*m.transport[zero]*(m.coefficients[...,zero]/denominator)**.55
        gradient[h,1:] = jac.mean(0)
    return values,gradient


def quantize_recipe(weights, lower, upper, prices, budget, *, bands=()):
    """Joint 4-decimal-percent rounding; preserve mass, caps and request rows.

    Public recipes use 0.0001 percent increments (= 1e-6 mass fractions).
    Independent rounding can create zero-dose rows or exceed a stock cap.
    Binary floor/ceil choices minimize rounding error under the original rows.
    """
    w,lo,hi,cost = map(lambda x:np.asarray(x,float), (weights,lower,upper,prices))
    units = 1_000_000
    base = np.floor(w*units+1e-8)
    fraction = w*units-base
    low = np.maximum(0.,np.ceil(lo*units-1e-8)-base)
    high = np.minimum(1.,np.floor(hi*units+1e-8)-base)
    if np.any(low>high):
        return None
    rows = [np.ones(len(w)),cost]
    bottom = [units-base.sum(),-np.inf]
    top = [units-base.sum(),budget*units-cost@base]
    for v,a,b in bands:
        rows.append(v)
        bottom.append(a*units-np.asarray(v)@base-1e-8)
        top.append(b*units-np.asarray(v)@base+1e-8)
    solved = milp(1-2*fraction,integrality=np.ones(len(w)),bounds=Bounds(low,high),
        constraints=LinearConstraint(np.asarray(rows),bottom,top),
        options={'time_limit':allowance('quantization',2.),'mip_rel_gap':0.})
    if solved.x is None:
        return None
    result = (base+np.rint(solved.x))/units
    if (abs(result.sum()-1)>1e-12 or np.any(result<lo-1e-10) or np.any(result>hi+1e-10)
            or cost@result>budget+1e-7 or
            any(not a-1e-8<=np.asarray(v)@result<=b+1e-8 for v,a,b in bands)):
        return None
    return result


def _softmax(values, tau, axis=-1):
    maximum = np.max(values, axis=axis, keepdims=True)
    exp = np.exp((values-maximum)/tau)
    norm = exp.sum(axis=axis, keepdims=True)
    return np.squeeze(maximum+tau*np.log(norm), axis=axis), exp/norm


def smooth_profile_loss(predicted, target, time_weights, avoided, *, epsilon=1e-5, tau=.002,
                        avoid_ceiling=None, point_ceiling=None):
    """Smooth upper approximations for optimization; exact loss stays separate."""
    diff = predicted-target
    absolute = np.sqrt(diff*diff+epsilon*epsilon)
    norm = np.linalg.norm(predicted,axis=-1)
    qnorm = np.linalg.norm(target,axis=-1)
    cosine = (predicted*target).sum(-1)/np.maximum(norm*qnorm,1e-30)
    parts = np.stack((.5*absolute.sum(-1),1-cosine,(predicted*avoided).sum(-1)),-1)
    losses, coefficients = _softmax(parts,tau)
    partial = coefficients[...,0,None]*(.5*diff/absolute)
    partial += coefficients[...,1,None]*(cosine[...,None]*predicted/np.maximum(norm[...,None]**2,1e-30)
        -target/np.maximum((norm*qnorm)[...,None],1e-30))
    partial += coefficients[...,2,None]*avoided
    weights = np.array(time_weights,float,copy=True)
    weights[0] = 0
    if not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('positive finite temporal weights required')
    weights /= weights.sum()
    heads, head_mix = _softmax(np.stack((losses[:,0], losses@weights),-1),tau)
    total, source_mix = _softmax(heads,tau)
    temporal_mix = head_mix[:,1,None]*weights[None]
    temporal_mix[:,0] += head_mix[:,0]
    derivative = partial*(source_mix[:,None]*temporal_mix)[...,None]
    if avoid_ceiling is not None:
        # Existing exclusion non-regression is a search condition, not a
        # penalty subtracted from the public similarity score.
        excess = np.maximum((predicted*avoided).sum(-1)-avoid_ceiling,0.)
        total += 30*excess.sum()
        derivative += 30*(excess>0)[...,None]*avoided
    if point_ceiling is not None:
        excess = np.maximum(losses-point_ceiling,0.)
        total += 30*excess.sum()
        derivative += 30*(excess>0)[...,None]*partial
    return float(total), derivative


def nominal_seed(profiles, target, gains, lower, upper, prices, budget, *, bands=()):
    """Charnes-Cooper mass coordinates; all stock caps/costs stay linear.

    v = w/(gain @ w), scale=sum(v), gain@v=1, w=v/scale.
    The objective is the worst-head nominal total variation. It is a seed,
    NOT a certificate that temporal/cosine constraints or 95 points are met.
    """
    p,q = np.asarray(profiles,float),np.asarray(target,float)
    gain,lo,hi,cost = [np.asarray(v,float) for v in (gains,lower,upper,prices)]
    if p.ndim!=3 or q.shape!=(p.shape[0],p.shape[2]):
        raise ValueError('head/material/descriptor profile coordinates required')
    h,n,d = p.shape
    if (any(x.shape!=(n,) for x in (gain,lo,hi,cost)) or
        not all(np.isfinite(x).all() for x in (p,q,gain,lo,hi,cost)) or
        np.any(gain<0) or not np.any(gain>0) or np.any(lo<0) or np.any(hi<lo) or
        np.any(p<0) or np.any(q<0) or not np.isfinite(budget) or budget<=0):
        raise ValueError('invalid nominal inverse constraints')
    if not np.allclose(p[:,hi>0].sum(-1),1,atol=1e-7) or not np.allclose(q.sum(-1),1,atol=1e-7):
        raise ValueError('normalized full profiles required')
    size = n+2+h*d
    scale,t = n,n+1
    matrix = sparse.csr_matrix((p*gain[None,:,None]).transpose(0,2,1).reshape(h*d,n))
    pad = sparse.csr_matrix((h*d,2))
    eye = sparse.eye(h*d,format='csr')
    rows = [sparse.hstack((matrix,pad,-eye)), sparse.hstack((-matrix,pad,-eye))]
    rhs = [q.ravel(),-q.ravel()]
    head_sum = sparse.kron(sparse.eye(h),np.ones((1,d)),format='csr')
    prefix = sparse.csr_matrix((np.full(h,-2.),(np.arange(h),np.full(h,t))),shape=(h,n+2))
    rows.append(sparse.hstack((prefix,head_sum)))
    rhs.append(np.zeros(h))
    cap = sparse.hstack((sparse.eye(n),-hi[:,None],sparse.csr_matrix((n,1+h*d))))
    rows.append(cap)
    rhs.append(np.zeros(n))
    required = np.flatnonzero(lo>0)
    if len(required):
        low = sparse.hstack((-sparse.eye(n,format='csr')[required],lo[required,None],sparse.csr_matrix((len(required),1+h*d))))
        rows.append(low)
        rhs.append(np.zeros(len(required)))
    price = np.zeros(size)
    price[:n],price[scale] = cost,-budget
    rows.append(sparse.csr_matrix(price[None]))
    rhs.append(np.zeros(1))
    for values,minimum,maximum in bands:
        values = np.asarray(values,float)
        if values.shape!=(n,):
            raise ValueError('linear request band has wrong material coordinates')
        row = np.zeros(size)
        row[:n],row[scale] = values,-maximum
        rows.append(sparse.csr_matrix(row[None]))
        rhs.append(np.zeros(1))
        row = np.zeros(size)
        row[:n],row[scale] = -values,minimum
        rows.append(sparse.csr_matrix(row[None]))
        rhs.append(np.zeros(1))
    eq = np.zeros((2,size))
    eq[0,:n],eq[1,:n],eq[1,scale] = gain,1.,-1.
    objective = np.zeros(size)
    objective[t] = 1.
    result = linprog(objective,A_ub=sparse.vstack(rows,format='csr'),b_ub=np.concatenate(rhs),
        A_eq=sparse.csr_matrix(eq),b_eq=[1.,0.],bounds=[(0.,None)]*size,method='highs',
        options={'time_limit':allowance('nominal_seed',8.,n),'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    report = {'version':VERSION,'status':int(result.status),'all_material_columns':n,
        'source_targets_unchanged':True,'not_final_physical_score':True}
    if not result.success or result.x is None or result.x[scale]<=0:
        return None,report
    w = result.x[:n]/result.x[scale]
    w = np.maximum(w,0.)
    w /= w.sum()
    if (np.any(w<lo-1e-8) or np.any(w>hi+1e-8) or w@cost>budget+1e-6 or
            any(not minimum-1e-8<=np.asarray(values)@w<=maximum+1e-8 for values,minimum,maximum in bands)):
        return None,{**report,'status':'numerical_constraint_residual'}
    prediction = np.einsum('i,hid->hd',w*gain,p)/(w@gain)
    report.update(nominal_tv_seed_score=float(100*(1-.5*np.abs(prediction-q).sum(-1).max())),
        positive_materials=int((w>1e-10).sum()))
    return w,report


def polish(ingredients, properties, concentration, profiles, targets, time_weights, avoided,
           initial, lower, upper, prices, budget, *, draws=200, bands=(), maxiter=35, avoid_ceiling=None,
           signal_floor=None, point_ceiling=None, persistence_ratio=None):
    """Finite-gradient Hill coordinates and exact-feasible incumbent retention."""
    engine = NonlinearDoseObjective(ingredients,properties,concentration,draws=draws)
    return polish_prepared(engine, profiles, targets, time_weights, avoided, initial,
        lower, upper, prices, budget, bands=bands, maxiter=maxiter, avoid_ceiling=avoid_ceiling,
        signal_floor=signal_floor, point_ceiling=point_ceiling, persistence_ratio=persistence_ratio)


def polish_prepared(engine, profiles, targets, time_weights, avoided, initial,
                    lower, upper, prices, budget, *, bands=(), maxiter=35, avoid_ceiling=None,
                    signal_floor=None, point_ceiling=None, persistence_ratio=None):
    """Reuse immutable sampled physics; final acceptance uses its exact forward."""
    p,q = np.asarray(profiles,float),np.asarray(targets,float)
    w,lo,hi,cost = [np.asarray(x,float).copy() for x in (initial,lower,upper,prices)]
    alpha = .55
    def feasible(x):
        return (np.isfinite(x).all() and abs(x.sum()-1)<1e-7 and np.all(x>=lo-1e-9)
            and np.all(x<=hi+1e-9) and x@cost<=budget+1e-6
            and all(minimum-1e-8<=np.asarray(v)@x<=maximum+1e-8 for v,minimum,maximum in bands))
    repaired = False
    if not feasible(w) and abs(w.sum()-1)<=1.01e-5:
        # Public percentages are rounded to four decimals. A tiny mass
        # residual must not disable the incumbent's refinement entirely.
        from .mass_projection import repair_mass_residual
        fixed = repair_mass_residual(w[None],lo[None],hi[None])[0]
        if feasible(fixed):
            w,repaired = fixed,True
    if not feasible(w):
        return None,{'status':'infeasible_polish_seed','version':VERSION}
    base,_ = engine.predict(p,w)
    starting = 100*(1-profile_loss(base,q,time_weights,avoided)[0])
    evaluations = 0
    def allowed(pred,x):
        if avoid_ceiling is not None and np.any((pred*avoided).sum(-1)>avoid_ceiling+1e-6):
            return False
        if point_ceiling is not None and np.any(point_losses(pred,q,avoided)>point_ceiling+1e-8):
            return False
        if signal_floor is not None:
            signal,_ = engine.signal_value_gradient(x)
            if np.any(signal<signal_floor*(1-1e-9)):
                return False
            if persistence_ratio is not None and signal[-1]<persistence_ratio*signal[0]*(1-1e-9):
                return False
        return True
    # A high-scoring but constraint-breaking LP seed must not dominate lower-
    # scoring feasible iterates. The service's incumbent is preserved outside.
    best,score = (w.copy(),starting) if allowed(base,w) else (None,-np.inf)
    def derivative(z):
        return (1/alpha)*np.maximum(z,0.)**(1/alpha-1)
    def weights(z):
        return np.maximum(z,0.)**(1/alpha)
    conditions = [{'type':'eq','fun':lambda z:weights(z).sum()-1,'jac':derivative},
        {'type':'ineq','fun':lambda z:budget-cost@weights(z),'jac':lambda z:-cost*derivative(z)}]
    if signal_floor is not None:
        cached_key,cached_signal = None,None
        def signal_constraint(z):
            nonlocal cached_key,cached_signal
            key = z.tobytes()
            if key!=cached_key:
                s,g = engine.signal_value_gradient(weights(z))
                required = np.asarray(signal_floor)>0
                floor = np.maximum(signal_floor,1e-30)
                values,gradient = s[required]/floor[required]-1,g[required]/floor[required,None]
                if persistence_ratio is not None:
                    scale = max(float(floor[-1]),1e-30)
                    values = np.r_[values,(s[-1]-persistence_ratio*s[0])/scale]
                    gradient = np.vstack((gradient,(g[-1]-persistence_ratio*g[0])/scale))
                cached_key,cached_signal = key,(values,gradient)
            return cached_signal
        conditions.append({'type':'ineq','fun':lambda z:signal_constraint(z)[0],
            'jac':lambda z:signal_constraint(z)[1]})
    def add_profile_constraint(kind,ceiling,keep,tolerance):
        cached_key,cached = None,None
        def value_gradient(z):
            nonlocal cached_key,cached
            key = z.tobytes()
            if key!=cached_key:
                values,jac = point_value_gradient(engine,p,q,avoided,weights(z),kind=kind)
                cached = ((ceiling-values+tolerance)[keep],-jac[keep])
                cached_key = key
            return cached
        conditions.append({'type':'ineq','fun':lambda z:value_gradient(z)[0],
            'jac':lambda z:value_gradient(z)[1]})
    if point_ceiling is not None:
        keep = np.asarray(point_ceiling)<1
        if np.any(keep):
            add_profile_constraint('profile',point_ceiling,keep,1e-8)
    if avoid_ceiling is not None:
        keep = np.broadcast_to(np.any(avoided,axis=-1),q.shape[:2])
        if np.any(keep):
            add_profile_constraint('avoidance',avoid_ceiling,keep,1e-6)
    for values,minimum,maximum in bands:
        values = np.asarray(values,float)
        conditions.extend([
            {'type':'ineq','fun':lambda z,v=values,m=minimum:v@weights(z)-m,
             'jac':lambda z,v=values:v*derivative(z)},
            {'type':'ineq','fun':lambda z,v=values,m=maximum:m-v@weights(z),
             'jac':lambda z,v=values:-v*derivative(z)}])
    outcomes = []
    z = w**alpha
    from time import monotonic
    started = monotonic()
    seconds = allowance('physical_polish',float('inf'),len(w))
    class WorkLimit(Exception):
        pass
    def callback(value):
        if exhausted() or monotonic()-started>=seconds:
            raise WorkLimit()
    for epsilon,tau in ((1e-4,.005),(1e-5,.001)):
        if exhausted() or monotonic()-started>=seconds:
            break
        def loss(predicted,target,tw,mask):
            # Preservation is an explicit analytic constraint, not an
            # arbitrary large penalty competing with the actual objective.
            return smooth_profile_loss(predicted,target,tw,mask,epsilon=epsilon,tau=tau)
        def fun(value):
            nonlocal best,score,evaluations
            current = weights(value)
            losses,grads,predictions = engine(p[None],q[None],None,current[None],[0],
                time_weights=np.asarray(time_weights)[None],avoided=np.asarray(avoided)[None],
                loss_function=loss,gradient_coordinates='hill_power')
            evaluations += 1
            exact = 100*(1-profile_loss(predictions[0],q,time_weights,avoided)[0])
            if feasible(current) and exact>score+1e-7 and allowed(predictions[0],current):
                normalized = current/current.sum()
                if feasible(normalized):
                    pred,_ = engine.predict(p,normalized)
                    verified = 100*(1-profile_loss(pred,q,time_weights,avoided)[0])
                    if allowed(pred,normalized) and verified>score+1e-7:
                        best,score = normalized,verified
            return float(losses[0]),grads[0]
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=list(zip(lo**alpha,hi**alpha)),
                constraints=conditions,callback=callback,options={'maxiter':maxiter,'ftol':1e-9,'disp':False})
        except WorkLimit:
            outcomes.append({'status':'work_budget_exhausted','success':False})
            break
        outcomes.append({'status':int(result.status),'iterations':int(result.nit),'success':bool(result.success)})
        z = (best if best is not None else weights(result.x))**alpha
    return best,{'version':VERSION,'starting_exact_score':float(starting),'selected_exact_score':float(score) if best is not None else None,
        'evaluations':evaluations,'stages':outcomes,'gradient_coordinates':'hill_power',
        'pointwise_preservation_in_solver':True,'roundoff_mass_repaired':repaired,
        'score_smoothing_used_for_search_only':True,'final_objective_unchanged':True}


def point_losses(predicted,target,avoided):
    cosine = (predicted*target).sum(-1)/np.maximum(np.linalg.norm(predicted,axis=-1)*np.linalg.norm(target,axis=-1),1e-30)
    return np.maximum(np.maximum(.5*np.abs(predicted-target).sum(-1),1-cosine),(predicted*avoided).sum(-1))


def source_fixed_proposals(items, brief, properties, original, minimums, policy, guidance, diagnostics):
    """Replace non-required support as well as adjusting existing ingredient doses.

    Only named required ingredients remain locked. Every screened, reference-
    covered material is a column in the first LP. A restricted seed is never
    reported as proof that the entire catalogue is infeasible.
    """
    from .adaptive_pyramid import DIFFUSION
    p = np.stack([guidance.shapes.shape(item) for item in items],axis=1)
    gain = np.asarray([item.odor_impact*item.active_strength_percent/100 for item in items])
    w = np.asarray([original.get(item.ingredient_id,0.)/100 for item in items])
    lo = np.asarray([minimums.get(item.ingredient_id,0.)/100 for item in items])
    hi = np.asarray([min(1.,item.as_supplied_cap_percent()/100) for item in items])
    price = np.asarray([item.price_per_kg for item in items])
    budget = brief.constraints.max_formula_cost_per_kg
    bands = []
    for note,value in policy['explicit_percentages'].items():
        bands.append((np.asarray([float(i.pyramid==note) for i in items]),value/100,value/100))
    interval = policy['intensity_range']
    if interval is not None:
        bands.append((gain,2.5*interval[0],2.5*interval[1] if interval[1]<1 else gain.max()))
    if policy['diffusion_range'] is not None:
        bands.append((np.asarray([DIFFUSION[i.pyramid] for i in items]),*policy['diffusion_range']))
    active = np.flatnonzero(w>0)
    base_engine = NonlinearDoseObjective([items[i] for i in active],properties,
        brief.constraints.product_concentration_percent,draws=brief.constraints.simulation_draws)
    base,_ = base_engine.predict(p[:,active],w[active])
    avoid_ceiling = (base*guidance.mask).sum(-1)
    baseline_signal,_ = base_engine.signal_value_gradient(w[active])
    signal_floor = np.where(guidance.tw[1:]>0,baseline_signal*policy['signal_retention_floor'],0.)
    persistence_ratio = None
    if policy['long_lasting']:
        signal_floor[-1] = baseline_signal[-1]
        persistence_ratio = baseline_signal[-1]/baseline_signal[0]
    base_loss = point_losses(base,guidance.q,guidance.mask)
    positive = guidance.tw>0
    point_ceiling = np.full(base_loss.shape,1.)
    point_ceiling[:,positive] = base_loss[:,positive].max()
    for j,row in enumerate(guidance.intent['targets']):
        if positive[j] and row.get('phase') in brief.phase_target_profiles:
            point_ceiling[:,j] = base_loss[:,j].max()
    # Non-increasing nominal avoided odor remains linear after fractional
    # normalization. Temporal avoidance is checked with the actual forward model.
    for h in range(p.shape[0]):
        values = gain*((p[h]*guidance.mask[0]).sum(-1)-avoid_ceiling[h,0])
        if np.any(guidance.mask[0]):
            bands.append((values,float(values.min()),0.))
    report = {'version':VERSION,'all_candidate_columns':len(items),'support_seeds':[],
        'polishes':[],'source_targets_unchanged':True,'final_objective_unchanged':True}
    diagnostics['source_fixed_inverse'] = report
    full,seed_report = nominal_seed(p,guidance.q[:,0],gain,lo,hi,price,budget,bands=bands)
    report['nominal_all_pool'] = seed_report
    seeds = [('incumbent',active,w[active])]
    if full is not None:
        support = np.flatnonzero(full>1e-10)
        limit = brief.constraints.max_ingredients
        if len(support)<=limit:
            seeds.append(('all_pool_nominal',support,full[support]/full[support].sum()))
        else:
            required = set(np.flatnonzero(lo>0))
            for name,priority in (('mass',full),('odor_contribution',full*gain)):
                selected = set(required)
                for i in np.argsort(-priority,kind='stable'):
                    if len(selected)>=limit:
                        break
                    if hi[i]>0:
                        selected.add(int(i))
                ix = np.asarray(sorted(selected),int)
                restricted,status = nominal_seed(p[:,ix],guidance.q[:,0],gain[ix],lo[ix],hi[ix],price[ix],
                    budget,bands=[(v[ix],a,b) for v,a,b in bands])
                report['support_seeds'].append({'ranking':name,'materials':len(ix),**status,
                    'restricted_support_not_full_pool_certificate':True})
                if restricted is not None:
                    seeds.append(('replacement_'+name,ix,restricted))
    seen = set()
    for name,ix,initial in seeds:
        if exhausted():
            report['request_budget_exhausted'] = True
            break
        key = tuple(zip(ix,np.round(initial,12)))
        if key in seen:
            continue
        seen.add(key)
        # Stock minima and user note proportions, price and exclusions are
        # optimized in their original as-supplied mass units.
        candidate,detail = polish([items[i] for i in ix],properties,
            brief.constraints.product_concentration_percent,p[:,ix],guidance.q,guidance.tw,guidance.mask,
            initial,lo[ix],hi[ix],price[ix],budget,draws=brief.constraints.simulation_draws,
            bands=[(v[ix],a,b) for v,a,b in bands],avoid_ceiling=avoid_ceiling,maxiter=80 if ACTIVE.get() else 28,
            signal_floor=signal_floor,point_ceiling=point_ceiling,persistence_ratio=persistence_ratio)
        report['polishes'].append({'seed':name,'materials':len(ix),**detail})
        if candidate is not None:
            candidate = quantize_recipe(candidate,lo[ix],hi[ix],price[ix],budget,
                bands=[(v[ix],a,b) for v,a,b in bands])
            if candidate is None:
                report['polishes'][-1]['output_quantization'] = 'no_feasible_floor_ceil_rounding'
                continue
            check = NonlinearDoseObjective([items[i] for i in ix],properties,
                brief.constraints.product_concentration_percent,draws=brief.constraints.simulation_draws)
            pred,_ = check.predict(p[:,ix],candidate)
            score = 100*(1-profile_loss(pred,guidance.q,guidance.tw,guidance.mask)[0])
            report['polishes'][-1].update(output_quantization='mass_caps_and_request_bands_preserved',
                quantized_exact_score=float(score))
            yield {'weights_percent':{items[i].ingredient_id:float(v*100) for i,v in zip(ix,candidate) if v>1e-10},
                'anchor_weights_percent':original,'replacement_mode':VERSION+'_'+name,
                'allocation_mode':'inferred_full_range','full_pool_considered':len(items),
                'proposal_profile_score':float(score),
                'target_sha256':guidance.intent['sha256'],'final_objective_unchanged':True}
