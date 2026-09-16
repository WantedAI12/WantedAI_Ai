"""Isolated dose-sensitive PhysSim-Core adaptation over the existing encoder.

This tests a mixture interaction hypothesis, not a new physical force law or
a replacement for measured molecular odor profiles. Dataset dose units must
be declared by the caller; no unknown dilution is relabelled as measured gas.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FrozenAggregation(nn.Module):
    """The current shared backbone after its frozen 256-D molecular encoder."""
    def __init__(self, arrays):
        super().__init__()
        self.names = {}
        prefixes = ('context_in.','attention_key.','attention_query.','fusion.','shared_in.','shared.')
        for name,value in arrays.items():
            if name.startswith(prefixes):
                key = name.replace('.','__')
                self.register_buffer(key,torch.as_tensor(value.copy(),dtype=torch.float32))
                self.names[name] = key

    def value(self,name):
        return getattr(self,self.names[name])

    def linear(self,x,name):
        bias = self.value(name+'.bias') if name+'.bias' in self.names else None
        return F.linear(x,self.value(name+'.weight'),bias)

    def forward(self,embedded,amounts,context):
        total = amounts.sum(-1,keepdim=True)
        weights = amounts/total.clamp_min(1e-30)
        mean = (weights[...,None]*embedded).sum(1)
        variance = (weights[...,None]*(embedded-mean[:,None])**2).sum(1)
        ctx = F.relu(self.linear(context,'context_in'))
        logits = (self.linear(embedded,'attention_key')*self.linear(ctx,'attention_query')[:,None]).sum(-1)/8
        logits = (logits+amounts.clamp_min(1e-30).log()).masked_fill(amounts<=0,-1e30)
        attention = logits.softmax(-1)*(amounts>0)
        attention = attention/attention.sum(-1,keepdim=True).clamp_min(1e-30)
        attended = (attention[...,None]*embedded).sum(1)
        process = torch.zeros((len(embedded),96),device=embedded.device,dtype=embedded.dtype)
        h = F.relu(self.linear(F.relu(self.linear(torch.cat((mean,variance,attended,ctx,process),-1),'fusion')),'shared_in'))
        for i in range(2):
            name = f'shared.{i}'
            normalized = F.layer_norm(h,(256,),self.value(name+'.norm.weight'),self.value(name+'.norm.bias'),1e-5)
            h = h+self.linear(F.relu(self.linear(normalized,name+'.up')),name+'.down')
        return h,mean,variance


class DoseLatentDynamics(nn.Module):
    """Weighted radial interactions; learned variables remain dimensionless.

    Row splitting and ordering leave the result invariant. A zero-dose row
    exerts no interaction and has zero pooling weight. Absolute amount is a
    separate input to the interactions and output, not normalized away.
    """
    def __init__(self,width=128,steps=8):
        super().__init__()
        if width<2 or not 1<=steps<=32:
            raise ValueError('bounded positive latent width and relaxation steps required')
        self.width,self.steps = width,steps
        self.encoder = nn.Sequential(nn.LayerNorm(256),nn.Linear(256,width),nn.GELU(),nn.LayerNorm(width))
        self.position = nn.Linear(width,width)
        self.velocity = nn.Linear(width,width)
        self.mass = nn.Linear(width,1)
        self.charge = nn.Linear(width,1)
        self.radius = nn.Linear(width,1)
        self.log_constants = nn.Parameter(torch.zeros(5))
        self.output = nn.Sequential(nn.Linear(width+6,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,256))

    def advance(self,position,velocity,mass,activity,charge,radius,constants):
        attraction,coupling,well,speed_limit,decay = constants
        dt = .1/self.steps
        delta = .5
        difference = position[:,:,None]-position[:,None,:]
        distance = (difference.square().sum(-1,keepdim=True)+delta**2).sqrt()
        direction = difference/distance
        sigma = (radius[:,:,None]+radius[:,None,:])*.5
        power = (sigma/distance).pow(6)
        radial = (-attraction*mass[:,:,None]*mass[:,None,:]+coupling*charge[:,:,None]*charge[:,None,:])/distance.square()
        radial = radial+24*well/distance*(2*power.square()-power)
        acceleration = (activity[:,None,:,None]*radial*direction).sum(2)/mass
        trial = velocity+dt*acceleration
        norm = trial.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        velocity = speed_limit*torch.tanh(norm/speed_limit)*trial/norm
        position = position+dt*velocity
        mass = mass*torch.exp(-decay*dt/(mass.square()+delta**2))
        return position,velocity,mass

    def forward(self,embedded,amounts):
        if embedded.ndim!=3 or amounts.shape!=embedded.shape[:2]:
            raise ValueError('batched molecular embeddings and matching nominal amounts required')
        total = amounts.sum(-1,keepdim=True)
        if torch.any(amounts<0) or not torch.isfinite(amounts).all() or torch.any(total<=0):
            raise ValueError('finite nonnegative amounts with positive total required')
        relative = amounts/total
        # A declared nominal amount of one is the comparison's reference unit.
        # This bounded activity is a model feature, not an odor threshold law.
        activity = amounts/(1+total)
        h = self.encoder(embedded)
        position = .25*torch.tanh(self.position(h))
        initial = position
        velocity = .1*torch.tanh(self.velocity(h))
        mass = F.softplus(self.mass(h))+.1
        initial_mass = mass
        charge = torch.tanh(self.charge(h))
        radius = .1+.4*torch.sigmoid(self.radius(h))
        constants = F.softplus(self.log_constants)+.01
        displacement = torch.zeros_like(mass)
        for _ in range(self.steps):
            if self.training and embedded.device.type=='cuda' and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                position,velocity,mass = checkpoint(self.advance,position,velocity,mass,activity,charge,radius,constants,use_reentrant=False)
            else:
                position,velocity,mass = self.advance(position,velocity,mass,activity,charge,radius,constants)
            displacement = displacement+(position-initial).square().sum(-1,keepdim=True)/self.steps
        vector = torch.cat((position,displacement,velocity.square().sum(-1,keepdim=True),mass,
            mass/initial_mass,charge.abs(),total.log1p()[:,None].expand(-1,amounts.shape[1],-1)),-1)
        return self.output((relative[...,None]*vector).sum(1))


class PairComparison(nn.Module):
    MODES = ('current','capacity_control','physmix')
    def __init__(self,arrays,mode,*,steps=8):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError('unknown comparison arm')
        self.mode = mode
        self.backbone = FrozenAggregation(arrays)
        if mode=='physmix':
            self.interaction = DoseLatentDynamics(128,steps)
        elif mode=='capacity_control':
            reference = DoseLatentDynamics(128,steps)
            parameters = sum(p.numel() for p in reference.parameters())
            width = max(1,round((parameters-256)/(512+256+1)))
            self.interaction = nn.Sequential(nn.Linear(512,width),nn.GELU(),nn.Linear(width,256))
        else:
            self.interaction = None
        if self.interaction is not None:
            self.gate = nn.Parameter(torch.zeros(()))
        self.head = nn.Sequential(nn.Linear(515,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))

    def encode(self,embedded,amounts,context):
        if (embedded.ndim!=3 or embedded.shape[-1]!=256 or amounts.shape!=embedded.shape[:2]
                or context.shape!=(len(embedded),64) or torch.any(amounts<0)
                or any(not torch.isfinite(x).all() for x in (embedded,amounts,context))
                or torch.any(amounts.sum(-1)<=0) or not torch.isfinite(amounts.sum(-1)).all()):
            raise ValueError('finite complete molecular mixtures and context required')
        h,mean,variance = self.backbone(embedded,amounts,context)
        if self.interaction is not None:
            extra = self.interaction(embedded,amounts) if self.mode=='physmix' else self.interaction(torch.cat((mean,variance),-1))
            h = h+.25*torch.tanh(self.gate)*extra
        return h,amounts.sum(-1).log1p()

    def forward(self,a,wa,b,wb,context):
        first,ca = self.encode(a,wa,context)
        second,cb = self.encode(b,wb,context)
        cosine = F.cosine_similarity(first,second)[:,None]
        features = torch.cat(((first-second).abs(),first*second,cosine,(ca-cb).abs()[:,None],(ca*cb)[:,None]),-1)
        return torch.sigmoid(self.head(features).squeeze(-1))


def parameter_counts(model):
    return {'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_buffer_values':sum(b.numel() for b in model.buffers()),
        'interaction':sum(p.numel() for p in model.interaction.parameters()) if model.interaction is not None else 0}
