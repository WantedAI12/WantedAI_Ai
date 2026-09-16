"""Exact scalar VJP of the shared sampled headspace model for inverse learning.

Unlike the old inverse task, this objective includes liquid partition, Hill
saturation, cross-component suppression and evaporation before profile loss.
It does not materialize the N x N x descriptor Jacobian.
"""

import numpy as np


def profile_loss(predicted, target, time_weights, avoided):
    from .perceptual_objective import perfume_loss
    return perfume_loss(predicted, target, time_weights, avoided)

class NonlinearDoseObjective:
    def __init__(self, ingredients, properties, concentration, draws=16):
        from .dose_refinement import DoseModel

        self.model = DoseModel(ingredients, properties, concentration, draws=draws)
        self.draws = draws

    def predict(self, profiles, weights):
        m = self.model
        weights = np.asarray(weights, float)
        if (
            weights.shape != m.gain.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "finite nonnegative composition with positive total required"
            )
        active = np.flatnonzero(weights > 0)
        denominator = max(1e-30, m.base_moles + weights @ m.total_moles)
        activity = weights * m.coefficients / denominator
        powered = activity**0.55
        response = powered / (1 + powered) * m.transport
        # Exact sparse contraction: zero-dose components produce no inhibition.
        # Every inactive candidate still receives its full entry derivative.
        divisor = 1 + m.suppression[:, None, None] * (
            response[..., active] @ m.interaction[:, active].T
        )
        suppressed = response / divisor
        raw = np.einsum("kti,hid->kthd", suppressed[..., active], profiles[:, active])
        total = raw.sum(-1)
        scaled = np.divide(
            raw, total[..., None], out=np.zeros_like(raw), where=total[..., None] > 0
        )
        nominal_raw = np.einsum("i,hid->hd", weights * m.gain, profiles)
        nominal = np.divide(
            nominal_raw,
            nominal_raw.sum(-1, keepdims=True),
            out=np.zeros_like(nominal_raw),
            where=nominal_raw.sum(-1, keepdims=True) > 0,
        )
        predicted = np.concatenate(
            (nominal[:, None], scaled.mean(0).transpose(1, 0, 2)), axis=1
        )
        return predicted, (
            denominator,
            response,
            divisor,
            suppressed,
            scaled,
            total,
            nominal_raw,
        )

    def __call__(
        self,
        profiles,
        targets,
        responses,
        weights,
        product,
        *,
        time_weights=None,
        avoided=None,
        compute_gradient=True,
        loss_function=None,
        gradient_coordinates='mass',
    ):
        if np.any(np.asarray(product) != 0):
            raise ValueError(
                "headspace objective is for perfume, not inferred lotion transport"
            )
        if gradient_coordinates not in ('mass', 'hill_power'):
            raise ValueError('unknown inverse gradient coordinates')
        losses = []
        gradients = []
        predictions = []
        for b, w in enumerate(np.asarray(weights, float)):
            p = np.asarray(profiles[b], float)
            q = np.asarray(targets[b], float)
            predicted, cache = self.predict(p, w)
            tw = (
                np.r_[0.0, np.ones(predicted.shape[1] - 1)]
                if time_weights is None
                else time_weights[b]
            )
            mask = (
                np.zeros_like(q)
                if avoided is None
                else np.broadcast_to(avoided[b], q.shape)
            )
            loss, partial = (profile_loss if loss_function is None else loss_function)(predicted, q, tw, mask)
            if not compute_gradient:
                losses.append(loss)
                gradients.append(np.zeros_like(w))
                predictions.append(predicted)
                continue
            denominator, response, divisor, suppressed, scaled, total, nominal_raw = (
                cache
            )
            m = self.model
            active = np.flatnonzero(w > 0)
            nominal_total = nominal_raw.sum(-1)
            nominal_partial = partial[:, 0]
            grad = (
                np.einsum(
                    "hd,hid->i",
                    nominal_partial / np.maximum(nominal_total[:, None], 1e-30),
                    p,
                )
                * m.gain
            )
            grad -= (
                np.einsum(
                    "h,hi->i",
                    (nominal_partial * predicted[:, 0]).sum(-1)
                    / np.maximum(nominal_total, 1e-30),
                    p.sum(-1),
                )
                * m.gain
            )
            partial_time = partial[:, 1:].transpose(1, 0, 2) / self.draws
            adjusted = (
                partial_time[None] - (partial_time[None] * scaled).sum(-1)[..., None]
            ) / np.maximum(total[..., None], 1e-30)
            # Contract descriptors/heads with BLAS, not an unoptimized
            # draw*time*head*descriptor*material scalar einsum loop.
            adjoint = (adjusted.reshape(-1, p.shape[0]*p.shape[2]) @
                       p.transpose(0, 2, 1).reshape(-1, p.shape[1])).reshape(
                           adjusted.shape[0], adjusted.shape[1], p.shape[1])
            before = adjoint / divisor - m.suppression[:, None, None] * (
                (
                    adjoint[..., active]
                    * response[..., active]
                    / divisor[..., active] ** 2
                )
                @ m.interaction[active]
            )
            slope = 0.55 * response * (1 - response / m.transport)
            log_gradient = before * slope
            direct = np.divide(
                log_gradient, w, out=np.zeros_like(log_gradient), where=w > 0
            )
            # Hill exponent < 1 has an infinite derivative at exact zero.
            # A labelled one-sided search probe supplies a finite entry
            # direction without adding any signal to the actual prediction.
            probe_activity = 1e-8 * m.coefficients / denominator
            probe_power = probe_activity**0.55
            probe_response = probe_power / (1 + probe_power) * m.transport
            direct = np.where(w > 0, direct, before * probe_response / 1e-8)
            grad += (
                direct.sum((0, 1)) - log_gradient.sum() * m.total_moles / denominator
            )
            if gradient_coordinates == 'hill_power':
                # z = w**alpha removes the infinite mass derivative at zero.
                # The entry derivative is analytic, not a finite-dose probe.
                grad *= (1/0.55)*np.power(w, 1-0.55)
                zero = w == 0
                if np.any(zero):
                    derivative = m.transport[zero]*np.power(m.coefficients[..., zero]/denominator, 0.55)
                    grad[zero] = (before[..., zero]*derivative).sum((0, 1))
            losses.append(loss)
            gradients.append(grad)
            predictions.append(predicted)
        return np.asarray(losses), np.asarray(gradients), np.asarray(predictions)

    def torch(self, device):
        return TorchNonlinearObjective(self, device)

    def signal_value_gradient(self, weights):
        """Mean total receptor response and analytic derivative in w**0.55."""
        w = np.asarray(weights,float)
        m = self.model
        if w.shape!=m.gain.shape or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:
            raise ValueError('finite nonnegative dose with positive mass required')
        denominator = max(1e-30,m.base_moles+w@m.total_moles)
        power = (w*m.coefficients/denominator)**.55
        response = power/(1+power)*m.transport
        divisor = 1+m.suppression[:,None,None]*(response@m.interaction.T)
        signal = (response/divisor).sum(-1).mean(0)
        adjoint = 1/divisor-m.suppression[:,None,None]*((response/divisor**2)@m.interaction)
        log_derivative = adjoint*.55*response*(1-response/m.transport)
        direct = np.divide(log_derivative,w,out=np.zeros_like(log_derivative),where=w>0)
        grad = (direct-log_derivative.sum(-1)[...,None]*m.total_moles/denominator)
        grad *= (1/.55)*w**(1-.55)
        zero = w==0
        if np.any(zero):
            grad[...,zero] = adjoint[...,zero]*m.transport[zero]*(m.coefficients[...,zero]/denominator)**.55
        return signal,grad.mean(0)


class TorchNonlinearObjective:
    """Same equations, differentiable through dose and scalar VJP in training."""

    def __init__(self, source, device):
        import torch

        self.values = {
            name: torch.as_tensor(
                getattr(source.model, name), device=device, dtype=torch.float32
            )
            for name in (
                "coefficients",
                "transport",
                "suppression",
                "interaction",
                "total_moles",
                "gain",
            )
        }
        self.base = source.model.base_moles
        self.draws = source.draws

    def __call__(
        self,
        profiles,
        targets,
        responses,
        weights,
        product,
        *,
        time_weights=None,
        avoided=None,
        compute_gradient=True,
    ):
        import torch

        losses = []
        gradients = []
        predictions = []
        m = self.values
        for b, w in enumerate(weights):
            active = torch.nonzero(w > 0, as_tuple=True)[0]
            p, q = profiles[b].float(), targets[b].float()
            denominator = self.base + w @ m["total_moles"]
            activity = w * m["coefficients"] / denominator
            powered = activity.clamp_min(1e-30) ** 0.55
            response = torch.where(
                activity > 0,
                powered / (1 + powered) * m["transport"],
                torch.zeros_like(powered),
            )
            divisor = 1 + m["suppression"][:, None, None] * (
                response[..., active] @ m["interaction"][:, active].T
            )
            suppressed = response / divisor
            raw = torch.einsum("kti,hid->kthd", suppressed[..., active], p[:, active])
            total = raw.sum(-1)
            scaled = raw / total[..., None].clamp_min(1e-30)
            nominal_raw = torch.einsum("i,hid->hd", w * m["gain"], p)
            nominal = nominal_raw / nominal_raw.sum(-1, keepdim=True).clamp_min(1e-30)
            predicted = torch.cat(
                (nominal[:, None], scaled.mean(0).permute(1, 0, 2)), 1
            )
            norm = predicted.norm(dim=-1)
            qnorm = q.norm(dim=-1)
            cosine = (predicted * q).sum(-1) / (norm * qnorm).clamp_min(1e-30)
            mask = torch.zeros_like(q) if avoided is None else avoided[b].expand_as(q)
            loss_components = torch.stack(
                (
                    0.5 * (predicted - q).abs().sum(-1),
                    1 - cosine,
                    (predicted * mask).sum(-1),
                ),
                -1,
            )
            points, kind = loss_components.max(-1)
            partial = torch.where(
                (kind == 0)[..., None],
                0.5 * (predicted - q).sign(),
                torch.where(
                    (kind == 1)[..., None],
                    cosine[..., None]
                    * predicted
                    / norm[..., None].square().clamp_min(1e-30)
                    - q / (norm * qnorm)[..., None].clamp_min(1e-30),
                    mask,
                ),
            )
            tw = (
                torch.ones(predicted.shape[1], device=w.device)
                if time_weights is None
                else time_weights[b].clone()
            )
            tw[0] = 0
            tw = tw / tw.sum()
            temporal = points @ tw
            head_loss = torch.maximum(points[:, 0], temporal)
            head = head_loss.argmax()
            if not compute_gradient:
                losses.append(head_loss[head])
                gradients.append(torch.zeros_like(w))
                predictions.append(predicted)
                continue
            coeff = torch.nn.functional.one_hot(head, len(head_loss)).to(w.dtype)[
                :, None
            ] * torch.where(
                (points[:, 0] >= temporal)[:, None],
                torch.nn.functional.one_hot(
                    torch.tensor(0, device=w.device), len(tw)
                ).to(w.dtype)[None],
                tw[None],
            )
            partial = partial * coeff[..., None]
            pn = partial[:, 0]
            nt = nominal_raw.sum(-1)
            gradient = (
                torch.einsum("hd,hid->i", pn / nt[:, None].clamp_min(1e-30), p)
                * m["gain"]
            )
            gradient -= (
                torch.einsum(
                    "h,hi->i", (pn * nominal).sum(-1) / nt.clamp_min(1e-30), p.sum(-1)
                )
                * m["gain"]
            )
            pt = partial[:, 1:].permute(1, 0, 2) / self.draws
            adjusted = (pt[None] - (pt[None] * scaled).sum(-1)[..., None]) / total[
                ..., None
            ].clamp_min(1e-30)
            adjoint = torch.einsum("kthd,hid->kti", adjusted, p)
            before = adjoint / divisor - m["suppression"][:, None, None] * (
                (
                    adjoint[..., active]
                    * response[..., active]
                    / divisor[..., active].square()
                )
                @ m["interaction"][active]
            )
            log_gradient = before * (0.55 * response * (1 - response / m["transport"]))
            direct = torch.where(
                w > 0, log_gradient / w.clamp_min(1e-30), torch.zeros_like(log_gradient)
            )
            probe_power = (1e-8 * m["coefficients"] / denominator).clamp_min(
                1e-30
            ) ** 0.55
            probe_response = probe_power / (1 + probe_power) * m["transport"]
            direct = torch.where(w > 0, direct, before * probe_response / 1e-8)
            gradient = (
                gradient
                + direct.sum((0, 1))
                - log_gradient.sum() * m["total_moles"] / denominator
            )
            losses.append(head_loss[head])
            gradients.append(gradient)
            predictions.append(predicted)
        return torch.stack(losses), torch.stack(gradients), torch.stack(predictions)
