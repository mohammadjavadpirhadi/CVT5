import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss


class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, outputs, targets):
        p = torch.sigmoid(outputs)
        return torch.mean(torch.mean(targets * -torch.log(p) + (1 - targets) * -torch.log(1 - p)))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        return sigmoid_focal_loss(outputs, targets, self.alpha, self.gamma, self.reduction)


class LossWithOneClassPenalty(nn.Module):
    def __init__(self, base_loss_fn, penalty_weight, l1_reduction):
        super(LossWithOneClassPenalty, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.penalty_weight = penalty_weight
        self.l1_loss_fn = nn.L1Loss(reduction=l1_reduction)

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        base_loss = self.base_loss_fn(outputs, targets)
        sum_probabilities = torch.sum(outputs, dim=-1)
        desired_sum = torch.ones_like(sum_probabilities)
        penalty_loss = self.l1_loss_fn(sum_probabilities, desired_sum)
        return base_loss + self.penalty_weight * penalty_loss
    
class WeightedPeriodsLoss(nn.Module):
    def __init__(self, base_loss_fn, past_loss_weight, future_loss_weight, short_memory_len, reduction='none'):
        super(WeightedPeriodsLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.past_loss_weight = past_loss_weight
        self.future_loss_weight = future_loss_weight
        self.short_memory_len = short_memory_len
        self.reduction = reduction

    def forward(self, outputs, targets, temporal_masks):
        past_logits = outputs[:,0,:]
        present_logits = outputs[:,1,:]
        future_logits = outputs[:,2,:]
        past_spotting_labels = targets[:,0,:]
        present_spotting_labels = targets[:,1,:]
        future_spotting_labels = targets[:,2,:]
        past_loss = self.past_loss_weight*self.base_loss_fn(past_logits, past_spotting_labels)
        present_loss = self.base_loss_fn(present_logits, present_spotting_labels)
        future_loss = self.future_loss_weight*self.base_loss_fn(future_logits, future_spotting_labels)
        loss = past_loss+present_loss+future_loss
        # mask_weights = torch.ones_like(temporal_masks, dtype=targets.dtype)
        # mask_weights[temporal_masks] = 0
        # loss = torch.cat((past_loss, present_loss, future_loss), dim=1) * mask_weights.unsqueeze(-1)

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


def get_loss_fn(dataset, loss_fn_config):
    print("Loss function info:")
    print(f"Base loss function: {loss_fn_config['name']}")
    if loss_fn_config["name"].startswith("BCEWithLogitsLoss"):
        pos_weight = dataset.get_bce_loss_pos_weights(**loss_fn_config["pos_weights_params"]) if "pos_weight" not in loss_fn_config["params"] else loss_fn_config["params"].pop("pos_weight")
        print(f"Pos weight: {pos_weight}")
        print(f"Params: {loss_fn_config['params']}")
        base_loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='none',
            **loss_fn_config["params"]
        )
    elif loss_fn_config["name"].startswith("FocalLoss"):
        alpha = dataset.get_focal_loss_alpha() if "alpha" not in loss_fn_config["params"] else loss_fn_config["params"].pop("alpha")
        print(f"Alpha: {alpha}")
        print(f"Params: {loss_fn_config['params']}")
        base_loss_fn = FocalLoss(
            alpha=alpha,
            reduction='none',
            **loss_fn_config["params"]
        )
    else:
        raise Exception("Unkown loss function name!")
    if loss_fn_config["name"].endswith("-OCP"):
        print(f"OneClassPenalty added with params: {loss_fn_config['ocp_params']}")
        base_loss_fn = LossWithOneClassPenalty(
            base_loss_fn,
            **loss_fn_config["ocp_params"]
        )
    print(f"WeightedPeriodsLoss params: {loss_fn_config['wp_params']}")
    return WeightedPeriodsLoss(base_loss_fn, **loss_fn_config["wp_params"])
