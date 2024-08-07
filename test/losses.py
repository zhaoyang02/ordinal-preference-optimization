import torch
from loss_utils import dcg, deterministic_neural_sort, sinkhorn_scaling, stochastic_neural_sort
import torch.nn.functional as F

all_logps=torch.tensor([[-1.2402, -0.8511, -0.9594, -1.0351, -0.9984, -0.9719, -0.8879, -1.4855]])
scores=torch.tensor([[0.7891, 0.7827, 0.7803, 0.7798, 0.7720, 0.7715, 0.7207, 0.6147]])

def neuralNDCG(y_pred, y_true, padded_value_indicator=-1, temperature=0.1, powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :return: loss value, a torch.Tensor
    """
    dev = "cpu"

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # Choose the deterministic/stochastic variant
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores,dev=dev)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask,dev=dev).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    # Mask P_hat and apply to true labels, ie approximately sort them
    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    discounted_gains = ground_truth * discounts

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + 1e-10)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    #print(f"temperature: {temperature}")
    return -1. * mean_ndcg, P_hat

def lambda_loss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, beta=0.1):
        """
        Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
        Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param alpha: score difference weight used in the sigmoid function
        :return: loss value, a torch.Tensor
        """
        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        print(f"y_pred: {y_pred}")
        print(f"y_true: {y_true}")

        '''if torch.all(y_pred.eq(0)):
            rank_matrix = torch.ones_like(y_pred)
            print("All zeros")
            print(f"Rank matrix:\n{rank_matrix}")
        else:
            y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
            rank_matrix = torch.zeros_like(indices_pred)
            for i in range(y_pred.size(0)):
                    for j, idx in enumerate(indices_pred[i]):
                            rank_matrix[i, idx] = j+1
            print(f"Rank matrix:\n{rank_matrix}")'''
        
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        # 创建一个 rank_matrix
        rank_matrix = torch.zeros_like(indices_pred)
        for i in range(y_pred.size(0)):
        # 当前行的排序索引
            row_indices = indices_pred[i]
            # 当前行的排序值
            row_sorted = y_pred_sorted[i]
            
            # 初始化排名
            rank = 1
            for j in range(len(row_sorted)):
                if j > 0 and row_sorted[j] != row_sorted[j - 1]:
                    rank = j + 1
                rank_matrix[i, row_indices[j]] = rank

        #print(f"Rank matrix:\n{rank_matrix}")

        D = torch.log2(1. + rank_matrix.float())
        #print(f"D: {D}")
        #D的倒数
        D_rec = 1. / D
        #print(f"D_rec: {D_rec}")
        G = (torch.pow(2, y_true) - 1)
        #print(f"G: {G}")

        D_rec_diffs = D_rec[:, :, None] - D_rec[:, None, :]
        #print(f"D_rec_diffs: {D_rec_diffs}")
        G_diffs = G[:, :, None] - G[:, None, :]
        #print(f"G_diffs: {G_diffs}")

        pred_diffs = y_pred[:, :, None] - y_pred[:, None, :]
        #print(f"pred_diffs: {pred_diffs}")

        diffs_trans = torch.log(1. + torch.exp(-pred_diffs))
        #print(f"diff_trans:{diffs_trans}")

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1)
        maxD = torch.log2(1. + pos_idxs.float())[None, :]
        #print(f"maxD: {maxD}")
        maxD_rec = 1. / maxD
        #print(f"maxD_rec: {maxD_rec}")
        maxD_rec_diffs = maxD_rec[:, :, None] - maxD_rec[:, None, :]
        #print(f"maxD_rec_diffs: {maxD_rec_diffs}")

        delta = torch.abs(G_diffs * D_rec_diffs/maxD_rec_diffs)
        print(f"delta: {delta}")

        losses = delta * diffs_trans
        print(f"losses: {losses}")

        indices=torch.triu_indices(row=3, col=3, offset=1)
        losses=losses[:,indices[0], indices[1]]
        #print(f"losses2: {losses}")
        losses=losses.reshape(1,-1)

        return -losses

def dpo_loss(
        logratios: torch.FloatTensor,
        scores: torch.FloatTensor,
        beta=1.0,
):

        rewards = beta * logratios
        print(f"rewards: {rewards}")

        #list-mle
        exp_scores = torch.exp(rewards)
        print(f"exp_scores: {exp_scores}")
        result = torch.zeros(scores.size(0))
        print(f"result: {result}")
        for i in range(scores.size(0)):
                prod = 1.0
                for k in range(scores.size(1)):
                        numerator = exp_scores[i, k]
                        denominator = torch.sum(exp_scores[i, k:])
                        prod *= numerator / denominator
                result[i] = torch.log(prod)

        print(result)


        '''reward_diffs = rewards[:, :, None] - rewards[:, None, :]
        print(f"reward_diffs: {reward_diffs}")
        indices=torch.triu_indices(row=3, col=3, offset=1)
        print(f"indices: {indices}")
        diag_diffs3=reward_diffs[:,indices[0], indices[1]]
        print(f"diag_diffs3: {diag_diffs3}")
        result=diag_diffs3.reshape(1,-1)
        print(f"result: {result}")

        #losses = -torch.mean(reward_diffs * scores)
        first_rows = reward_diffs[:, 0, 1:]
        print(f"first_rows: {first_rows}")
        first_rows = first_rows.reshape(1,-1)
        print(f"first_rows2: {first_rows}")
        diffs = reward_diffs[:, 0, -1]
        print(f"diffs: {diffs}")'''

        return -result

def approxNDCGLoss(y_pred, y_true, alpha=1000.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    #print(f"y_pred: {y_pred}")
    #print(f"y_true: {y_true}")

    # Here we sort the true and predicted relevancy scores.
    #y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    #y_true_sorted, _ = y_true.sort(descending=True, dim=-1)
    #print(f"indices_pred: {indices_pred}")

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    #true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    #print(f"true_sorted_by_preds: {true_sorted_by_preds}")

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true) - 1) / D, dim=-1)
    print(f"maxDCGs: {maxDCGs}")
    
    '''if torch.all(y_pred.eq(0)):
        print("All zeros")
        for i in range(y_pred.shape[0]):
            permutation = torch.randperm(y_pred.shape[1])
            true_sorted_by_preds[i] = true_sorted_by_preds[i, permutation]'''

    '''G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]
    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    print(f"y_pred_sorted: {y_pred_sorted}")
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])'''
    G = (torch.pow(2, y_true) - 1) / maxDCGs[:, None]
    #print(f"G: {G}")
    scores_diffs = (y_pred[:, :, None] - y_pred[:, None, :])
    #print(f"scores_diffs: {scores_diffs}")
    # we calculate the total sum including diagonal elements, which equals to 1/(1+exp(0))=0.5
    approx_pos = 0.5 + torch.sum(torch.sigmoid(-alpha * scores_diffs), dim=-1)
    print(f"\nalpha: {alpha}")
    print(f"approx_pos: {approx_pos}")
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)

def approxNDCGLoss2(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, alpha=1.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    print(f"alpha2: {alpha}")
    print(f"approx_pos2: {approx_pos}")
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)

def lambdaloss2(y_pred,y_true,eps=1e-10, padded_value_indicator=-1,sigma=1.,k=None):
    device = "cpu"
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
    print(f"padded_pairs_mask: {padded_pairs_mask}")

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    weights = torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])
    print(f"weights: {weights}")
    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)

    losses = torch.log(weighted_probas)
    print("raw losses:",losses)

    return -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])

def hinge_loss(
        self,
        y_pred,
        beta=0.1,
        list_size=8,
    ) -> torch.FloatTensor:

        logratios = y_pred
        print(f"logratios: {logratios}")        
        logratios_diffs = logratios[:, :, None] - logratios[:, None, :]
        print(f"logratios_diffs: {logratios_diffs}")
        indices=torch.triu_indices(row=list_size, col=list_size, offset=1)
        print(f"indices: {indices}")
        diag_diffs=logratios_diffs[:,indices[0], indices[1]].reshape(1,-1)
        print(f"diag_diffs: {diag_diffs}")

        losses = F.relu(1- beta * diag_diffs)
        print(f"losses: {losses}")        
        return losses

taus=[0.01,0.1,1.0,10.0]
alpha=250.0
scores=torch.tensor([[5,4,3,2]]).float()
all_logps=torch.tensor([[9,1,5,2]]).float()

for tau in taus:
    print("********************************************************")
    loss, P_hat = neuralNDCG(all_logps, scores, temperature=tau)
    print(f"tau: {tau}")
    print(f"neural_ndcg: {loss}")
    print(f"P_hat: {P_hat}")
    sorted_scores = torch.matmul(P_hat, all_logps.unsqueeze(-1)).squeeze(-1)
    print(f"sorted_scores: {sorted_scores}")
    print(f"sum: {torch.sum(sorted_scores)}")

