import torch
import time
import numpy as np

def mean_average_precision(query_code,
                           retrieval_code,
                           query_targets,
                           retrieval_targets,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Database data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        retrieval_targets (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()
    model.train()
    return code


def test(model, query_dataloader, retrieval_dataloader, code_length, topk, device):
    load_start = time.time()
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    load_end = time.time()
    
    cal_start = time.time()
    mAP = mean_average_precision(
        query_code.to(device),
        retrieval_code.to(device),
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_dataloader.dataset.get_onehot_targets().to(device),
        device,
        topk,
    )
    cal_end = time.time()
    # logger.debug('[load_time:{:.4f}][cal_time:{:.4f}]'.format(load_end-load_start, cal_end-cal_start))
    return mAP, query_code, retrieval_code
