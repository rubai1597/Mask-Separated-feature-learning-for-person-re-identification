import torch
import numpy as np


class ReIDEvaluator(object):
    def __init__(self, args, model, num_query):
        self.cuda = args.cuda
        self.model = model
        self.num_query = num_query
        self.test_norm = args.test_norm

    def evaluate(self, val_loader, ranks=(1, 2, 4, 5, 8, 10, 16, 20)):
        distmat, (_, q_pids, q_camids, _), (_, g_pids, g_camids, _) = self._eval_base(val_loader)

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc, mAP

    def ranking_results(self, val_loader, rank_max=5):
        distmat, (_, q_pids, _, q_paths), (_, g_pids, _, g_paths) = self._eval_base(val_loader)

        print(f"Extract top-{rank_max} results for each query")
        _, top_k_indices = torch.topk(distmat, rank_max, dim=1, largest=False)

        top_k_g_paths = g_paths[top_k_indices.cpu().numpy()]

        correct = (q_pids.unsqueeze(1) == g_pids[top_k_indices])

        return q_paths, top_k_g_paths, correct

    def _parse_features(self, dataloader):
        df, d_pids, d_camids, d_paths = ([], [], [], [])
        for imgs, pids, camids, paths in dataloader:
            if self.cuda:
                imgs = imgs.cuda()

            features = torch.cat(self.model(imgs)[1], dim=1)

            if self.test_norm:
                features = features / torch.norm(features, p=2, dim=1, keepdim=True)

            df.append(features.detach())
            d_pids.extend(pids.view(1, -1))
            d_camids.extend(camids.view(1, -1))
            d_paths.extend(paths)

        df = torch.cat(df, 0)
        d_pids = torch.cat(d_pids, 0)
        d_camids = torch.cat(d_camids, 0)
        d_paths = np.array(d_paths)

        return df, d_pids, d_camids, d_paths

    def _eval_base(self, val_loader):
        if self.test_norm:
            print("The test feature is normalized")
        self.model.eval()

        feats, pids, camids, paths = self._parse_features(val_loader)

        # query
        qf = feats[:self.num_query]
        q_pids = pids[:self.num_query]
        q_camids = camids[:self.num_query]
        q_paths = paths[:self.num_query]

        # gallery
        gf = feats[self.num_query:]
        g_pids = pids[self.num_query:]
        g_camids = camids[self.num_query:]
        g_paths = paths[self.num_query:]

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))
        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")
        m, n, d = qf.size(0), gf.size(0), gf.size(1)
        with torch.no_grad():
            q_g_dist = torch.zeros((m, n), dtype=gf.dtype)
            if self.cuda:
                q_g_dist = q_g_dist.cuda()
            for idx in range(d):
                q_g_dist += torch.abs(qf[:, idx].view(m, 1) - gf[:, idx].view(1, n)).clamp(min=1e-12).pow(2)

        return q_g_dist, [qf, q_pids, q_camids, q_paths], [gf, g_pids, g_camids, g_paths]

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view(num_q, -1)
        keep = ~((g_pids[indices] == q_pids.view(num_q, -1)) & (g_camids[indices] == q_camids.view(num_q, -1)))

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float().cpu()
        num_rel = torch.tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.tensor(range(1, max_rank + 1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel.cpu()
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()
