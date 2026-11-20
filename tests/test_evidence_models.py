import numpy as np
import torch

from scripts.research.evidence_data import opposite_rows, scene_folds
from scripts.research.evidence_eval import apply_candidate
from scripts.research.evidence_model import EvidenceVerifier, pair_loss


def test_opposite_images_require_signed_label_and_stay_in_training_partition():
    captions = ["exactly one cylinder"] * 5 + ["there is a blue cube"]
    labels = np.array([1, 0, 1, 0, 0, 0])
    scenes = np.array([0, 0, 1, 2, 3, 4])
    allowed = np.array([True, True, True, True, False, True])
    swaps = opposite_rows(captions, labels, scenes, allowed, seed=421)
    for i, j in enumerate(swaps):
        if j >= 0:
            assert allowed[i] and allowed[j]
            assert scenes[i] != scenes[j] and labels[i] != labels[j]
            assert captions[i] == captions[j]
    assert swaps[0] == 3
    assert (swaps[[4, 5]] == -1).all()
    np.testing.assert_array_equal(np.bincount(scene_folds(12, 4, seed=421)), [3, 3, 3, 3])


def test_logits_are_per_caption_and_pair_loss_trains_visual_encoder_head():
    torch.manual_seed(421)
    model = EvidenceVerifier().eval()
    detail = torch.randn(2, 4, 6, 99)
    semantic = torch.randn(2, 2, 2, 144)
    words = torch.zeros(2, 3, 4, dtype=torch.long)
    words[:, 0, 0] = 15
    words[:, 2, 0] = 13
    meta = torch.tensor([[0, 1, 2], [0, 1, 1]])
    score, count = model(detail, semantic, words, meta)
    singleton, _ = model(detail[:1], semantic[:1], words[:1], meta[:1])
    torch.testing.assert_close(singleton[0], score[0])
    changed_image, _ = model(detail[1:], semantic[1:], words[:1], meta[:1])
    assert (singleton - changed_image).abs().item() > 1e-5
    loss = pair_loss(score.view(1, 2), count.view(1, 2, 20), meta.view(1, 2, 3), torch.ones(1))
    loss.backward()
    assert torch.isfinite(model.query.weight.grad).all()
    assert model.detail[1].weight.grad.abs().sum() > 0


def test_candidate_blend_can_repair_uncertain_pair_and_keeps_high_margin():
    base = np.array([-.25, .25, 7, -7, .50])
    alt = np.array([3., -3., -20, 20, -20])
    combined = apply_candidate(base, alt, np.array([[0, 1], [2, 3]]))
    assert combined[0] > combined[1]
    np.testing.assert_array_equal(combined[2:], base[2:])
