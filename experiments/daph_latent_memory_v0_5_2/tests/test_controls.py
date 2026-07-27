import torch
from daph_latent_memory.evaluation.controls import (
    deranged_indices, shuffled_global, shuffled_same_class, shuffled_wrong_class,
    nearest_neighbor_wrong, farthest_neighbor, normalized_latents, scaled_latents,
    sign_flipped_latents, mean_latent, centroid_latent, pca_low_rank, pca_residual,
    random_latents_like, random_norm_matched_latents_like, corruption_sweep,
)
from daph_latent_memory.state.corruption import corrupt_latent_vector

def test_derangement_has_no_fixed_points():
    perm = deranged_indices(20, seed=1)
    assert all(i != j for i, j in enumerate(perm))

def test_derangement_small_n():
    perm = deranged_indices(2, seed=1)
    assert perm == [1, 0]

def test_shuffled_global_no_fixed_points():
    z = torch.randn(10, 4, 8)
    shuffled = shuffled_global(z, seed=42)
    # Check no row is in its original position
    for i in range(10):
        assert not torch.allclose(z[i], shuffled[i])

def test_shuffled_same_class_preserves_class_groups():
    z = torch.randn(10, 4, 8)
    labels = ["add"] * 5 + ["sub"] * 5
    shuffled = shuffled_same_class(z, labels, seed=42)
    # Each shuffled entry should come from the same class
    for i in range(10):
        original_class = labels[i]
        # Find which original index this shuffled entry came from
        for j in range(10):
            if torch.allclose(shuffled[i], z[j]):
                assert labels[j] == original_class, f"Class mismatch at {i}"
                break

def test_shuffled_wrong_class_changes_class():
    z = torch.randn(10, 4, 8)
    labels = ["add"] * 5 + ["sub"] * 5
    shuffled = shuffled_wrong_class(z, labels, seed=42)
    for i in range(10):
        # Should not be from same class
        for j in range(10):
            if torch.allclose(shuffled[i], z[j]):
                assert labels[j] != labels[i], f"Same class at {i} from {j}"
                break

def test_normalized_latents_unit_norm():
    z = torch.randn(5, 4, 8)
    normed = normalized_latents(z)
    norms = normed.float().norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_scaled_latents():
    z = torch.randn(5, 4, 8)
    scaled = scaled_latents(z, c=3.0)
    assert torch.allclose(scaled, z * 3.0)

def test_sign_flipped():
    z = torch.randn(5, 4, 8)
    flipped = sign_flipped_latents(z)
    assert torch.allclose(flipped, -z)

def test_mean_latent_all_same():
    z = torch.randn(5, 4, 8)
    mean = mean_latent(z)
    for i in range(5):
        assert torch.allclose(mean[i], mean[0])

def test_centroid_latent_per_class():
    z = torch.randn(10, 4, 8)
    labels = ["add"] * 5 + ["sub"] * 5
    cent = centroid_latent(z, labels)
    add_centroid = z[:5].mean(dim=0)
    for i in range(5):
        assert torch.allclose(cent[i], add_centroid, atol=1e-5)

def test_pca_low_rank_shape():
    z = torch.randn(10, 4, 8)
    pca = pca_low_rank(z, k=4)
    assert pca.shape == z.shape

def test_pca_residual_shape():
    z = torch.randn(10, 4, 8)
    residual = pca_residual(z, k=4)
    assert residual.shape == z.shape

def test_corruption_sweep_reproducible():
    z = torch.randn(5, 4, 8)
    sweep1 = corruption_sweep(z, [0.1, 0.5], [1, 2])
    sweep2 = corruption_sweep(z, [0.1, 0.5], [1, 2])
    for alpha in [0.1, 0.5]:
        for seed in [1, 2]:
            assert torch.allclose(sweep1[alpha][seed], sweep2[alpha][seed])

def test_corruption_zero_alpha_unchanged():
    z = torch.randn(5, 4, 8)
    corrupted = corrupt_latent_vector(z, alpha=0.0, seed=42)
    assert torch.allclose(corrupted, z)

def test_corruption_nonzero_alpha_changes():
    z = torch.randn(5, 4, 8)
    corrupted = corrupt_latent_vector(z, alpha=1.0, seed=42)
    assert not torch.allclose(corrupted, z)

def test_random_norm_matched_preserves_norm():
    z = torch.randn(5, 4, 8)
    rand = random_norm_matched_latents_like(z, seed=42)
    orig_norms = z.norm(dim=-1)
    rand_norms = rand.norm(dim=-1)
    assert torch.allclose(orig_norms, rand_norms, atol=1e-4)
