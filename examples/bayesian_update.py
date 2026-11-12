"""Bayesian updating after missed searches, and the unconditional probability of detection.

One stationary target is in exactly one cell. Searching cell k misses with probability 1 - d_k if the
target is there, so after a miss the weight of k is multiplied by 1 - d_k and all weights are
renormalised. The unconditional probability of detection after n_j searches of each cell is
P = 1 - sum_j p_j (1 - d_j)**n_j, which does not depend on the order of the searches.

Run: uv run python examples/bayesian_update.py
"""

from optimal_search.spatial_search import (
    cumulative_detection_probability_for_sequence,
    posterior_after_search_sequence,
    update_after_failed_search,
)


def fmt(values) -> str:
    return "(" + ", ".join(f"{float(v):.5f}" for v in values) + ")"


def main() -> None:
    # Search Example 2 of Matthiesen's 2011 presentation, as transcribed: four cells, Pd = 0.6.
    prior, detection = [0.1, 0.2, 0.4, 0.3], [0.6] * 4
    print("2011 Search Example 2: prior", fmt(prior), "detection 0.6 per look")
    print(f"  Search cell 3: chance of finding the target {prior[2] * detection[2]:.2f}; a miss leaves {1 - prior[2] * detection[2]:.2f}")
    print("  Posterior after the miss:", fmt(update_after_failed_search(prior, detection, 2)))
    print("  (presentation: 0.13158, 0.26316, 0.21053, 0.39474)")

    # Search Example 1: four equally likely cells, perfect detection, searched in turn.
    prior, detection = [0.25] * 4, [1.0] * 4
    print("\n2011 Search Example 1: four equally likely cells, perfect detection")
    for looks in range(1, 5):
        sequence = list(range(looks))
        print(f"  after {looks} look(s): P(detected) = {cumulative_detection_probability_for_sequence(prior, detection, sequence):.2f}, "
              f"posterior {fmt(posterior_after_search_sequence(prior, detection, sequence[:3])) if looks < 4 else '(found)'}")

    # Order independence: the same searches in a different order give the same probability of detection.
    prior, detection = [0.6, 0.4], [0.5, 0.5]
    print("\nTwo cells, prior (0.6, 0.4), detection 0.5: the probability of detection depends only on the counts")
    for sequence in ([0, 1], [1, 0], [0, 1, 0], [1, 0, 0]):
        print(f"  searches {sequence}: P(detected) = {cumulative_detection_probability_for_sequence(prior, detection, sequence):.4f}")
    print("  The 1999/2003 'actual cumulative probability' sums conditional terms and gives 0.5857 and 0.5750 for")
    print("  the first two orders (see docs/spatial-search-verification.md); the law of total probability gives 0.5.")


if __name__ == "__main__":
    main()
