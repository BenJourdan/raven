use std::collections::HashMap;

/// Compute the adjusted Rand index between two cluster labellings.
///
/// The result is `1.0` for identical labellings. Values near `0.0` indicate
/// agreement close to chance, while negative values indicate worse-than-chance
/// agreement. Both slices must describe the same items in the same order.
pub fn adjusted_rand_index(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    assert_eq!(
        labels_true.len(),
        labels_pred.len(),
        "label arrays must be the same length"
    );
    if labels_true == labels_pred {
        return 1.0;
    }
    let n = labels_true.len();
    if n < 2 {
        return 1.0;
    }

    let mut contingency = HashMap::<(usize, usize), usize>::new();
    let mut count_true = HashMap::<usize, usize>::new();
    let mut count_pred = HashMap::<usize, usize>::new();

    for (&t, &p) in labels_true.iter().zip(labels_pred.iter()) {
        *contingency.entry((t, p)).or_insert(0) += 1;
        *count_true.entry(t).or_insert(0) += 1;
        *count_pred.entry(p).or_insert(0) += 1;
    }

    let choose2 = |x: usize| -> f64 { (x as f64) * ((x as f64) - 1.0) / 2.0 };

    let sum_comb_c = contingency.values().fold(0.0, |acc, &v| acc + choose2(v));
    let sum_comb_true = count_true.values().fold(0.0, |acc, &v| acc + choose2(v));
    let sum_comb_pred = count_pred.values().fold(0.0, |acc, &v| acc + choose2(v));

    let total_pairs = choose2(n);
    let expected_index = (sum_comb_true * sum_comb_pred) / total_pairs;
    let max_index = 0.5 * (sum_comb_true + sum_comb_pred);

    if (max_index - expected_index).abs() < f64::EPSILON {
        0.0
    } else {
        (sum_comb_c - expected_index) / (max_index - expected_index)
    }
}

#[cfg(test)]
mod tests {
    use super::adjusted_rand_index;

    #[test]
    fn ari_is_one_for_identical_labels() {
        let labels = [0, 0, 1, 1, 2, 2];
        assert_eq!(adjusted_rand_index(&labels, &labels), 1.0);
    }

    #[test]
    fn ari_is_permutation_invariant() {
        let labels_true = [0, 0, 1, 1, 2, 2];
        let labels_pred = [7, 7, 4, 4, 9, 9];
        assert_eq!(adjusted_rand_index(&labels_true, &labels_pred), 1.0);
    }

    #[test]
    fn ari_matches_known_partial_agreement_value() {
        let labels_true = [0, 0, 0, 1, 1, 1];
        let labels_pred = [0, 0, 1, 1, 2, 2];
        let ari = adjusted_rand_index(&labels_true, &labels_pred);
        assert!((ari - 0.24242424242424243).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "label arrays must be the same length")]
    fn ari_rejects_length_mismatch() {
        adjusted_rand_index(&[0, 1], &[0]);
    }
}
