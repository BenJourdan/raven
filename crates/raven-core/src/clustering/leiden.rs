use std::sync::Arc;

use faer::sparse::SparseRowMat;
use leiden_rs::{GraphDataBuilder, Leiden, LeidenConfig};

use crate::types::{AlgType, FloatScalar};

/// Cluster a symmetric sparse graph using the Leiden community detection algorithm.
///
/// Raven's coreset graph is treated as undirected. Only entries above the
/// diagonal are copied into `leiden-rs`, so diagonal entries are ignored and the
/// lower triangle is assumed to mirror the upper triangle.
pub fn leiden_community_detection<T>(
    graph: &mut SparseRowMat<usize, T>,
    config: &LeidenConfig,
) -> (Vec<usize>, usize)
where
    T: FloatScalar,
{
    let (symbolic, vals) = graph.parts();
    let (nrows, ncols, row_ptr, _row_nnz, col_idx) = symbolic.parts();
    assert_eq!(nrows, ncols, "Leiden clustering requires a square graph");

    if nrows == 0 {
        return (Vec::new(), 0);
    }

    let mut builder = GraphDataBuilder::new(nrows);
    for i in 0..nrows {
        for idx in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[idx];
            if i >= j {
                continue;
            }

            let weight = vals[idx];
            assert!(
                weight.is_finite() && weight >= T::ZERO,
                "Leiden clustering requires finite non-negative edge weights"
            );
            if weight > T::ZERO {
                builder
                    .add_edge(
                        i,
                        j,
                        weight
                            .to_f64()
                            .expect("finite Raven float should convert to f64"),
                    )
                    .expect("validated Leiden graph edge should be accepted");
            }
        }
    }

    let leiden_graph = builder
        .build()
        .expect("validated Leiden graph should build successfully");
    let output = Leiden::new(config.clone())
        .run(&leiden_graph)
        .expect("Leiden clustering failed");
    let labels = output.partition.as_slice().to_vec();
    let num_communities = output.partition.num_communities();

    (labels, num_communities)
}

/// Wrap [`leiden_community_detection`] as a Raven clustering callback.
///
/// The requested cluster count is ignored. Leiden chooses its own number of
/// communities from the configured quality objective and resolution parameter.
pub fn leiden_community_detection_alg<T>(config: LeidenConfig) -> AlgType<T>
where
    T: FloatScalar + Send + Sync + 'static,
{
    Arc::new(move |graph, _requested_k| leiden_community_detection(graph, &config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::SymbolicSparseRowMat;

    fn two_block_graph<T: FloatScalar>(with_diagonal: bool) -> SparseRowMat<usize, T> {
        let n = 6;
        let mut rows = vec![Vec::<(usize, T)>::new(); n];
        let strong = T::ONE;
        let weak = T::from(0.01).expect("test scalar should convert");
        let diagonal = T::from(1000.0).expect("test scalar should convert");

        for &(i, j, weight) in &[
            (0, 1, strong),
            (0, 2, strong),
            (1, 2, strong),
            (3, 4, strong),
            (3, 5, strong),
            (4, 5, strong),
            (2, 3, weak),
        ] {
            rows[i].push((j, weight));
            rows[j].push((i, weight));
        }

        if with_diagonal {
            for (i, row) in rows.iter_mut().enumerate() {
                row.push((i, diagonal));
            }
        }

        for row in &mut rows {
            row.sort_unstable_by_key(|(col, _)| *col);
        }

        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut nnz_per_row = Vec::with_capacity(n);
        indptr.push(0);

        for row in rows {
            nnz_per_row.push(row.len());
            for (col, weight) in row {
                indices.push(col);
                data.push(weight);
            }
            indptr.push(indices.len());
        }

        SparseRowMat::new(
            SymbolicSparseRowMat::new_checked(n, n, indptr, Some(nnz_per_row), indices),
            data,
        )
    }

    #[test]
    fn empty_graph_returns_empty_labels() {
        let mut graph = SparseRowMat::new(
            SymbolicSparseRowMat::new_checked(0, 0, vec![0], Some(Vec::new()), Vec::new()),
            Vec::<f64>::new(),
        );

        assert_eq!(
            leiden_community_detection(&mut graph, &LeidenConfig::default()),
            (Vec::new(), 0)
        );
    }

    #[test]
    fn diagonal_entries_are_ignored() {
        let config = LeidenConfig {
            seed: Some(42),
            ..LeidenConfig::default()
        };
        let mut without_diagonal = two_block_graph::<f64>(false);
        let mut with_diagonal = two_block_graph::<f64>(true);

        let baseline = leiden_community_detection(&mut without_diagonal, &config);
        let diagonal = leiden_community_detection(&mut with_diagonal, &config);

        assert_eq!(baseline, diagonal);
        assert_eq!(baseline.0.len(), 6);
    }
}
