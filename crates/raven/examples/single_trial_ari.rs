use raven::{EdgeUpdate, Raven, RavenConfig};
use raven_core::metrics::adjusted_rand_index;

fn main() -> Result<(), raven::RavenError> {
    let mut config = RavenConfig::new(2);
    config.coreset_size = 4;
    config.sampling_seeds = 2;
    config.num_trials = 1;
    config.rng_seed = Some(42);
    config.node_capacity = 16;
    config.expected_edges_per_node = 4;

    let mut index = Raven::new(config)?;
    let query_nodes = [1, 2, 3, 10, 11, 12];
    let true_labels = [0, 0, 0, 1, 1, 1];

    let batches = [
        vec![
            EdgeUpdate::set(1, 2, 1.0),
            EdgeUpdate::set(2, 3, 1.0),
            EdgeUpdate::set(10, 11, 1.0),
            EdgeUpdate::set(11, 12, 1.0),
        ],
        vec![EdgeUpdate::set(1, 3, 1.0), EdgeUpdate::set(10, 12, 1.0)],
    ];

    for (batch_index, batch) in batches.into_iter().enumerate() {
        index.update_edges(batch)?;
        index.flush()?;

        let result = index.query(&query_nodes)?;
        let ari = adjusted_rand_index(&true_labels, &result.labels);
        println!(
            "batch={batch_index} labels={:?} ari={ari:.3}",
            result.labels
        );
    }

    Ok(())
}
