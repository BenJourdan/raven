use raven::{EdgeUpdate, Raven, RavenConfig, TrialWeighting};

fn main() -> Result<(), raven::RavenError> {
    let mut config = RavenConfig::new(2);
    config.coreset_size = 4;
    config.sampling_seeds = 2;
    config.num_trials = 5;
    config.rng_seed = Some(42);
    config.node_capacity = 16;
    config.expected_edges_per_node = 4;

    let mut index = Raven::new(config)?;
    index.update_edges([
        EdgeUpdate::set(1, 2, 1.0),
        EdgeUpdate::set(2, 3, 1.0),
        EdgeUpdate::set(1, 3, 1.0),
        EdgeUpdate::set(10, 11, 1.0),
        EdgeUpdate::set(11, 12, 1.0),
        EdgeUpdate::set(10, 12, 1.0),
    ])?;
    index.flush()?;

    let pairs = [(1, 2), (1, 10), (10, 11)];
    let scores = index.score_pairs(&pairs, TrialWeighting::ScoreSoftmax, None)?;

    for ((u, v), score) in pairs.into_iter().zip(scores) {
        println!("pair=({u}, {v}) score={score:.3}");
    }

    Ok(())
}
