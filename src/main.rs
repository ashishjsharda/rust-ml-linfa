use linfa::traits::{Fit, Predict};
use linfa::Dataset;
use linfa_clustering::KMeans;
use ndarray::Array2;

use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;

fn main() {
    // Number of samples and features
    let n_samples = 1000;
    let n_features = 2;

    // Generate a blob-like dataset
    let rng = Isaac64Rng::seed_from_u64(42);
    let dataset = Array2::random_using((n_samples, n_features), Normal::new(0., 1.).unwrap(), &mut rand::rngs::StdRng::from_rng(rng).unwrap());

    // Wrap the dataset in a Linfa `Dataset`
    let dataset = Dataset::from(dataset);

    // Configure and train KMeans model
    let n_clusters = 3;
    let model = KMeans::params(n_clusters)
        .fit(&dataset)
        .expect("KMeans fitting failed");

    // Predict the cluster for each sample
    let prediction = model.predict(&dataset);

    println!("Cluster centers:\n{:?}", model.centroids());
    println!("Predictions:\n{:?}", prediction);
}
