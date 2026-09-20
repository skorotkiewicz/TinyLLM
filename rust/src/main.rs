// TinyLLM in Rust — bigram Markov chain, port of your JS version
// run: cargo new tinyllm-rs --bin  (copy this into src/main.rs)
//      cargo add rand && cargo run
use rand::seq::SliceRandom;
use std::collections::HashMap;

fn main() {
    let text = "mama dad mama cat dad mama dog";
    let words: Vec<&str> = text.split_whitespace().collect();

    // train: word -> [next words]
    let mut model: HashMap<&str, Vec<&str>> = HashMap::new();
    for w in words.windows(2) {
        model.entry(w[0]).or_default().push(w[1]);
    }

    // generate: random walk
    let mut rng = rand::thread_rng();
    let mut word = "mama";
    let mut out = vec![word];
    for _ in 0..10 {
        word = match model.get(word) {
            Some(next) => next.choose(&mut rng).unwrap(),
            None => words.choose(&mut rng).unwrap(),
        };
        out.push(word);
    }
    println!("{}", out.join(" "));
}
