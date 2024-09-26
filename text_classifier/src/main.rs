use std::collections::HashMap;
use rust_stemmers::{Algorithm, Stemmer};

struct NaiveBayes {
    class_counts: HashMap<String, usize>,
    word_counts: HashMap<String, HashMap<String, usize>>,
    vocab: Vec<String>,
}

impl NaiveBayes {
    fn new() -> Self {
        NaiveBayes {
            class_counts: HashMap::new(),
            word_counts: HashMap::new(),
            vocab: Vec::new(),
        }
    }

    fn train(&mut self, text: &str, class: &str) {
        *self.class_counts.entry(class.to_string()).or_insert(0) += 1;

        let en_stemmer = Stemmer::create(Algorithm::English);
        for word in text.split_whitespace() {
            let stemmed_word = en_stemmer.stem(word).to_string();
            if !self.vocab.contains(&stemmed_word) {
                self.vocab.push(stemmed_word.clone());
            }
            *self.word_counts
                .entry(class.to_string())
                .or_insert_with(HashMap::new)
                .entry(stemmed_word)
                .or_insert(0) += 1;
        }
    }

    fn classify(&self, text: &str) -> String {
        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;

        let en_stemmer = Stemmer::create(Algorithm::English);
        let words: Vec<String> = text
            .split_whitespace()
            .map(|word| en_stemmer.stem(word).to_string())
            .collect();

        for (class, count) in &self.class_counts {
            let mut score = (*count as f64 / self.class_counts.values().sum::<usize>() as f64).ln();
            let word_counts = self.word_counts.get(class).unwrap();

            for word in &words {
                let word_count = *word_counts.get(word).unwrap_or(&0) as f64;
                let total_words = word_counts.values().sum::<usize>() as f64;
                score += ((word_count + 1.0) / (total_words + self.vocab.len() as f64)).ln();
            }

            if score > best_score {
                best_score = score;
                best_class = class.clone();
            }
        }

        best_class
    }
}

fn main() {
    let mut classifier = NaiveBayes::new();

    // Training data
    classifier.train("I love this product", "positive");
    classifier.train("This is great", "positive");
    classifier.train("Awesome experience", "positive");
    classifier.train("Terrible service", "negative");
    classifier.train("Disappointed with the quality", "negative");
    classifier.train("Waste of money", "negative");

    // Test the classifier
    println!("Classification: {}", classifier.classify("This is amazing"));
    println!("Classification: {}", classifier.classify("Worst purchase ever"));
}