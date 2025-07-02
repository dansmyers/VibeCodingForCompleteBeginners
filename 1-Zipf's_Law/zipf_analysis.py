import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def read_and_preprocess_text(filename):
    """
    Read a text file and preprocess it for word frequency analysis.
    
    Args:
        filename (str): Path to the text file
        
    Returns:
        list: List of preprocessed words
    """
    print(f"Reading file: {filename}")
    
    # Read the entire file
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    print(f"File loaded. Total characters: {len(text)}")
    
    # Convert to lowercase to ignore capitalization
    text = text.lower()
    
    # Remove punctuation and keep only letters and spaces
    # This regex keeps only alphabetic characters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Split text into individual words and remove empty strings
    words = [word for word in text.split() if word]
    
    print(f"Total words after preprocessing: {len(words)}")
    return words

def count_word_frequencies(words):
    """
    Count the frequency of each word and rank them.
    
    Args:
        words (list): List of words
        
    Returns:
        tuple: (list of (word, frequency) tuples sorted by frequency, 
                list of frequencies in rank order)
    """
    print("Counting word frequencies...")
    
    # Count frequency of each word
    word_counts = Counter(words)
    
    # Sort by frequency (highest first) - this gives us the ranking
    sorted_word_freq = word_counts.most_common()
    
    # Extract just the frequencies in rank order
    frequencies = [freq for word, freq in sorted_word_freq]
    
    print(f"Unique words found: {len(sorted_word_freq)}")
    print(f"Most common word: '{sorted_word_freq[0][0]}' appears {sorted_word_freq[0][1]} times")
    
    return sorted_word_freq, frequencies

def calculate_zipf_predictions(frequencies):
    """
    Calculate what Zipf's Law predicts for each rank.
    
    Zipf's Law: frequency = C / rank, where C is the frequency of the top-ranked word
    
    Args:
        frequencies (list): List of actual frequencies in rank order
        
    Returns:
        list: Predicted frequencies according to Zipf's Law
    """
    print("Calculating Zipf's Law predictions...")
    
    # In Zipf's Law, C (the constant) equals the frequency of the most common word
    C = frequencies[0]
    
    # Calculate predicted frequency for each rank
    # Rank starts at 1 (not 0), so we use rank = index + 1
    zipf_predictions = [C / (rank + 1) for rank in range(len(frequencies))]
    
    return zipf_predictions

def create_zipf_plot(frequencies, zipf_predictions, word_freq_pairs, max_rank=1000):
    """
    Create a log-log plot comparing actual frequencies to Zipf's Law predictions.
    
    Args:
        frequencies (list): Actual word frequencies in rank order
        zipf_predictions (list): Predicted frequencies from Zipf's Law
        word_freq_pairs (list): List of (word, frequency) tuples for labeling
        max_rank (int): Maximum rank to plot (to avoid overcrowding)
    """
    print(f"Creating plot for top {max_rank} words...")
    
    # Limit data to top max_rank words to make plot readable
    plot_frequencies = frequencies[:max_rank]
    plot_predictions = zipf_predictions[:max_rank]
    ranks = list(range(1, len(plot_frequencies) + 1))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual frequencies
    plt.loglog(ranks, plot_frequencies, 'bo-', linewidth=2, markersize=4, 
               label='Actual Frequencies', alpha=0.7)
    
    # Plot Zipf's Law predictions
    plt.loglog(ranks, plot_predictions, 'r--', linewidth=2, 
               label="Zipf's Law Prediction", alpha=0.8)
    
    # Formatting
    plt.xlabel('Rank (log scale)', fontsize=12)
    plt.ylabel('Frequency (log scale)', fontsize=12)
    plt.title("Word Frequency vs Rank: Actual Data vs Zipf's Law", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some annotations for the most common words
    for i in range(min(5, len(word_freq_pairs))):
        word, freq = word_freq_pairs[i]
        plt.annotate(f'{word}', 
                    xy=(i+1, freq), 
                    xytext=(i+1, freq*1.5),
                    fontsize=9,
                    ha='center')
    
    plt.tight_layout()
    plt.show()

def calculate_fit_quality(frequencies, zipf_predictions, top_n=100):
    """
    Calculate how well the actual data fits Zipf's Law.
    
    Args:
        frequencies (list): Actual frequencies
        zipf_predictions (list): Zipf's Law predictions
        top_n (int): Number of top words to analyze
        
    Returns:
        float: R-squared value (closer to 1.0 means better fit)
    """
    # Focus on top_n words for fit analysis
    actual = np.array(frequencies[:top_n])
    predicted = np.array(zipf_predictions[:top_n])
    
    # Calculate R-squared value
    # R² = 1 - (sum of squared residuals / total sum of squares)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared

def main():
    """
    Main function that orchestrates the entire analysis.
    """
    print("=== Zipf's Law Analysis ===\n")
    
    # Step 1: Read and preprocess the text file
    filename = "moby_dick.txt"  # Change this to your file path
    words = read_and_preprocess_text(filename)
    
    print()
    
    # Step 2: Count word frequencies and rank them
    word_freq_pairs, frequencies = count_word_frequencies(words)
    
    print()
    
    # Step 3: Calculate Zipf's Law predictions
    zipf_predictions = calculate_zipf_predictions(frequencies)
    
    print()
    
    # Step 4: Display some statistics
    print("=== Top 10 Most Common Words ===")
    for i, (word, freq) in enumerate(word_freq_pairs[:10], 1):
        zipf_pred = zipf_predictions[i-1]
        print(f"{i:2d}. '{word}': {freq:4d} times (Zipf predicts: {zipf_pred:.1f})")
    
    print()
    
    # Step 5: Calculate how well data fits Zipf's Law
    r_squared = calculate_fit_quality(frequencies, zipf_predictions)
    print(f"R-squared fit for top 100 words: {r_squared:.3f}")
    print("(Values closer to 1.0 indicate better fit to Zipf's Law)")
    
    print()
    
    # Step 6: Create the visualization
    create_zipf_plot(frequencies, zipf_predictions, word_freq_pairs)
    
    print("\nAnalysis complete!")

# Run the analysis
if __name__ == "__main__":
    main()