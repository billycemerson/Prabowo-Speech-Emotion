import pandas as pd
from senticnet.senticnet import SenticNet

# Initialize SenticNet
sn = SenticNet()

def analyze_emotion(text):
    """
    Analyze a sentence using SenticNet.
    Returns emotions, average polarity, and sentiment label.
    """
    words = text.lower().split()
    emotions = []
    polarity_values = []

    for word in words:
        try:
            # Get polarity value for each word
            polarity_value = float(sn.polarity_value(word))
            moodtags = sn.moodtags(word)

            # Collect emotions and polarity
            emotions.extend(moodtags)
            polarity_values.append(polarity_value)
        except:
            # Skip words not found in SenticNet
            continue

    # Compute average polarity
    avg_polarity = sum(polarity_values) / len(polarity_values) if polarity_values else 0

    # Determine sentiment based on polarity threshold
    if avg_polarity > 0.05:
        sentiment = "positive"
    elif avg_polarity < -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Keep only unique emotions, take top 3
    unique_emotions = list(set(emotions))[:3]

    return {
        "emotions": unique_emotions,
        "avg_polarity": avg_polarity,
        "sentiment": sentiment
    }

def main():
    # Load dataset
    data = pd.read_csv("../data/data.csv")

    # Apply SenticNet analysis on the 'translated' column
    data["analysis"] = data["translated"].apply(analyze_emotion)

    # Expand analysis results into separate columns
    data["emotions"] = data["analysis"].apply(lambda x: x["emotions"])
    data["avg_polarity"] = data["analysis"].apply(lambda x: x["avg_polarity"])
    data["sentiment"] = data["analysis"].apply(lambda x: x["sentiment"])

    # Save results to CSV
    data.to_csv("../data/result.csv", index=False, encoding="utf-8")
    print("[+] Analysis completed. Results saved to ../data/result.csv")

if __name__ == "__main__":
    main()