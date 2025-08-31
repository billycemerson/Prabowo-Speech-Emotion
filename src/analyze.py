import pandas as pd
from collections import Counter
import ast

def main():
    # Read processed result.csv
    df = pd.read_csv("../data/result.csv")

    # Parse emotions column safely
    def safe_parse(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except:
            return []
    
    df["emotions"] = df["emotions"].apply(safe_parse)

    # Drop joy from emotions
    df["emotions"] = df["emotions"].apply(lambda emos: [e for e in emos if e != "#joy"])

    # Basic counts
    total_sentences = len(df)
    no_emo = (df["emotions"].apply(len) == 0).sum()
    one_emo = (df["emotions"].apply(len) == 1).sum()
    multi_emo = (df["emotions"].apply(len) >= 2).sum()

    # Flatten all emotions
    all_emotions = [emo for sublist in df["emotions"] for emo in sublist]
    emotion_counter = Counter(all_emotions)

    # Sentiment distribution
    sentiment_counter = Counter(df["sentiment"])

    # Alur emosional (dominant emotion per sentence, in order)
    emotion_flow = [emos[0] if emos else "-" for emos in df["emotions"]]

    # Generate summary
    summary = []
    summary.append("📊 EMOTIONAL ANALYSIS OF PRESIDENT'S SPEECH")
    summary.append("=" * 50)
    summary.append(f"Total sentences analyzed: {total_sentences}")
    summary.append(f"- No emotion detected: {no_emo} sentences")
    summary.append(f"- Single emotion: {one_emo} sentences")
    summary.append(f"- Multiple emotions: {multi_emo} sentences\n")

    summary.append("🔹 Overall Emotion Frequency (excluding #joy):")
    for emo, count in emotion_counter.most_common():
        summary.append(f"- {emo}: {count}")

    summary.append("\n🔹 Sentiment Distribution:")
    for sent, count in sentiment_counter.items():
        summary.append(f"- {sent}: {count} ({count/total_sentences:.1%})")

    # Emotional trajectory
    summary.append("\n🔹 Emotional Flow Across Speech (dominant emotion per sentence):")
    summary.append(" → ".join(emotion_flow))

    # Key observations
    summary.append("\nKEY INSIGHTS:")
    if emotion_counter:
        summary.append(f"- Dominant emotion: {emotion_counter.most_common(1)[0][0]}")
    if "positive" in sentiment_counter and "negative" in sentiment_counter:
        ratio = sentiment_counter["positive"] / max(1, sentiment_counter["negative"])
        summary.append(f"- Positive to Negative ratio: {ratio:.2f}:1")
    summary.append("- Emotional flow suggests how tone shifts across the speech.")

    # ----- Additional: Analysis by 4 logical parts -----
    summary.append("\n🔹 EMOTIONAL ANALYSIS PER PART:")
    # Define parts (indexes of sentences; adjust sesuai data)
    parts = {
        "Part 1": range(0, 5),   # sentences 0–4
        "Part 2": range(5, 8),   # sentences 5–7
        "Part 3": range(8, 14),  # sentences 8–13
        "Part 4": range(14, total_sentences)  # sentences 14–end
    }

    for part_name, idx_range in parts.items():
        part_df = df.iloc[idx_range]
        # Flatten emotions for the part
        part_emotions = [emo for sublist in part_df["emotions"] for emo in sublist]
        part_counter = Counter(part_emotions)
        dominant_emo = part_counter.most_common(1)[0][0] if part_counter else "-"
        avg_polarity = part_df["avg_polarity"].mean() if "avg_polarity" in part_df.columns else 0
        summary.append(f"\n{part_name}:")
        summary.append(f"- Dominant emotion: {dominant_emo}")
        summary.append(f"- Average polarity: {avg_polarity:.3f}")
        summary.append(f"- Sentence count: {len(part_df)}")
        if len(part_df) > 0:
            flow = [emos[0] if emos else "-" for emos in part_df["emotions"]]
            summary.append(f"- Emotional flow in part: {' → '.join(flow)}")

    # Save results
    with open("../data/analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("[+] Simplified analysis with per-part insight saved to ../data/analysis.txt")

if __name__ == "__main__":
    main()