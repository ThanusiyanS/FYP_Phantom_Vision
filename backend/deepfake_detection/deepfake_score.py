import pandas as pd
import os

def compute_final_deepfake_score(audio_score, video_score, audio_thresh=0.52, video_thresh=0.68):
    """
    Compute final deepfake score based on weighted combination of audio and video scores.
    Weights are inversely proportional to thresholds (lower threshold = more confident modality).
    """
    inv_audio = 1 / audio_thresh
    inv_video = 1 / video_thresh

    weight_audio = inv_audio / (inv_audio + inv_video)
    weight_video = 1 - weight_audio

    final_score = (weight_audio * audio_score) + (weight_video * video_score)
    return round(final_score, 4), weight_audio, weight_video

def calculate_final_deepfake_score():
    """
    Calculate final deepfake score by applying weighted combination of audio and video scores
    based on their respective thresholds
    """
    # Path to the input CSV file
    input_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_score.csv")
    output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_deepfake_score.csv")
    
    # Check if input file exists
    if not os.path.exists(input_csv_path):
        print(f"Input file not found: {input_csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} records from {input_csv_path}")
    
    # Convert score columns to numeric, handling empty strings and NaN values
    df['avg_audio_deepfake_score'] = pd.to_numeric(df['avg_audio_deepfake_score'], errors='coerce').fillna(0)
    df['avg_video_deepfake_score'] = pd.to_numeric(df['avg_video_deepfake_score'], errors='coerce').fillna(0)
    
    # Ensure is_audio_deepfake and is_video_deepfake are present and numeric
    df['is_audio_deepfake'] = pd.to_numeric(df.get('is_audio_deepfake', 0), errors='coerce').fillna(0).astype(int)
    df['is_video_deepfake'] = pd.to_numeric(df.get('is_video_deepfake', 0), errors='coerce').fillna(0).astype(int)
    
    # Calculate final deepfake score using weighted combination based on thresholds
    final_scores = []
    audio_weights = []
    video_weights = []
    
    for _, row in df.iterrows():
        audio_score = row['avg_audio_deepfake_score']
        video_score = row['avg_video_deepfake_score']
        
        final_score, weight_audio, weight_video = compute_final_deepfake_score(audio_score, video_score)
        final_scores.append(final_score)
        audio_weights.append(weight_audio)
        video_weights.append(weight_video)
    
    df['final_deepfake_score'] = final_scores
    df['audio_weight'] = audio_weights
    df['video_weight'] = video_weights
    
    # Calculate deepfake_label using OR logic
    df['deepfake_label'] = ((df['is_audio_deepfake'] == 1) | (df['is_video_deepfake'] == 1)).astype(int)
    
    # Create output DataFrame with required columns
    output_df = df[['resource_id', 'audio_file', 'video_file', 'final_deepfake_score', 'deepfake_label']].copy()
    
    # Save to new CSV file
    output_df.to_csv(output_csv_path, index=False)
    print(f"Final deepfake scores saved to: {output_csv_path}")
    print(f"Processed {len(output_df)} records")
    
    # Display some statistics
    print(f"\nFinal Deepfake Score Statistics:")
    print(f"Mean: {output_df['final_deepfake_score'].mean():.4f}")
    print(f"Max: {output_df['final_deepfake_score'].max():.4f}")
    print(f"Min: {output_df['final_deepfake_score'].min():.4f}")
    
    # Show top 5 highest scores
    print(f"\nTop 5 Highest Final Deepfake Scores:")
    top_scores = output_df.nlargest(5, 'final_deepfake_score')
    for _, row in top_scores.iterrows():
        print(f"Resource ID: {row['resource_id']}, Score: {row['final_deepfake_score']:.4f}, "
              f"Audio: {row['audio_file']}, Video: {row['video_file']}")

if __name__ == "__main__":
    calculate_final_deepfake_score()
