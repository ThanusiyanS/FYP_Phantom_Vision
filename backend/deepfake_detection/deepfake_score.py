import pandas as pd
import os

def calculate_final_deepfake_score():
    """
    Calculate final deepfake score by comparing audio and video scores
    and selecting the higher value for each resource_id
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
    
    # Calculate final deepfake score by selecting the higher value
    df['final_deepfake_score'] = df[['avg_audio_deepfake_score', 'avg_video_deepfake_score']].max(axis=1)
    
    # Create output DataFrame with required columns
    output_df = df[['resource_id', 'audio_file', 'video_file', 'final_deepfake_score']].copy()
    
    # Round the final score to 4 decimal places
    output_df['final_deepfake_score'] = output_df['final_deepfake_score'].round(4)
    
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
