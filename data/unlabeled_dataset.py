from dataset.data_processing_pipeline import *
from dataset.data_rough_processing import get_processed_df


def get_dataloaders(batch_size, data_path="dataset/biden_tweets_clean.csv", text=False):
    data_df = get_processed_df(data_path=data_path)
    data_df = data_df.drop_duplicates()
    pipeline = DataProcessingPipeline([
        replace_emoji_with_text,
        ReplaceWords(),
    ])
    data_df["Text"] = data_df["Text"].apply(pipeline)

