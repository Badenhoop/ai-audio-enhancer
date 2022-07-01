from argparse import ArgumentParser
from matplotlib.pyplot import title
from pytube import Search
import os
from uuid import uuid4
import pandas as pd
from tqdm import tqdm
import logging


class InsufficientResults(Exception):
    pass


def download_youtube_audio(vid, path):
    stream = vid.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').asc().first()
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    info = dict(audio_bitrate=stream.abr)
    stream.download(output_path=directory, filename=filename)
    return info


def check_vid_title(title, artist, vid):
    vid_title = vid.title.lower()
    return title in vid_title and artist in vid_title and 'cover' in vid_title


def search_cover_songs(title, artist, max_num_results=10, strict=True):
    assert max_num_results <= 20, 'Querying more than 20 results not supported yet.'
    s = Search(f'{title} {artist} cover')
    results = s.results
    if strict:
        results = [vid for vid in results if check_vid_title(title, artist, vid)]
    results = results[:min(len(s.results), max_num_results)]
    return results


def download_cover_songs(title, 
                         artist, 
                         root_dir, 
                         max_length=600, 
                         min_num_results=5,
                         max_num_results=10):
    vids = search_cover_songs(title, artist, max_num_results=max_num_results)
    if len(vids) < min_num_results:
        raise InsufficientResults(f'Found {len(vids)} videos for title "{title}" and artist "{artist}" which is below the minimum number of videos ({min_num_results}).')

    infos = []
    for i, vid in enumerate(vids):
        if vid.length > max_length:
            continue
        id = str(uuid4())
        rel_path = os.path.join(artist, title, f'{id}.mp4')
        download_path = os.path.join(root_dir, rel_path)
        info = download_youtube_audio(vid, download_path)
        info |= dict(
            id=id,
            title=title,
            artist=artist,
            video_title=vid.title,
            url=vid.watch_url,
            length=vid.length,
            views=vid.views,
            result_index=i,
            path=rel_path)
        infos.append(info)

    return infos


def download_cover_song_dataset(download_songs_csv, 
                                processed_songs_csv, 
                                dataset_csv, 
                                root_dir):
    download_songs_df = pd.read_csv(download_songs_csv)

    processed_songs_df = pd.read_csv(processed_songs_csv) \
        if os.path.exists(processed_songs_csv) \
        else pd.DataFrame([], columns=['title', 'artist', 'success'])
    processed_songs = processed_songs_df.to_dict('records')
    
    dataset = pd.read_csv(dataset_csv).to_dict('records') \
        if os.path.exist(dataset_csv) \
        else []

    # Filter out songs that have already been processed.
    download_songs_df = download_songs_df[
        ~download_songs_df.set_index(['title', 'artist'].index.isin(processed_songs_df.set_index(['title', 'artist']).index))
    ]

    download_songs = list(download_songs_df.itertuples())
    success_counter = 0
    for i, song in enumerate(tqdm(download_songs)):
        logging.debug(f'Iteration {i+1}/{len(download_songs)}: Looking at song {song}.')
        success = False
        try:
            infos = download_cover_songs(
                title=song.title,
                artist=song.artist,
                root_dir=root_dir,
                max_length=600,
                min_num_results=5,
                max_num_results=10)
            dataset.extend(infos)
            success = True
            logging.info('Successfully downloaded song!')
        except InsufficientResults as e:
            logging.warning(e)

        if success:
            if success_counter % 10 == 0:
                pd.DataFrame(dataset).to_csv(dataset_csv, index=False)
            success_counter += 1

        processed_songs.append(dict(
            title=song.title,
            artist=song.artist,
            success=success))
        if i % 10 == 0:
            pd.DataFrame(processed_songs).to_csv(processed_songs_csv, index=False)

    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv(dataset_csv, index=False)
    return dataset_df


def main(args):
    logging.basicConfig(filename='downloader.log', encoding='utf-8', level=logging.DEBUG)
    logging.debug(f'Arguments: {args}')
    download_cover_song_dataset(
        download_songs_csv=args.songs_csv,
        processed_songs_csv=args.processed_songs_csv,
        dataset_csv=args.dataset_csv,
        root_dir=args.root_dir)


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a dataset to train DiffWave')
    parser.add_argument('download-songs-csv', type=str, help='csv file of the songs to download.')
    parser.add_argument('processed-songs-csv', type=str, help='csv file that contains the already processed songs.')
    parser.add_argument('dataset-csv', type=str, help='csv file of the dataset.')
    parser.add_argument('root-dir', type=str, help='Root directory of the downloaded content.')
    main(parser.parse_args())