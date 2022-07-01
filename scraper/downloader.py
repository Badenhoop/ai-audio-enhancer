from argparse import ArgumentParser
from pytube import Search, Stream
import os
from uuid import uuid4
import pandas as pd
from tqdm import tqdm
import logging


logger = logging.getLogger('downloader')


class InsufficientResults(Exception):
    pass


def download_stream(stream, path):
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    info = dict(
        bitrate=stream.bitrate,
        audio_codec=stream.audio_codec,
        filesize=stream.filesize)
    stream.download(output_path=directory, filename=filename)
    return info


def get_videos_with_valid_stream(vids):
    result_vids = []
    result_streams = []
    for vid in vids:
        try:
            streams = vid.streams.filter(only_audio=True, file_extension='mp4')
            if len(streams) == 0:
                continue
            stream = streams.order_by('bitrate').asc().first()
            result_vids.append(vid)
            result_streams.append(stream)
        except KeyError as a:
            # There seems to be a bug in pytube in which it tries to reference 
            # the attribute 'bitrate' even though it does not exist in the 
            # method streams.filter().
            logger.error(a)
    return result_vids, result_streams


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
    vids, streams = get_videos_with_valid_stream(vids)
    if len(vids) < min_num_results:
        raise InsufficientResults(f'Found {len(vids)} videos for title "{title}" and artist "{artist}" which is below the minimum number of videos ({min_num_results}).')

    infos = []
    for i, (vid, stream) in enumerate(zip(vids, streams)):
        if vid.length > max_length:
            continue
        id = str(uuid4())
        rel_path = os.path.join(artist, title, f'{id}.mp4')
        download_path = os.path.join(root_dir, rel_path)
        info = download_stream(stream, download_path)
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
        if os.path.exists(dataset_csv) \
        else []

    # Filter out songs that have already been processed.
    download_songs_df = download_songs_df[
        ~download_songs_df.set_index(['title', 'artist']).index.isin(processed_songs_df.set_index(['title', 'artist']).index)
    ]

    download_songs = list(download_songs_df.itertuples())
    success_counter = 0
    for i, song in enumerate(tqdm(download_songs)):
        logger.info(f'Iteration {i+1}/{len(download_songs)}: Looking at song {song}.')
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
            logger.info('Successfully downloaded song!')
        except InsufficientResults as e:
            logger.warning(e)

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
    logger.info(f'Program arguments: {args}')
    download_cover_song_dataset(
        download_songs_csv=args.download_songs_csv,
        processed_songs_csv=args.processed_songs_csv,
        dataset_csv=args.dataset_csv,
        root_dir=args.root_dir)


if __name__ == '__main__':
    logging.basicConfig(
        filename='downloader.log', 
        filemode='w', 
        encoding='utf-8',
        level=logging.INFO)

    parser = ArgumentParser(description='Downloads cover songs from YouTube.')
    parser.add_argument('download_songs_csv', type=str, help='csv file of the songs to download.')
    parser.add_argument('processed_songs_csv', type=str, help='csv file that contains the already processed songs.')
    parser.add_argument('dataset_csv', type=str, help='csv file of the dataset.')
    parser.add_argument('root_dir', type=str, help='Root directory of the downloaded content.')
    main(parser.parse_args())