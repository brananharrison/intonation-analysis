import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dynamic_time_warp import map_frequency_vectors
from import_and_analyze_sheet_music import musicxml_to_csv
from analyze_audio import audio_to_json
from audio_json_to_csv import json_to_time_indexed_table
from annotate_music import annotate_sheet_music
from r_and_d.functions import extract_audio_segment


def run_audio_to_json(audio_path):
    audio_to_json(audio_path)

async def async_audio_to_json(audio_path):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, run_audio_to_json, audio_path)


if __name__ == "__main__":
    extract_audio_segment('/Users/branan/Downloads/How to play Twinkle Twinkle on the violin!.wav',
                          '/Users/branan/Downloads/Twinkle_excerpt.wav',
                          40, 95)

    audio_path = "/Users/branan/Downloads/Twinkle_excerpt.wav"
    sheet_music_path = "/Users/branan/Downloads/Twinkle_Twinkle_Little_Star_-_Main_Theme.mxl"


    print("Starting process...")
    overall_start_time = time.time()
    start_time = time.time()
    asyncio.run(async_audio_to_json(audio_path))
    end_time = time.time()
    print(f"Audio to json compute time: {end_time - start_time:.2f} seconds.")


    start_time = time.time()
    json_to_time_indexed_table(json_file_path="exports/audio_json.json",
                               output_csv_path="exports/audio_csv.csv",
                               appearance_threshold=2)
    end_time = time.time()
    print(f"Json to csv compute time: {end_time - start_time:.2f} seconds.")


    start_time = time.time()
    musicxml_to_csv(sheet_music_path)
    end_time = time.time()
    print(f"Sheet music path to csv compute time: {end_time - start_time:.2f} seconds.")


    start_time = time.time()
    map_frequency_vectors(
        audio_csv_path="exports/audio_csv.csv",
        sheet_csv_path="exports/sheet_music_csv.csv",
        exports_dir="exports"
    )
    end_time = time.time()
    print(f"Dynamic time warping compute time: {end_time - start_time:.2f} seconds.")


    start_time = time.time()
    annotate_sheet_music(sheet_music_path,
                         "exports/one_to_one_mapping_to_sheet_music.csv")
    end_time = time.time()
    print(f"Export annotated sheet music compute time: {end_time - start_time:.2f} seconds.")


    overall_end_time = time.time()
    print(f"\nOverall processing time: {overall_end_time - overall_start_time:.2f} seconds.")