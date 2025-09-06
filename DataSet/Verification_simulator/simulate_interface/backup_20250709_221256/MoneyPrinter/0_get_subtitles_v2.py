from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.MoneyPrinter import ENV_DIR
from Inspection.adapters.custom_adapters.MoneyPrinter import *
exe = Executor('MoneyPrinter', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 可能需要手动修改的部分：
# 设置视频生成的参数
data = {
    'paragraphNumber': 1,
    'aiModel': 'gpt-3.5-turbo',
    'threads': 2,
    'subtitlesPosition': 'bottom',
    'color': '#FFFF00',
    'useMusic': True,
    'automateYoutubeUpload': False,
    'zipUrl': None,
    'videoSubject': 'The Future of AI',
    'customPrompt': 'Discuss the advancements in AI technology.',
    'voice': 'en_us_001'
}
# end

# 导入原有的包
import os
from utils import *
from dotenv import load_dotenv
from gpt import *
from video import *
from search import *
from uuid import uuid4
from tiktokvoice import *
from flask_cors import CORS
from termcolor import colored
from youtube import upload_video
from apiclient.errors import HttpError
from flask import Flask
from flask import request
from flask import jsonify
from moviepy.config import change_settings

# end

load_dotenv(os.path.join(ENV_DIR, '.env'))
check_env_vars()
SESSION_ID = os.getenv('TIKTOK_SESSION_ID')
openai_api_key = os.getenv('OPENAI_API_KEY')
change_settings({'IMAGEMAGICK_BINARY': os.getenv('IMAGEMAGICK_BINARY')})
AMOUNT_OF_STOCK_VIDEOS = 5
GENERATING = False

def generate_video_process(data):
    global GENERATING
    GENERATING = True
    clean_dir(os.path.join(ENV_DIR, 'temp/'))
    clean_dir(os.path.join(ENV_DIR, 'subtitles/'))
    paragraph_number = int(data.get('paragraphNumber', 1))
    ai_model = data.get('aiModel')
    n_threads = data.get('threads')
    subtitles_position = data.get('subtitlesPosition')
    text_color = data.get('color')
    use_music = data.get('useMusic', False)
    automate_youtube_upload = data.get('automateYoutubeUpload', False)
    songs_zip_url = data.get('zipUrl')
    
    if use_music:
        if songs_zip_url:
            fetch_songs(songs_zip_url)
        else:
            fetch_songs('https://filebin.net/2avx134kdibc4c3q/drive-download-20240209T180019Z-001.zip')
    
    print(colored('[Video to be generated]', 'blue'))
    print(colored('   Subject: ' + data['videoSubject'], 'blue'))
    print(colored('   AI Model: ' + ai_model, 'blue'))
    print(colored('   Custom Prompt: ' + data['customPrompt'], 'blue'))
    
    if not GENERATING:
        return {'status': 'error', 'message': 'Video generation was cancelled.', 'data': []}
    
    voice = data['voice']
    voice_prefix = voice[:2]
    if not voice:
        print(colored('[!] No voice was selected. Defaulting to "en_us_001"', 'yellow'))
        voice = 'en_us_001'
        voice_prefix = voice[:2]
    
    script = exe.run('generate_script', video_subject=data['videoSubject'], paragraph_number=paragraph_number, ai_model=ai_model, voice=voice, custom_prompt=data['customPrompt'])
    search_terms = exe.run('get_search_terms', video_subject=data['videoSubject'], amount_of_stock_videos=AMOUNT_OF_STOCK_VIDEOS, script=script, ai_model=ai_model)
    video_urls = []
    it = 15
    min_dur = 10
    
    for search_term in search_terms:
        if not GENERATING:
            return {'status': 'error', 'message': 'Video generation was cancelled.', 'data': []}
        found_urls = exe.run('search_for_stock_videos', search_term=search_term, pexels_api_key=os.getenv('PEXELS_API_KEY'), it=it, min_dur=min_dur)
        for url in found_urls:
            if url not in video_urls:
                video_urls.append(url)
                break
    
    if not video_urls:
        print(colored('[-] No videos found to download.', 'red'))
        return {'status': 'error', 'message': 'No videos found to download.', 'data': []}
    
    video_paths = []
    print(colored(f'[+] Downloading {len(video_urls)} videos...', 'blue'))
    
    for video_url in video_urls:
        if not GENERATING:
            return {'status': 'error', 'message': 'Video generation was cancelled.', 'data': []}
        try:
            saved_video_path = exe.run('save_video', video_url=video_url)
            video_paths.append(saved_video_path)
        except Exception:
            print(colored(f'[-] Could not download video: {video_url}', 'red'))
    
    print(colored('[+] Videos downloaded!', 'green'))
    print(colored('[+] Script generated!\n', 'green'))
    
    if not GENERATING:
        return {'status': 'error', 'message': 'Video generation was cancelled.', 'data': []}
    
    sentences = script.split('. ')
    sentences = list(filter(lambda x: x != '', sentences))
    paths = []
    
    for sentence in sentences:
        if not GENERATING:
            return {'status': 'error', 'message': 'Video generation was cancelled.', 'data': []}
        current_tts_path = os.path.join(ENV_DIR, 'temp', f'{uuid4()}.mp3')
        exe.run('tts', sentence=sentence, voice=voice, filename=current_tts_path)
        audio_clip = AudioFileClip(current_tts_path)
        paths.append(audio_clip)
    
    final_audio = concatenate_audioclips(paths)
    tts_path = os.path.join(ENV_DIR, 'temp', f'{uuid4()}.mp3')
    final_audio.write_audiofile(tts_path)
    
    try:
        subtitles_path = exe.run('generate_subtitles', audio_path=tts_path, sentences=sentences, audio_clips=paths, voice=voice_prefix)
    except Exception as e:
        print(colored(f'[-] Error generating subtitles: {e}', 'red'))
        subtitles_path = None
    
    temp_audio = AudioFileClip(tts_path)
    combined_video_path = exe.run('combine_videos', video_paths=video_paths, max_duration=temp_audio.duration, max_clip_duration=5, threads=n_threads or 2)
    
    try:
        final_video_path = exe.run('generate_video', combined_video_path=combined_video_path, tts_path=tts_path, subtitles_path=subtitles_path, threads=n_threads or 2, subtitles_position=subtitles_position, text_color=text_color or '#FFFF00')
    except Exception as e:
        print(colored(f'[-] Error generating final video: {e}', 'red'))
        final_video_path = None
    
    title, description, keywords = exe.run('generate_metadata', video_subject=data['videoSubject'], script=script, ai_model=ai_model)
    print(colored('[-] Metadata for YouTube upload:', 'blue'))
    print(colored('   Title: ', 'blue'))
    print(colored(f'   {title}', 'blue'))
    print(colored('   Description: ', 'blue'))
    print(colored(f'   {description}', 'blue'))
    print(colored('   Keywords: ', 'blue'))
    print(colored(f"  {', '.join(keywords)}", 'blue'))
    
    if automate_youtube_upload:
        client_secrets_file = os.path.abspath('./client_secret.json')
        SKIP_YT_UPLOAD = False
        if not os.path.exists(client_secrets_file):
            SKIP_YT_UPLOAD = True
            print(colored('[-] Client secrets file missing. YouTube upload will be skipped.', 'yellow'))
            print(colored('[-] Please download the client_secret.json from Google Cloud Platform and store this inside the /Backend directory.', 'red'))
        
        if not SKIP_YT_UPLOAD:
            video_category_id = '28'
            privacyStatus = 'private'
            video_metadata = {
                'video_path': os.path.abspath(os.path.join(FILE_RECORD_PATH, final_video_path)),
                'title': title,
                'description': description,
                'category': video_category_id,
                'keywords': ','.join(keywords),
                'privacyStatus': privacyStatus
            }
            try:
                video_response = upload_video(video_path=video_metadata['video_path'], title=video_metadata['title'], description=video_metadata['description'], category=video_metadata['category'], keywords=video_metadata['keywords'], privacy_status=video_metadata['privacyStatus'])
                print(f"Uploaded video ID: {video_response.get('id')}")
            except HttpError as e:
                print(f'An HTTP error {e.resp.status} occurred:\n{e.content}')
    
    video_clip = VideoFileClip(os.path.join(FILE_RECORD_PATH, final_video_path))
    
    if use_music:
        song_path = choose_random_song()
        original_duration = video_clip.duration
        original_audio = video_clip.audio
        song_clip = AudioFileClip(song_path).set_fps(44100)
        song_clip = song_clip.volumex(0.1).set_fps(44100)
        comp_audio = CompositeAudioClip([original_audio, song_clip])
        video_clip = video_clip.set_audio(comp_audio)
        video_clip = video_clip.set_fps(30)
        video_clip = video_clip.set_duration(original_duration)
        video_clip.write_videofile(os.path.join(FILE_RECORD_PATH, final_video_path), threads=n_threads or 1)
    else:
        video_clip.write_videofile(os.path.join(FILE_RECORD_PATH, final_video_path), threads=n_threads or 1)
    
    print(colored(f'[+] Video generated: {final_video_path}!', 'green'))
    
    if os.name == 'nt':
        os.system('taskkill /f /im ffmpeg.exe')
    else:
        os.system('pkill -f ffmpeg')
    
    GENERATING = False
    return {'status': 'success', 'message': 'Video generated!', 'data': final_video_path}

# 直接运行主逻辑
result = generate_video_process(data)
print(result)
