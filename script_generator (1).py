import json
from newsapi import NewsApiClient
from datetime import date
from openai import OpenAI
import os
from dotenv import load_dotenv
import requests 
from moviepy.editor import VideoFileClip, concatenate_videoclips
import urllib.request
import cv2



os.environ['NEWS_API']='f07498d0a13f44d6b1c63c041ed0f3f4'
os.environ['OPENAI_API_KEY']='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'
os.environ['MEDIUM_API']='251d8010a4eb4cb279c5d2d44ffe85f4fbe5614230bb2f6d129cd3e21c722d98f'

def save_file(filename, content):
    '''
    Saves a file in the 'articles' directory
    Args:
        filename (str): the name of the new file
        content (str): the content of the file. 
    '''

    file = open('articles/' + filename, "w")    # open the new file in writing mode
    file.write(content)
    file.close()


def get_author_id(token):
    '''
    Uses the medium api to get the user's author id
    
    Args:
        token (str): Medium's integration token
    
    '''


    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Charset": "utf-8",
    }

    response = requests.get("https://api.medium.com/v1/me", headers=headers, params={"Authorization": "Bearer {}".format(token)})

    if response.status_code == 200:
		# if response is OK, return the authorId 
        return response.json()['data']['id']
    else:
        print('Failed to Retrieve User Id. Status code: ', response.status_code)
        print('Response: ', response.text)
        return None


def push_to_medium(
    file_to_upload,
    medium_id,
    token,
    title,
    tag_list,
    publish_status="draft",
    content_format="html",
):
    """
    Push an HTML file to Medium as a draft post.

    Args:
        file_to_upload (str): Path to the HTML file to be uploaded.
        id (str): User ID for Medium.
        token (str): Medium API token.
    """

    # Error control 
    if len(tag_list) > 5:
        raise ValueError("Tag list should not contain more than 5 elements.")
    
    # Open html file to be uploaded 
    with open(file_to_upload, "r") as content_text:
        content = content_text.read()

    # Prepare https request
    url = f"https://api.medium.com/v1/users/{medium_id}/posts"

    post_data = {
        "title": title,
        "contentFormat": content_format,
        "content": content,
        "tags": tag_list,
        "publishStatus": publish_status,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Charset": "utf-8",
    }

    response = requests.post(url, headers=headers, json=post_data)

    if response.status_code == 201:
        post_details = response.json()
        print("Draft Post Created Successfully:")
        print("Post Details:")
        print(post_details)
    else:
        print("Failed to create draft post. Status code:", response.status_code)
        print("Response:", response.text)



def get_article_info(token, terms):
    """
    Retrieve a list of articles together with a short description from the Google News API 
    Args:
        token (str): News-api token
        terms : list of terms to look for among the articles
    
    """

    # Get today's date a month ago.

    today = date.today()
    month_ago = date(today.year, today.month - 1, today.day) if today.month != 1 else date(today.year - 1, 12, today.day)

    # API KEY SET UP 
    newsapi = NewsApiClient(api_key=token)

    news_results = []
    for term in terms:
        news = newsapi.get_everything(qintitle=term, 
                                from_param=month_ago, 
                                to=today,
                                language='en',
                                sort_by='popularity')  
        for i in range(1, int(news['totalResults']/100) + 2):
            news_results += newsapi.get_everything(qintitle=term, 
                                    from_param=month_ago, 
                                    to=today,
                                    language='en',
                                    page = i,
                                    sort_by='popularity')['articles']



        # retrieve titles of the articles 
        titles = [d['title'] for d in news_results]
        for t in titles:
            print(f"{t}\n")
        article_info = [(d['title'], d['description']) for d in news_results]
        return article_info



def get_article(filename, image_url, article_info):
    """
    Generate an article using ChatGPT-3.5 turbo. 

    Args:
        filename: name of the file where the article should be stored
        image_url (str): url of the image to be included in the article
        article_info (str): list of articles and descriptions retrieved. 
    """

    # Set up openAI environment
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": "You are a senior dermatologist with knowledge in skincare. Do not mention your position. "},
              {"role": "user", "content": f"You need to produce an article about the top beauty and skincare trends. Write the article taking into account the latest news presented in double hashes (##): ## {article_info} ## \
            . There should be a maximum of 5 beauty trends and do not copy the exact words of the given articles. Add a subtitle and a concluding paragraph. The article has to contain 1500 words. Add the following image: {image_url}. The output should be written in HTML format."}
    ]
    )   

    save_file(filename, completion.choices[0].message.content)
    client = OpenAI()
    completion1 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": "You are a senior dermatologist with knowledge in skincare. Do not mention your position. "},
              {"role": "user", "content": f"You need to produce an article about the top beauty and skincare trends. Write the article taking into account the latest news presented in double hashes (##): ## {article_info} ## \
            . There should be a maximum of 5 beauty trends and do not copy the exact words of the given articles. "}
    ]
    )
    return completion1.choices[0].message.content

def get_key_words(article):
    """
    Pulling keywords from the article to be able to generate picture later
    "article":
     article text about skincare
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a profession in text analysing "},
                {"role": "user", "content": f'You need to extract the keywords from an article about the top beauty and skincare trends, that is written in double hashes (##): ## {article} ## Wtite these words with comma , like skincare, body, girl. For example: question: My name is Sofiya, I love eating mushrooms, i think they can help me to make my skin look pretty. Answer:skin, mushrooms, Sofiya. Save the order according to appearance in the text.'}]
    )

    return completion.choices[0].message.content

def get_videos_by_keywords(keywords):
    videos = []
    keywords = keywords.split(',')
    for keyword in keywords:
        result = requests.get(f'https://pixabay.com/api/videos/?key=43538806-bd9860b4a1df3c7e4c56f8005&q={keyword}')
        response = result.content
        videos.append(response)

    new_videos = []
    for video in videos:
        video_json = json.loads(video.decode('utf-8'))
        if video_json['total'] != 0:
            hits = video_json['hits']
            hits = hits[0]
            print(hits)
            url = hits['videos']
            url_real = url['large']['url']
            print(url_real)
            new_videos.append(url_real)

    return new_videos

def clear_directory(directory):
    """
    Clear all files from the specified directory.

    Args:
        directory (str): Directory to be cleared.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
            else:
                print(f"Skipped {file_path} (not a file)")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def save_videos_locally(videos, directory):
    """
    Save videos locally in the specified directory.

    Args:
        videos (list): List of video data to be saved.
        directory (str): Directory where the videos will be saved.
    """
    clear_directory(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, video_data in enumerate(videos):
        filename = os.path.join(directory, f"video_{i}.mp4")
        print(video_data)
        response = requests.get(video_data)
        with open(filename, 'wb') as file:
	        file.write(response.content)

        print(f"Video {i} saved as {filename}")

def merge_videos_from_directory(directory, output_filename):
    """
    Merge multiple videos from a directory into one using OpenCV.

    Args:
        directory (str): Directory containing the videos.
        output_filename (str): Filename for the merged video.
    """
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    video_paths = [os.path.join(directory, f) for f in video_files]

    video_clips = []
    for path in video_paths:
        video_clips.append(cv2.VideoCapture(path))

    # Get the width and height of videos
    width = int(video_clips[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_clips[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_clips[0].get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for clip in video_clips:
        while clip.isOpened():
            ret, frame = clip.read()
            if ret:
                out.write(frame)
            else:
                break

    # Release everything if job is finished
    for clip in video_clips:
        clip.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Merged video saved as '{output_filename}'")



def get_image():

    """
    Generate an image based on 'skincare' using dall-e-2. 

    """

    # Set up OpenAI environment. 
    client = OpenAI()

    response = client.images.generate(
        model="dall-e-2",
        prompt="skincare, beautiful woman makes skincare routine",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    print(f"Generated image is: {response.data[0].url}")
    return response.data[0].url




def main() -> None:

    # Set needed tokens
    load_dotenv()
    news_token = os.getenv('NEWS_API')
    medium_token = os.getenv('MEDIUM_API')
    
    # Retrieve the news information
    terms = ['SKINCARE', 'top beauty hacks', 'beauty trends', 'top beauty', 'top beauty trend', 'popular beauty']
    article_info = get_article_info(news_token,terms)

    # DALL-E - generate image 
    image_url = get_image()

    # GPT-3 - generate article
    filename = 'output.html'
    article = get_article(filename, image_url, article_info)
    keywords = get_key_words(article)
    
    print(keywords)

    videos = get_videos_by_keywords(keywords)
    save_videos_locally(videos, "videos")

    merge_videos_from_directory("videos", "merged_video.mp4")
    # POST TO MEDIUM
    medium_id = get_author_id(medium_token)     # retrieve author id
    
    push_to_medium('articles/' + filename, medium_id, medium_token, 'Top Skincare Trends', ['beauty', 'skincare'])


if __name__ == "__main__":
    main()


