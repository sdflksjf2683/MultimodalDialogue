from openai import OpenAI
import pandas as pd
import random
from datetime import datetime, timedelta
import json
import re
  
def get_response(prompt1, prompt2, model_name, api_key):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,  # gpt-3.5-turbo, gpt-4o, gpt-3.5-turbo-0125
        messages=[
            {
                "role": "system",
                "content": f"{prompt1}",
            },
            {
                "role": "user",
                "content": f"{prompt2}",
            },
        ],
        temperature = 0.9,
        max_tokens = 4096,
     )
    return completion.choices[0].message.content

def get_us_full_format_random_dates(start_year=2022, end_year=2024, count=5):
    # 시작 날짜와 종료 날짜를 정의
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    # 두 날짜 사이의 일수 계산
    delta_days = (end_date - start_date).days

    # 랜덤 날짜 생성
    random_dates = []
    for _ in range(count):
        random_days = random.randint(0, delta_days)
        random_date = start_date + timedelta(days=random_days)

        # 랜덤 시간 추가
        random_seconds = random.randint(0, 86399)  # 하루의 초 (24 * 60 * 60 - 1)
        random_time = random_date + timedelta(seconds=random_seconds)
        
        random_dates.append(random_time)
    
    # 시간 순서대로 정렬
    random_dates.sort()

    # 반환값 정리 (미국 날짜 형식으로 변환)
    formatted_dates = [
        date.strftime('%A, %B %d, %Y, %I:%M:%S %p')  # 요일, 월, 일, 연도, 시간 (AM/PM)
        for date in random_dates
    ]
    
    return formatted_dates

def get_relationship(speaker1_persona, speaker2_persona, name , relationship, model_name, api_key):
    prompt1 = f'''You will be provided with the personas of two individuals and the nature of their relationship. Your task is to analyze the personas and their relationship to create a profile for each individual. For each person, include their name, age, nationality, gender, physical description, occupation, and personality traits. When crafting the physical description, describe their appearance solely using adjectives. Use only standalone adjectives (e.g., calm demeanor, slicked-back hair, fair skin, large eyes) and include details such as hairstyle, physical features, and clothing items. Ensure you list nouns like tailored suit, relaxed posture, or leather satchel, avoiding full sentences and sticking to words or phrases. For example, instead of describing with complete sentences, simply provide a list of relevant terms like "tailored suit, relaxed posture, leather satchel." Finally, specify the relationship between the two individuals in detail. The relationship description must be specific, expansive, and fully utilize all provided context.Be sure to write the role in the relationship and personality traits with specificity and detail.
    The names listed below cannot be used as profile names during the creation process. If no specific name is provided, feel free to use any name. You may use a variety of names from around the world. However, if a name appears on the list, it must not be used. Names commonly used in various countries are acceptable
    list : {name}
    The output format should be as follows:
 
    <speaker1's profile>
    -speaker1's name :
    -speaker1's age :
    -speaker1's nationality : 
    -speaker1's gender :
    -speaker1's physical description :
    -speaker1's occupation :
    -speaker1's personality traits :
    -speaker1's role in the relationship
   
    <speaker2's profile>
    -speaker2's name :
    -speaker2's age :
    -speaker2's nationality : 
    -speaker2's gender :
    -speaker2's physical description :
    -speaker2's occupation :
    -speaker2's personality traits :
    -speaker2's role in the relationship
    -relationship between speaker1 and speaker2 : '''
    
    prompt2 = f'''
    -speaker1's persona : 
    {speaker1_persona}
    -speaker2's persona : 
    {speaker2_persona}
    -relationship : 
    {relationship}
    Now, based on the given information, create detailed profiles for the two individuals.
    '''
    result = get_response(prompt1, prompt2, model_name, api_key)
    return result

def process_personas(data: str):
    """
    Processes a persona update string and separates it into three lists:
    1. Updated persona (all lines in Updated Persona section).
    2. Modified Personas (lines in Modified Personas section).
    3. Newly Added Personas (lines in Newly Added Personas section).

    Parameters:
    data (str): String containing updated persona information.

    Returns:
    tuple: A tuple containing three lists (updated_persona, modified_personas, newly_added_personas).
    """
    updated_persona = []
    modified_personas = []
    newly_added_personas = []

    # Split data into lines
    lines = data.splitlines()

    # Flags to determine current section
    in_updated_persona = False
    in_modified_personas = False
    in_newly_added_personas = False

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("- Updated Persona:"):
            in_updated_persona = True
            in_modified_personas = False
            in_newly_added_personas = False
            continue

        if stripped_line.startswith("- Updated Personas Only:"):
            in_updated_persona = False
            in_modified_personas = False
            in_newly_added_personas = False
            continue

        if stripped_line.startswith("- Modified Personas:"):
            in_modified_personas = True
            in_newly_added_personas = False
            continue

        if stripped_line.startswith("- Newly Added Personas:"):
            in_modified_personas = False
            in_newly_added_personas = True
            continue

        if stripped_line.startswith("(") and stripped_line.endswith(")"):
            continue  # Skip parentheses remarks

        # Append lines to the appropriate section
        if in_updated_persona and stripped_line:
            updated_persona.append(stripped_line)
        elif in_modified_personas and stripped_line:
            modified_personas.append(stripped_line)
        elif in_newly_added_personas and stripped_line:
            newly_added_personas.append(stripped_line)

    updated_persona = [item.strip("- ").strip().strip('\"') for item in updated_persona]
    modified_personas = [item.strip("- ").strip().strip('\"') for item in modified_personas]
    newly_added_personas = [item.strip("- ").strip().strip('\"') for item in newly_added_personas]

    updated_persona = [] if None in updated_persona else updated_persona
    modified_personas = [] if None in modified_personas else modified_personas
    newly_added_personas = [] if None in newly_added_personas else newly_added_personas

    updated_persona = [] if 'None' in updated_persona else updated_persona
    modified_personas = [] if 'None' in modified_personas else modified_personas
    newly_added_personas = [] if 'None' in newly_added_personas else newly_added_personas

    return updated_persona, modified_personas, newly_added_personas

def transform_single_dialogue(dialogue):
    dialogues = dialogue.split("\n\n")
    transformed_dialogues = []
    for turn_index, dialogue in enumerate(dialogues):
        if '<turn>' in dialogue or '<Turn>' in dialogue:
            print(dialogue)


            speaker, image = dialogue.split(":", 1)
            image = image.split("<")  # '<' 기준으로 나누기
            image = [item.rstrip(">") for item in image if item]
            transformed_dialogues.append({
                "turn": f"Turn {turn_index + 1}",
                "speaker": speaker.strip(),
                'people' : image[1],
                'day' : image[2],
                "photo_description": image[3].strip(),
                "photo_url": None 
            })
        
        
        elif ":" in dialogue:
            speaker, text = dialogue.split(":", 1)
            transformed_dialogues.append({
                "turn": f"Turn {turn_index + 1}",
                "speaker": speaker.strip(),
                "text": text.strip(),
                "photo": None
            })

    
    return transformed_dialogues

def persona_update(speaker, speaker_persona, dialogue ,model_name, api_key):
    prompt1 = f'''You will be provided with a persona of an individual and a dialogue between two people. Based on this, your task is to update the given persona about {speaker}. Follow the instructions below to perform the updates. 
Instructions:
1. If the persona extracted from the dialogue completely overlaps with the existing persona, use the existing persona as is.
2. If the persona extracted from the dialogue is a subset of the existing persona, use the persona extracted from the dialogue.
3. If the persona extracted from the dialogue can make the existing persona more specific, combine the two personas and use the result.
4. If the persona is significantly unrelated to the topic of conversation or feels unnecessary, exclude it.
5. If there are personas with similar themes or overlapping content, combine them to create a new persona.
6. If any temporal shifts are observed between the previous and current session persona, they should be seamlessly integrated into the persona
7. If none of the above apply, add the new persona.

The output format should be as follows: the updated persona should include the results after the updates, while the updated personas only should only include the personas that were modified or newly added

- Updated Persona:
  (Include the complete persona list after updates.)

- Updated Personas Only:
  - Modified Personas:
    (List personas that were changed.)
  - Newly Added Personas:
    (List personas that were added.)'''

    prompt2 = f'''
{speaker}'s persona: {speaker_persona}

dialogue: {dialogue}

- Updated Persona :'''
    result = get_response(prompt1, prompt2, model_name, api_key)
    return result
    
def summarize(speaker1, speaker2, speaker1_persona, speaker2_persona, speaker1_information, speaker2_information ,relationship, two_relationship ,date ,dialogue, model_name, api_key):
    prompt1 = f'''You will be given the personas of {speaker1} and {speaker2}, information from the previous session, date, their relationship, and profile details. Based on this, create a summary. The summary should be as rich in detail as possible, reflecting as much information as possible. The information about animal species or people must not change, so you always need to keep track of the details about people mentioned during the conversation.
    dialogue : {dialogue}
    relationship : {relationship}
    relationship_specify: {two_relationship}    
    date : {date}
    <{speaker1_information['name']}'s profile>
    age : {speaker1_information['age']} 
    nationality : {speaker1_information['nationality']} 
    gender : {speaker1_information['gender']}
    occupation : {speaker1_information['occupation']} 
    role in relationship : {speaker1_information['role']}
    persona : {speaker1_persona}

    <{speaker2_information['name']}'s profile>
    age : {speaker2_information['age']} 
    nationality : {speaker2_information['nationality']} 
    gender : {speaker2_information['gender']}
    occupation : {speaker2_information['occupation']} 
    role in relationship : {speaker2_information['role']}
    persona : {speaker2_persona}'''
    prompt2 = f'''summary :
    '''

    summary = get_response(prompt1, prompt2, model_name, api_key)
    return summary

def choose_persona(personas, model_name, api_key):
    prompt1 = f'''You will be provided with multiple personas. Your task is to select five personas that are free of contradictions. The selected personas must not have any logical overlaps or contradictions. Redundant information is also not allowed. For example, a person can have only one piece of information about their job or place of residence, but they may have multiple hobbies or preferences.
The output format should be as follows.
-persona1
-persona2
-persona3
-persona4
-persona5'''

    prompt2 = f'''Now, select five personas ensuring logical consistency.    
multimple personas: 
{personas}
output:'''

    chosen_persona = get_response(prompt1, prompt2, model_name, api_key)

    persona_list = []
    for line in chosen_persona.strip().split("\n"):
        persona_value = line.split(":", 1)[1].strip()
        persona_list.append(persona_value)


    prompt1 = f'''You are provided with five personas. Your task is to expand and further refine these personas with greater detail. Follow the instructions below to complete the expansion.
[Instructions]
- Create five personas for a single individual. Ensure that these personas are consistent and logically coherent. For example, the place of residence and workplace should be within the same country, while hobbies and interests can reflect diverse cultural backgrounds. Expand the personas with consideration of time and space to maintain plausibility..
- Background or Setting: Include specific locations or landmarks to ground the persona in a tangible context, explaining their significance.
- Personal Traits: Elaborate on their personality with examples that highlight their habits and preferences in a relatable way.
- Actions or Activities: Provide context for activities, specifying where, how, and why they occur.
- Preferences or Interests: Use specific brands, landmarks, or cultural elements to make interests vivid and personal.
- Conciseness: Focus on the most relevant and vivid details without adding unnecessary information.
- All sentences must contribute to a coherent, logically connected persona without contradictions, ensuring the profile is vivid, realistic, and relatable.

The output format should be as follows.
output:
- revised_persona1
- revised_persona2
- revised_persona3
- revised_persona4
- revised_persona5

Example
personas:
["I currently reside in the country to our north.", "Sometimes I feel depressed.","My beard is my best attribute.","I am broke.","I defend the goal."]
output:
- I currently reside in Canada, specifically in Montreal, Quebec. I enjoy exploring the historic Old Port area, especially during the winter when the streets are lit with festive decorations.
- I sometimes feel depressed, but I’ve found comfort in journaling at a quiet corner of my local library. Writing helps me process my emotions and reflect on the positives in life.
- My beard is my best attribute, and I take pride in grooming it with handmade oils from a local artisan in Portland. People often compliment its shine and fullness.
- I am broke but resourceful, always finding creative ways to save money. Recently, I started cooking all my meals at home, experimenting with budget-friendly recipes like lentil curry.
- I defend the goal as a semi-professional soccer goalkeeper. The adrenaline rush of a crucial save during weekend matches at our city’s stadium keeps me driven.'''
    prompt2 = f'''Read the persona logically and expand it appropriately, ensuring alignment with the cultural context.
personas: 
{persona_list}
output:'''
    
    personas = get_response(prompt1, prompt2, model_name, api_key)
    return personas     

def sesssion_n_dialogue(speaker1, speaker2, speaker1_persona, speaker2_persona,speaker1_information, speaker2_information,summary, date , relationship, two_relationship ,model_name, api_key):
    prompt1  =f'''Your task is to create a conversation between {speaker1} and {speaker2} based on their personas, the time, the day of the week, their relationship, summary and their profile, consisting of 15 turns. The conversation should feel natural and realistic, reflecting the dynamics of their relationship. They can exchange photos during the dialogue, and the content should align with the provided instructions.
<Dialogue Instruction>
1. The conversation revolves around creating events and discussing the events that have occurred.
2. Instead of using all personas, focus on creatively incorporating the given information—such as time, profile, relationships, and personas—to enrich the dialogue and make it more dynamic
3. Expand the conversation topic based on the personas, creating new personas and incorporating them into the dialogue when needed. It's also acceptable to explore themes of everyday life instead of strictly adhering to personas.
4. Make the dialogue reflect the characters' relationship through tone, accent, and realistic interactions. Clearly convey their roles and dynamic, ensuring the relationship is unmistakable from their communication. Use unique expressions and nuances to make it authentic and contextually clear. Explicitly state the nature of the relationship if needed for clarity
5. To make the dialogue engaging, incorporate these seven criteria that enhance the fun and enjoyment of a conversation:
    - Empathy and Connection
        - Shared interests: Weave in moments where characters discover common ground, making the conversation more lively.
        - Emotional empathy: Show genuine understanding of one another's feelings.
    - Humor
        - Lighthearted jokes: Add witty remarks or one-liners to energize the dialogue.
        - Situational humor: Let natural comedy arise from unexpected circumstances.
    - Balanced Tension
        - Building curiosity: Reveal information in steps to pique interest.
        - Playful teasing: Use gentle banter or harmless pranks to add a fun edge.
    - Unexpected Developments
        - Surprising twists: Keep the dialogue fresh with sudden turns or revelations.
        - Out-of-the-box topics: Introduce imaginative ideas that prompt a laugh or double-take.
    - Active Participation
        - Genuine reactions: Characters should respond with clear enthusiasm or shock.
        - Interactive back-and-forth: Encourage questions and follow-up to keep the flow natural.
    - Charming Expressions
        - Clever wordplay or vivid descriptions: Use creative language to paint a picture.
        - Expressive gestures or facial cues: Highlight physical reactions to amplify humor or empathy.
    - Closeness and Trust
        - Familiar “best friend” vibe: Display comfort in each other's humor and habits.
        - Mutual support: Foster an environment where sharing is easy, with no fear of judgment. 
6. Whenever a character appears in dialogue, always include their name. Create the conversation primarily using first names.

<image description instruction>
1. The image descriptions represent photos stored on each individual's phone, which might include landscapes or portraits, along with descriptions of the pictures. The format is: <turn><landscapes or people's names(specifically, include the name of individuals in portraits)> <the date when the photo was taken> <image description>. 
When adding people, include words in the image description that indicate their relationship to the speaker. If the photo is of the sender, Add "me" to the name of the person field.
Examples: 
<turn><me><2024-08-12 6:45 PM><A peaceful photo of me sitting on a wooden bench by the lake, watching the sunset with soft ripples on the water>
<turn><me, Liam><2023-07-15 11:30 AM><A warm family photo taken during our summer vacation at the beach, featuring me, Liam, the father>
<turn><me, Sarah, and David><2023-03-14 3:15 PM><A fun group photo from our trip to the amusement park, all of us holding cotton candy. Sarah is my friend, and David is my boyfriend.>
2. During the conversation, participants should carefully consider whether the photos are landscapes or portraits when responding. During conversations involving photos, the following examples can also be applied: 1. Including a description in the conversation that is only possible through the image, 2. Adding inferences or discoveries that can be deduced from viewing the image, and 3. Incorporating interactions or requests based on the image.
3. For photo-sharing turns, include the word "turn" followed by the description of the shared photo.
4. Based on the given persona, identify the type of person who frequently shares photos and tailor the photos you send to match their personality. Additionally, participants should ensure an equal balance between the number of landscape and portrait photos when sharing images.
5. When writing the image descriptions for the people in the photo, ensure their details are explicitly included in the image description, even if they are not clearly mentioned in the conversation. This is mandatory and should be followed strictly.
6. The name of the person in the photo must be included without exception. It is not permitted to use nouns such as "parents," "people," "mom," "dad," or "partner" instead of specific names, and this rule must be strictly adhered to 
7. In cases such as Relational Nouns, Referential Nouns, or Terms of Address, which are not specific names but rather forms of address, they must not be used in the name field and should instead be included in the image description
The following are examples of incorrect usage.
Example : 
Alex:<turn><people><2022-09-02 4:30 PM><A friendly group photo from the language exchange meetup at the Austin Public Library, with new friends Jorge, Mei, and Hana, all smiling in front of a bookshelf.>


<photo instruction>
Photos should be shared naturally and seamlessly, just like in real human interactions. Each speaker may send only 2-3 photos during the conversation. They don't need to be tied to specific requests or triggers but can be used to shift topics, highlight themes, or draw attention to a subject. As long as it fits the flow of the conversation, sharing photos without explicitly mentioning that a photo will be sent is perfectly acceptable. For instance, photos can be used to make requests, share experiences, evoke emotions, reminisce, or prompt compliments.
Use the following five intents to guide photo-sharing behavior:
- Information Dissemination: Sharing images like infographics or educational material to convey important information or educate.
- Social Bonding: Sharing personal photos or memories to strengthen relationships and deepen connections.
- Humor and Entertainment: Sharing funny pictures, memes, or entertaining visuals to bring joy and lighten the mood.
- Visual Clarification: Using images such as diagrams, item-specific photos, or location pictures to explain or clarify concepts or situations.
- Expression of Emotion or Opinion: Sharing emotive photos or art to succinctly express feelings, perspectives, or opinions.

If the conversation with each person is completed, separate it with two line breaks \n\n. Similarly, for the photo turn, proceed and separate it with two line breaks \n\n. And based on the example below, carry out the conversation accurately, especially ensuring that the explanation for the turn involving sending a photo is written precisely in the correct format.
<Example> 
Liam: Hey Livy, how's your morning been? Are you out in the garden again, tending to your little green kingdom?\n\nOlivia: Ace, you know me too well. Just finished watering the hydrangeas. They're blooming beautifully this year. I should show you—\n\nOlivia:<turn><landscape><2022-03-18 7:45 AM><A vibrant shot of blooming hydrangeas with morning sunlight filtering through.>\n\nLiam: Wow, Livy, those are stunning! The sunlight makes them look almost magical. Your green thumb amazes me every time.\n\nOlivia: Thank you, darling. Oh, and here's something you'll love—Emily was over this morning and insisted on grabbing coffee before helping me with the garden.\n\nOlivia:<turn><me,Emily><2022-03-18 10:15 AM><A candid photo of Olivia’s daughter, Emily, and me, smiling brightly while holding cups of coffee at a cozy café.>\n\nLiam: She looks so grown-up! Time really flies. You must be proud of her. And speaking of memories, I found this gem while sorting through my photos. Vermont feels like a lifetime ago now.\n\nLiam:<turn><landscape><2021-10-12 2:30 PM><A cozy countryside scene with a wooden cabin, a fireplace chimney puffing smoke, and autumn leaves carpeting the ground.>\n\nOlivia: That trip was unforgettable. The autumn leaves, the crisp air, and your obsession with finding the perfect maple syrup—it's one for the books. Speaking of cherished memories, let me share this one. It's from last month when Emily, James, and I went hiking. James brought his camera, so we captured a few gems.\n\nOlivia:<turn><Emily,James,me><2024-12-14 9:20 AM><A lively group photo of Emily, James, and me standing on a rocky trail, surrounded by towering pine trees, with a snow-capped mountain in the background.>'''
    prompt2 = f'''
    Now, based on the above instructions, generate a conversation. When reviewing the conversation, if a relevant turn involves a portrait, the name must always be specified rather than left unspecified. For example, instead of using collective nouns like "family," "partner," or "kittens," you must include the names of every individual in the group. If there is a group among the people included in the photo, list the names of all individuals in the group. and 

    date : {date}
    relationship : {relationship}
    relationship_specify: {two_relationship}

    <{speaker1_information['name']}'s profile>
    age : {speaker1_information['age']} 
    nationality : {speaker1_information['nationality']} 
    gender : {speaker1_information['gender']}
    occupation : {speaker1_information['occupation']} 
    role in relationship : {speaker1_information['role']}
    persona : {speaker1_persona}

    <{speaker2_information['name']}'s profile>
    age : {speaker2_information['age']} 
    nationality : {speaker2_information['nationality']} 
    gender : {speaker2_information['gender']}
    occupation : {speaker2_information['occupation']} 
    role in relationship : {speaker2_information['role']}
    persona : {speaker2_persona}

    summary : {summary}
    dialogue :'''
    dialogue = get_response(prompt1, prompt2, model_name, api_key)
    
    return dialogue

def parse_speaker_information(result):
    def extract_information(start_index):
        """Extracts information for a speaker from the result list starting at start_index."""
        keys = [
            'name', 'age', 'nationality', 'gender',
            'physical_description', 'occupation', 'personality', 'role'
        ]
        return {
            key: result[start_index + i].split(":")[1].strip()
            for i, key in enumerate(keys)
        }

    # Parse information for both speakers
    speaker1_information = extract_information(0)
    speaker2_information = extract_information(8)

    # Parse relationship between the two speakers
    two_relationship = result[16].split(":")[1].strip()

    return speaker1_information, speaker2_information, two_relationship
def process_relationship_text(relationship_text):
    # 한 번에 처리하는 함수
    lines = relationship_text.split('\n')
    result = []
    current_item = ""

    for line in lines:
        if line.startswith('-') or line.startswith('<'):
            # 이전 항목 저장 및 새로운 항목 시작
            if current_item.strip():
                result.append(current_item.strip())
            current_item = line
        else:
            # 현재 항목에 이어 붙이기
            current_item += f" {line}"

    # 마지막 항목 저장
    if current_item.strip():
        result.append(current_item.strip())

    # '-'가 포함된 항목만 필터링
    return [item.strip() for item in result if '-' in item]

def main():
    file_path = '/home/chanho/Model/photo-sharing/pre_code/dataset/relationships_list.txt'

    with open(file_path, 'r') as file:
        relationships_list = [line.strip() for line in file.readlines()]

    API_KEY = "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"
    persona_path = '/home/chanho/Model/photo-sharing/pre_code/dataset/Processed_PersonaChat.json'

    with open(persona_path, 'r', encoding='utf-8') as file:
        persona = json.load(file)['personas']  # JSON 데이터를 Python 객체로 로드
    data_all = []

    name = []

    for i in range(10):
        try:
        ### 정보 입력
            data = {}
            information = {}

            model_name = 'gpt-4o'
                

            speaker1_sampled_persona = random.sample(persona, 30)
            speaker2_sampled_persona = random.sample(persona, 30)

            speaker1_persona = choose_persona(speaker1_sampled_persona,model_name, API_KEY)
            speaker2_persona = choose_persona(speaker2_sampled_persona,model_name, API_KEY)

        #     speaker1_persona = '''
        # - "I like to eat Italian, especially enjoying hand-tossed pizzas and homemade pasta at Tony's Pizza Napoletana in North Beach, which remind me of my travels through Rome."
        # - "I get a buzz cut every two weeks at Joe's Barber Shop in Brooklyn, where the classic style and quick service keep me looking sharp."
        # - "I like fairy tales, often getting lost in the magical worlds created by authors like Hans Christian Andersen and the Brothers Grimm, whose stories fuel my imagination and creativity."
        # - "I was born in Argentina, in the vibrant city of Buenos Aires, where tango music fills the air and the rich cultural heritage has left a lasting impact on me."
        # - "Hiking is one of my favorite pastimes, especially on the scenic trails of Runyon Canyon in Los Angeles, where the panoramic city views invigorate my soul."'''
        #     speaker2_persona = '''
        # - "I have a dyed head, often experimenting with vibrant hues at Rockstar Beauty Salon in San Francisco, where the skillful stylists bring my colorful visions to life."
        # - "I am a marathoner who races biannually, tackling iconic courses like the New York City Marathon to challenge myself and push my limits."
        # - "I like songs that tell stories, especially those by Taylor Swift and Bob Dylan, as their lyrics resonate with my own experiences and emotions."
        # - "I love cats and their babies, often volunteering at the San Francisco SPCA to help care for newborn kittens, whose playful antics and soft purrs bring me immense joy."
        # - "I do not have any pet dogs, but I do enjoy visiting the local animal shelter in Dedham to volunteer and spend time with animals."'''

            date_list = get_us_full_format_random_dates(2022, 2024, 5)

            date = date_list[0]

            relationship = random.choice(relationships_list)
            relationship_text = get_relationship(speaker1_persona, speaker2_persona, name ,relationship, model_name, API_KEY).strip().replace("*", '')

            print(relationship_text)

            result = process_relationship_text(relationship_text)

            speaker1_information, speaker2_information, two_relationship = parse_speaker_information(result)


            prompt1  =f'''Your task is to create a conversation between {speaker1_information['name']} and {speaker2_information['name']} based on their personas, the time, the day of the week, their relationship and their profile, consisting of 15 turns. The conversation should feel natural and realistic, reflecting the dynamics of their relationship. They can exchange photos during the dialogue, and the content should align with the provided instructions.
    <Dialogue Instruction>
    1. The conversation revolves around creating events and discussing the events that have occurred.
    2. Instead of using all personas, focus on creatively incorporating the given information—such as time, profile, relationships, and personas—to enrich the dialogue and make it more dynamic
    3. Expand the conversation topic based on the personas, creating new personas and incorporating them into the dialogue when needed. It's also acceptable to explore themes of everyday life instead of strictly adhering to personas.
    4. Make the dialogue reflect the characters' relationship through tone, accent, and realistic interactions. Clearly convey their roles and dynamic, ensuring the relationship is unmistakable from their communication. Use unique expressions and nuances to make it authentic and contextually clear. Explicitly state the nature of the relationship if needed for clarity
    5. To make the dialogue engaging, incorporate these seven criteria that enhance the fun and enjoyment of a conversation:
        - Empathy and Connection
            - Shared interests: Weave in moments where characters discover common ground, making the conversation more lively.
            - Emotional empathy: Show genuine understanding of one another's feelings.
        - Humor
            - Lighthearted jokes: Add witty remarks or one-liners to energize the dialogue.
            - Situational humor: Let natural comedy arise from unexpected circumstances.
        - Balanced Tension
            - Building curiosity: Reveal information in steps to pique interest.
            - Playful teasing: Use gentle banter or harmless pranks to add a fun edge.
        - Unexpected Developments
            - Surprising twists: Keep the dialogue fresh with sudden turns or revelations.
            - Out-of-the-box topics: Introduce imaginative ideas that prompt a laugh or double-take.
        - Active Participation
            - Genuine reactions: Characters should respond with clear enthusiasm or shock.
            - Interactive back-and-forth: Encourage questions and follow-up to keep the flow natural.
        - Charming Expressions
            - Clever wordplay or vivid descriptions: Use creative language to paint a picture.
            - Expressive gestures or facial cues: Highlight physical reactions to amplify humor or empathy.
        - Closeness and Trust
            - Familiar “best friend” vibe: Display comfort in each other's humor and habits.
            - Mutual support: Foster an environment where sharing is easy, with no fear of judgment.
    6. Whenever a character appears in dialogue, always include their name. Create the conversation primarily using first names.
    <image description instruction>
    1. The image descriptions represent photos stored on each individual's phone, which might include landscapes or portraits, along with descriptions of the pictures. The format is: <turn><landscapes or people's names(specifically, include the name of individuals in portraits)> <the date when the photo was taken> <image description>. When adding people, include words in the image description that indicate their relationship to the speaker. If the photo is of the sender, Add "me" to the name of the person field.
    Examples: 
    <turn><me><2024-08-12 6:45 PM><A peaceful photo of me sitting on a wooden bench by the lake, watching the sunset with soft ripples on the water>
    <turn><me, Liam><2023-07-15 11:30 AM><A warm family photo taken during our summer vacation at the beach, featuring me, Liam, the father>
    <turn><me, Sarah, and David><2023-03-14 3:15 PM><A fun group photo from our trip to the amusement park, all of us holding cotton candy. Sarah is my friend, and David is my boyfriend.>
    2. During the conversation, participants should carefully consider whether the photos are landscapes or portraits when responding. During conversations involving photos, the following examples can also be applied: 1. Including a description in the conversation that is only possible through the image, 2. Adding inferences or discoveries that can be deduced from viewing the image, and 3. Incorporating interactions or requests based on the image.
    3. For photo-sharing turns, include the word "turn" followed by the description of the shared photo.
    4. Based on the given persona, identify the type of person who frequently shares photos and tailor the photos you send to match their personality. Additionally, participants should ensure an equal balance between the number of landscape and portrait photos when sharing images.
    5. When writing the image descriptions for the people in the photo, ensure their details are explicitly included in the image description, even if they are not clearly mentioned in the conversation. This is mandatory and should be followed strictly.
    6. The name of the person in the photo must be included without exception. It is not permitted to use nouns such as "parents," "people," "mom," "dad," or "partner" instead of specific names, and this rule must be strictly adhered to 
    7. In cases such as Relational Nouns, Referential Nouns, or Terms of Address, which are not specific names but rather forms of address, they must not be used in the name field and should instead be included in the image description
    The following are examples of incorrect usage.
    Example : 
    Alex:<turn><people><2022-09-02 4:30 PM><A friendly group photo from the language exchange meetup at the Austin Public Library, with new friends Jorge, Mei, and Hana, all smiling in front of a bookshelf.>

    <photo instruction>
    Photos should be shared naturally and seamlessly, just like in real human interactions. Each speaker may send only 2-3 photos during the conversation. They don't need to be tied to specific requests or triggers but can be used to shift topics, highlight themes, or draw attention to a subject. As long as it fits the flow of the conversation, sharing photos without explicitly mentioning that a photo will be sent is perfectly acceptable. For instance, photos can be used to make requests, share experiences, evoke emotions, reminisce, or prompt compliments.
    Use the following five intents to guide photo-sharing behavior:
    - Information Dissemination: Sharing images like infographics or educational material to convey important information or educate.
    - Social Bonding: Sharing personal photos or memories to strengthen relationships and deepen connections.
    - Humor and Entertainment: Sharing funny pictures, memes, or entertaining visuals to bring joy and lighten the mood.
    - Visual Clarification: Using images such as diagrams, item-specific photos, or location pictures to explain or clarify concepts or situations.
    - Expression of Emotion or Opinion: Sharing emotive photos or art to succinctly express feelings, perspectives, or opinions.

    If the conversation with each person is completed, separate it with two line breaks \n\n. Similarly, for the photo turn, proceed and separate it with two line breaks \n\n. And based on the example below, carry out the conversation accurately, especially ensuring that the explanation for the turn involving sending a photo is written precisely in the correct format.
    <Example> 
    Liam: Hey Livy, how's your morning been? Are you out in the garden again, tending to your little green kingdom?\n\nOlivia: Ace, you know me too well. Just finished watering the hydrangeas. They're blooming beautifully this year. I should show you—\n\nOlivia:<turn><landscape><2022-03-18 7:45 AM><A vibrant shot of blooming hydrangeas with morning sunlight filtering through.>\n\nLiam: Wow, Livy, those are stunning! The sunlight makes them look almost magical. Your green thumb amazes me every time.\n\nOlivia: Thank you, darling. Oh, and here's something you'll love—Emily was over this morning and insisted on grabbing coffee before helping me with the garden.\n\nOlivia:<turn><me,Emily><2022-03-18 10:15 AM><A candid photo of Olivia’s daughter, Emily, and me, smiling brightly while holding cups of coffee at a cozy café.>\n\nLiam: She looks so grown-up! Time really flies. You must be proud of her. And speaking of memories, I found this gem while sorting through my photos. Vermont feels like a lifetime ago now.\n\nLiam:<turn><landscape><2021-10-12 2:30 PM><A cozy countryside scene with a wooden cabin, a fireplace chimney puffing smoke, and autumn leaves carpeting the ground.>\n\nOlivia: That trip was unforgettable. The autumn leaves, the crisp air, and your obsession with finding the perfect maple syrup—it's one for the books. Speaking of cherished memories, let me share this one. It's from last month when Emily, James, and I went hiking. James brought his camera, so we captured a few gems.\n\nOlivia:<turn><Emily,James,me><2024-12-14 9:20 AM><A lively group photo of Emily, James, and me standing on a rocky trail, surrounded by towering pine trees, with a snow-capped mountain in the background.>'''

            prompt2 = f'''
        Now, based on the above instructions, generate a conversation. When reviewing the conversation, if a relevant turn involves a portrait, the name must always be specified rather than left unspecified. For example, instead of using collective nouns like "family," "partner," or "kittens," you must include the names of every individual in the group. If there is a group among the people included in the photo, list the names of all individuals in the group.
        date : {date}
        relationship : {relationship}
        relationship_specify: {two_relationship}

        <{speaker1_information['name']}'s profile>
        age : {speaker1_information['age']} 
        nationality : {speaker1_information['nationality']} 
        gender : {speaker1_information['gender']}
        occupation : {speaker1_information['occupation']} 
        role in relationship : {speaker1_information['role']}
        persona : {speaker1_persona}

        <{speaker2_information['name']}'s profile>
        age : {speaker2_information['age']} 
        nationality : {speaker2_information['nationality']} 
        gender : {speaker2_information['gender']}
        occupation : {speaker2_information['occupation']} 
        role in relationship : {speaker2_information['role']}
        persona : {speaker2_persona}

        dialogue :
            '''
            dialogue = get_response(prompt1, prompt2, model_name, API_KEY)

            ### 대화 생성 저장
            lines = speaker1_persona.strip().split('\n')

            # 3) 각 줄에서 맨 앞 '-' 제거 후 양쪽 공백 제거
            speaker1_persona_list = [line.lstrip('-').strip().strip('\"') for line in lines if line.strip()]
            lines = speaker2_persona.strip().split('\n')

            # 3) 각 줄에서 맨 앞 '-' 제거 후 양쪽 공백 제거
            speaker2_persona_list = [line.lstrip('-').strip().strip('\"') for line in lines if line.strip()]

            information[f"{speaker1_information['name']}'s persona"] = speaker1_persona_list
            information[f"{speaker2_information['name']}'s persona"] = speaker2_persona_list
            information['relationship'] = relationship
            information['relationship+'] = two_relationship 
            information['date'] = date
            information['dialogue'] = transform_single_dialogue(dialogue)
            information['all_dia'] = dialogue

            data['speaker1'] = speaker1_information
            data['speaker2'] = speaker2_information
            data['session_1'] = information

            name.append(speaker1_information['name'])
            name.append(speaker2_information['name'])
        ### Session 2부터 생성하기. 
            for num in range(2,6):
                date = date_list[int(num)-1]
                print(f'session {num} start')
                
                information = {}
                

                ### Persona 업데이트 하기.
                updated_speaker1_persona = persona_update(speaker1_information['name'], speaker1_persona, dialogue, model_name, API_KEY)
                speaker1_updated_persona, speaker1_modified_personas, speaker1_newly_added_personas = process_personas(updated_speaker1_persona)
                updated_speaker2_persona = persona_update(speaker2_information['name'], speaker2_persona, dialogue, model_name, API_KEY)
                speaker2_updated_persona, speaker2_modified_personas, speaker2_newly_added_personas = process_personas(updated_speaker2_persona)
                summary = summarize(speaker1_information['name'], speaker2_information['name'], speaker1_persona, speaker2_persona, speaker1_information, speaker2_information, relationship ,two_relationship ,date ,dialogue, model_name, API_KEY)
                dialogue = sesssion_n_dialogue(speaker1_information['name'], speaker2_information['name'], speaker1_updated_persona, speaker2_updated_persona, speaker1_information, speaker2_information ,summary, date, relationship, two_relationship ,model_name, API_KEY)
                
                
                information['dialogue'] = transform_single_dialogue(dialogue)
                information['all_dia'] = dialogue
                information['summary'] = summary
                information['date'] = date
                information[f"{speaker1_information['name']}'s persona"] = speaker1_updated_persona 
                information[f"{speaker2_information['name']}'s persona"] = speaker2_updated_persona
                information[f"{speaker1_information['name']}'s modified_persona"] = speaker1_modified_personas
                information[f"{speaker2_information['name']}'s modified_persona"] = speaker2_modified_personas       
                information[f"{speaker1_information['name']}'s newly_added_persona"] = speaker1_newly_added_personas
                information[f"{speaker2_information['name']}'s newly_added_persona"] = speaker2_newly_added_personas  
                
                data[f'session_{num}'] = information
            data_all.append(data)
        except:
            pass

    output_file_path = '/home/chanho/Model/photo-sharing/pre_code/dataset/data_v10.json'
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        # data_all 리스트 전체를 한 번에 저장
        json.dump(data_all, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
